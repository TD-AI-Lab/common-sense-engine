"""
FactFinder — Step 1 of the Common Sense Engine pipeline
-------------------------------------------------------
Retrieves hybrid evidence (lexical + vector) from ElasticSearch.

Live mode (ingestion web) :
- Si `input_data["live"] == True` **et** Google Custom Search est configuré,
  alors on lance une courte recherche Google, on récupère les pages, on les parse,
  on les indexe dans ES (refresh), puis on exécute la recherche hybride.
- Sinon, si la recherche ES ne renvoie rien (ou trop peu) **et** Google est configuré,
  on tente automatiquement une ingestion live « best-effort », puis on relance la recherche ES.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import uuid
import re
import hashlib
import html
import time
from urllib.parse import urlencode, urlparse
from datetime import datetime

from utils.logger import get_logger
from utils.metrics import timed
from utils.config import Settings

from elasticsearch import Elasticsearch
from typing import Tuple
import httpx

from modules.embedder import LocalEmbedder
import numpy as np

logger = get_logger(__name__)

# --------------------- Query tokenization (topic fallback) ---------------------
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+")
STOPWORDS = {
    # FR
    "le","la","les","des","de","du","d","un","une","et","ou","au","aux","en","dans","avec","sur","pour",
    "est","sont","plus","moins","que","qui","quoi","quand","pourquoi","comment","ce","cette","ces","aujourd",
    # EN
    "the","a","an","and","or","of","to","in","on","for","by","with","is","are","was","were","from","at",
    "why","how","what","when","which","this","that","these","those"
}

@dataclass
class Evidence:
    doc_id: str
    text: str
    span: list[int] | None = None
    date: str | None = None
    url: str | None = None
    # Conserve pour compat ascendante si d'autres modules l'utilisent encore
    source_url: str | None = None
    score: float = 0.0


class FactFinder:
    """Hybrid retrieval agent (lexical + vector) using ElasticSearch.
    The vector part uses Vertex AI Embeddings if configured (VERTEX_PROJECT / VERTEX_LOCATION).
    """

    # Minimal mapping sufficient for P0 (embedding field optional)
    _DEFAULT_MAPPING: Dict[str, Any] = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"dynamic": True, "properties": {
            "title": {"type": "text"}, "content": {"type": "text"},
            "date": {"type": "date", "format": "strict_date||date_optional_time"}, "source_url": {"type": "keyword"}, "tags": {"type": "keyword"}}}}

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        # 🔄 Auto-enable Google si clés présentes (prend le pas sur GOOGLE_ENABLED)
        self.google_enabled: bool = bool(
            getattr(self.settings, "GOOGLE_API_KEY", None)
            and getattr(self.settings, "GOOGLE_CX", None)
            or getattr(self.settings, "GOOGLE_ENABLED", False)
        )
        # Nom d'index utilisé partout (ensure_index, indexation, recherche)
        self.index_name: str = (self.settings.ES_INDEX or "facts_index").strip() or "facts_index"
        self.embedder = LocalEmbedder()
        self.vector_field = "embedding"
        # Banned phrases (optional, safe default = empty)
        try:
            raw_banned = getattr(self.settings, "RETRIEVAL_BANNED", [])
            if isinstance(raw_banned, str):
                self.banned = [t.strip() for t in raw_banned.split(",") if t.strip()]
            elif isinstance(raw_banned, (list, tuple, set)):
                self.banned = [str(t).strip() for t in raw_banned if str(t).strip()]
            else:
                self.banned = []
        except Exception:
            self.banned = []

        # --- Elasticsearch client (local or cloud) ---
        try:
            # ✅ On tente d’utiliser le connecteur centralisé d’abord
            try:
                from utils.elastic import get_elastic_client
                self.client = get_elastic_client()
                self.client.info()
                logger.info(
                    "Connected to Elasticsearch via get_elastic_client()",
                    extra={"endpoint": self.settings.ES_ENDPOINT, "index": self.index_name},
                )
            except Exception as e_inner:
                logger.warning(f"get_elastic_client() unavailable, using local connect: {e_inner}")
                self.client, err = self._connect_es(self.settings)
                if not self.client:
                    raise RuntimeError(err or "Unknown ES connection error")
        except Exception as e:
            logger.error(f"Elastic connection failed: {e}")
            self.client = None

        # --- Vertex Embeddings availability flag ---
        self.vertex_ready = False
        try:
            # Lazy import flags; we will init on demand in _embed_text
            import vertexai  # noqa: F401
            from vertexai.language_models import TextEmbeddingModel  # noqa: F401
            self.vertex_ready = bool(self.settings.VERTEX_PROJECT and self.settings.VERTEX_LOCATION)
            if self.vertex_ready:
                logger.info("Vertex AI Embeddings available (will be used for vector search).")
            else:
                logger.info("Vertex AI not configured (falling back to lexical-only).")
        except Exception:
            logger.info("Vertex AI libraries not installed — lexical-only mode.")

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _query_hash(self, q: str) -> str:
        qn = (q or "").strip().lower()
        return hashlib.md5(qn.encode("utf-8")).hexdigest()

    def _merge_meta(self, doc: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        m = dict(doc.get("meta") or {})
        m.update({k: v for k, v in extra.items() if v is not None})
        doc["meta"] = m
        return doc

    def _normalize_endpoint(self, url: str) -> str:
        if not url:
            return "http://localhost:9200"
        if not url.startswith(("http://", "https://")):
            return "http://" + url
        return url

    def _connect_es_once(self, settings: Settings, endpoint: str) -> Tuple[Optional[Elasticsearch], Optional[str]]:
        try:
            kwargs = dict(verify_certs=False, request_timeout=30)
            if getattr(settings, "ES_API_KEY", None):
                client = Elasticsearch(endpoint, api_key=settings.ES_API_KEY, **kwargs)
                auth_mode = "api_key"
            elif settings.ES_USER and settings.ES_PASSWORD:
                client = Elasticsearch(endpoint, basic_auth=(settings.ES_USER, settings.ES_PASSWORD), **kwargs)
                auth_mode = "basic"
            else:
                client = Elasticsearch(endpoint, **kwargs)
                auth_mode = "anon"
            client.info()  # raise if unreachable
            logger.info(f"Using Elasticsearch auth={auth_mode}", extra={"endpoint": endpoint})
            return client, None
        except Exception as e:
            return None, str(e)

    def _connect_es(self, settings: Settings) -> Tuple[Optional[Elasticsearch], Optional[str]]:
        endpoint = self._normalize_endpoint(settings.ES_ENDPOINT)
        client, err = self._connect_es_once(settings, endpoint)
        if client:
            return client, None
        # Retry in HTTPS if HTTP fails (ES8 default)
        parsed = urlparse(endpoint)
        if parsed.scheme == "http":
            https_endpoint = "https://" + parsed.netloc + (parsed.path or "")
            client2, err2 = self._connect_es_once(settings, https_endpoint)
            if client2:
                settings.ES_ENDPOINT = https_endpoint  # update for logs/next calls
                return client2, None
            return None, f"http_failed:{err} | https_failed:{err2}"
        return None, err

    def _reconnect_if_needed(self) -> None:
        """Try to (re)establish ES client if it's None."""
        if self.client is None:
            try:
                self.client, _ = self._connect_es(self.settings)
            except Exception:
                self.client = None

    def _ensure_index(self) -> None:
        """Create ES index with minimal mapping if missing."""
        if not self.client:
            return
        try:
            if not self.client.indices.exists(index=self.index_name):
                # --- Determine embedding dimension dynamically ---
                try:
                    emb_dim = len(self.embedder.embed(["probe"])[0])
                except Exception:
                    emb_dim = 384

                mapping = {
                    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                    "mappings": {
                        "dynamic": True,
                        "properties": {
                            "title": {"type": "text"},
                            "content": {"type": "text"},
                            "date": {"type": "date", "format": "strict_date||date_optional_time"},
                            "source_url": {"type": "keyword"},
                            "tags": {"type": "keyword"},
                            "meta": {
                                "properties": {
                                    "query_hash": {"type": "keyword"},
                                    "request_id": {"type": "keyword"},
                                    "ingested_at": {"type": "date", "format": "strict_date_optional_time||epoch_millis"}
                                }
                            },
                            "embedding": {
                                "type": "dense_vector",
                                "dims": emb_dim,
                                "index": True,
                                "similarity": "cosine"
                            },
                        },
                    },
                }
                self.client.indices.create(index=self.index_name, body=mapping)
                logger.info(
                    f"Created Elasticsearch index '{self.index_name}' with vector mapping (dims={emb_dim})"
                )
        except Exception as e:
            logger.warning(f"ensure_index failed: {e}")

    def _hash(self, s: str) -> str:
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    def _clean_text(self, txt: str) -> str:
        # normalize whitespace; keep readable newlines
        txt = html.unescape(txt or "")
        txt = re.sub(r"\r\n?", "\n", txt)
        txt = re.sub(r"[ \t]+", " ", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
        return txt

    def _extract_title(self, html_str: str) -> Optional[str]:
        m = re.search(r"<title[^>]*>(.*?)</title>", html_str, flags=re.I | re.S)
        if m:
            title = re.sub(r"\s+", " ", html.unescape(m.group(1)).strip())
            return title[:300] if title else None
        return None

    def _extract_date(self, html_str: str) -> Optional[str]:
        # Try common meta tags first
        patterns = [
            r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+name=["\']pubdate["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+name=["\']date["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+itemprop=["\']datePublished["\'][^>]+content=["\']([^"\']+)["\']',
        ]
        for pat in patterns:
            m = re.search(pat, html_str, flags=re.I)
            if m:
                val = m.group(1).strip()
                # keep ISO/date-ish strings, truncate if full datetime
                if "T" in val:
                    return val[:10] if len(val) >= 10 else val
                # fallback to simple YYYY-MM-DD if present
                m2 = re.search(r"\b\d{4}-\d{2}-\d{2}\b", val)
                if m2:
                    return m2.group(0)
                return val[:10]
        # last-resort regex anywhere in page
        m3 = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", html_str)
        return m3.group(1) if m3 else None

    def _strip_html(self, html_str: str) -> str:
        # crude fallback: remove scripts/styles then tags
        html_str = re.sub(r"(?is)<script.*?>.*?</script>", " ", html_str)
        html_str = re.sub(r"(?is)<style.*?>.*?</style>", " ", html_str)
        text = re.sub(r"(?s)<[^>]+>", " ", html_str)
        return self._clean_text(text)

    # ---------------------------------------------------------------------
    # Web parsing helpers
    # ---------------------------------------------------------------------

    def _parse_article(self, url: str, *, timeout: int = 8) -> Optional[Dict[str, Any]]:
        """Fetch URL and return {'id','title','content','date','source_url','tags'} or None on failure."""
        try:
            with httpx.Client(follow_redirects=True, timeout=timeout, headers={
                "User-Agent": "CSEBot/0.2 (+https://example.invalid/contact)",
                "Accept": "text/html,*/*",
            }) as client:
                r = client.get(url)
                if r.status_code >= 400 or not r.text:
                    logger.debug(f"fetch fail {url} => {r.status_code}")
                    return None
                html_str = r.text
        except Exception as e:
            logger.debug(f"fetch crash {url}: {e}")
            return None

        # Try trafilatura first (if available)
        title = self._extract_title(html_str) or url
        date = self._extract_date(html_str)
        content: Optional[str] = None
        try:
            import trafilatura  # type: ignore
            content = trafilatura.extract(html_str, include_comments=False, include_formatting=False)
        except Exception:
            content = None

        if not content:
            # Fallback: readability-lxml if present
            try:
                from readability import Document  # type: ignore
                doc = Document(html_str)
                title = (doc.short_title() or title)[:300]
                content = self._strip_html(doc.summary(html_partial=True))
            except Exception:
                content = None

        if not content:
            # Final crude fallback
            content = self._strip_html(html_str)

        content = (content or "").strip()
        if len(content) < 200:  # too short to be useful
            return None

        doc_id = self._hash(url)
        return {
            "id": doc_id,
            "title": title,
            "content": content,
            "date": date,
            "source_url": url,
            "tags": ["live", "NEW"],
        }

    def _index_docs(self, docs: List[Dict[str, Any]]) -> List[str]:
        """Index parsed docs into ES, return list of ids successfully written."""
        if not docs:
            return []
        if not self.client:
            self._reconnect_if_needed()
            if not self.client:
                return []
        self._ensure_index()
        ok: List[str] = []
        for d in docs:
            try:
                # Génère un embedding local pour le contenu du document
                if "content" in d and d["content"]:
                    try:
                        emb = self.embedder.embed([d["content"]])[0]
                        d["embedding"] = emb
                    except Exception as e:
                        logger.debug(f"embedding failed for {d.get('source_url')}: {e}")
                self.client.index(index=self.index_name, id=d["id"], document=d, refresh=False)
                ok.append(d["id"])
            except Exception as e:
                logger.debug(f"index fail {d.get('source_url')}: {e}")
        # Force a refresh so the new docs are immediately searchable for this request
        try:
            self.client.indices.refresh(index=self.index_name)
        except Exception:
            pass
        return ok

    def _live_ingest(
        self,
        query: str,
        *,
        max_docs: int = 5,
        timeout: int = 8,
        request_id: Optional[str] = None,
        query_hash: Optional[str] = None,
        max_ms: Optional[int] = None,
    ) -> List[str]:        
        """Search via Google CSE, fetch & parse pages, index to ES. Returns ingested doc ids."""
        key = getattr(self.settings, "GOOGLE_API_KEY", None)
        cx = getattr(self.settings, "GOOGLE_CX", None)
        if not (key and cx):
            logger.info("Live ingest skipped: GOOGLE_API_KEY/GOOGLE_CX not configured.")
            return []

        urls: List[str] = []
        # 1) Try the wrapper (googleapiclient)
        try:
            from ingest.google_search import search_google  # uses googleapiclient.discovery
            urls = search_google(query, key, cx, n_results=max_docs)
        except Exception as e:
            logger.info(f"googleapiclient unavailable or failed ({e}), using REST fallback.")
            # 2) Fallback: REST call to Google Custom Search API
            try:
                params = {"key": key, "cx": cx, "q": query, "num": min(10, max_docs)}
                api_url = "https://customsearch.googleapis.com/customsearch/v1?" + urlencode(params)
                with httpx.Client(timeout=8, headers={"Accept": "application/json"}) as client:
                    r = client.get(api_url)
                    r.raise_for_status()
                    data = r.json()
                    items = data.get("items", []) or []
                    urls = [it["link"] for it in items if "link" in it][:max_docs]
            except Exception as e2:
                logger.warning(f"Google CSE REST failed: {e2}")
                return []

        if not urls:
            return []

        parsed: List[Dict[str, Any]] = []
        deadline = (time.time() + (max_ms / 1000.0)) if (max_ms and max_ms > 0) else None
        for url in urls[:max_docs]:
            if deadline and time.time() > deadline:
                logger.info("Live ingest time budget reached; stopping early.")
                break
            art = self._parse_article(url, timeout=timeout)
            if art:
                parsed.append(
                    self._merge_meta(
                        art,
                        {
                            "live": True,
                            "request_id": request_id,
                            "query_hash": query_hash or self._query_hash(query),
                            "ingested_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        },
                    )
                )
        if not parsed:
            return []
        ids = self._index_docs(parsed)
        if ids:
            logger.info("Live ingest indexed docs", extra={"count": len(ids)})
        return ids
 
    def _sanitize_query(self, query: str) -> str:
        """Remove dangerous characters and normalize whitespace."""
        query = re.sub(r"[^\w\s\-']", " ", query)
        return re.sub(r"\s+", " ", query).strip()

    def _split_sentences(self, text: str) -> List[str]:
        """Very naive sentence splitter."""
        return [s.strip() for s in re.split(r"[.!?]\s+", text) if len(s.strip()) > 10]

    # --------------------- Topic tokens & fallback search ---------------------
    def _topic_tokens(self, query: str, *, min_len: int = 3, max_tokens: int = 8) -> list[str]:
        toks = [w.lower() for w in WORD_RE.findall(query or "")]
        toks = [t for t in toks if len(t) >= min_len and t not in STOPWORDS and not t.isdigit()]
        # garder l'ordre d'apparition et dédupliquer
        seen, out = set(), []
        for t in toks:
            if t not in seen:
                seen.add(t); out.append(t)
        return out[:max_tokens]

    def _fallback_topic_search(self, query: str, *, k: int, time_range: Optional[dict] = None) -> list[dict]:
        """Lexical fallback indépendant de meta.query_hash: contraint par tokens thématiques."""
        if not self.client:
            return []
        tokens = self._topic_tokens(query)
        if not tokens:
            return []
        try:
            body: Dict[str, Any] = {
                "size": max(50, k * 3),
                "query": {
                    "bool": {
                        "must": [
                            {"multi_match": {
                                "query": query,
                                "fields": ["title^2", "content"],
                                "operator": "and"
                            }}
                        ],
                        "should": (
                            [{"match_phrase": {"title": t}} for t in tokens] +
                            [{"match_phrase": {"content": t}} for t in tokens]
                        ),
                        "minimum_should_match": min(2, len(tokens)),
                        "must_not": (
                            [{"match_phrase": {"title": b}} for b in self.banned] +
                            [{"match_phrase": {"content": b}} for b in self.banned]
                        ),
                    }
                },
                "_source": ["title", "content", "date", "source_url", "tags", "meta"],
            }
            if time_range and (time_range.get("from") or time_range.get("to")):
                rng: Dict[str, Any] = {}
                if time_range.get("from"):
                    rng["gte"] = time_range["from"]
                if time_range.get("to"):
                    rng["lte"] = time_range["to"]
                body["query"]["bool"].setdefault("filter", []).append({"range": {"date": rng}})

            res = self.client.search(index=self.index_name, body=body)
            return res.get("hits", {}).get("hits", []) or []
        except Exception as e:
            logger.error(f"Fallback topic search failed: {e}")
            return []

    # ---------------------------------------------------------------------
    # Embedding helpers
    # ---------------------------------------------------------------------

    def _embed_text(self, text: str) -> Optional[List[float]]:
        """Compute embedding with Vertex AI (if configured)."""
        try:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel
            vertexai.init(project=self.settings.VERTEX_PROJECT, location=self.settings.VERTEX_LOCATION)
            model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
            emb = model.get_embeddings([text[:3072]])[0].values
            return [float(x) for x in emb]
        except Exception as e:
            logger.warning(f"Vertex embedding failed: {e}")
            try:
                return self.embedder.embed([text])[0]
            except Exception:
                return None

    # ---------------------------------------------------------------------
    # Core merging / evidence methods
    # ---------------------------------------------------------------------

    def _rrf_merge(
        self,
        lexical_hits: List[Dict[str, Any]],
        vector_hits: List[Dict[str, Any]],
        *,
        k: int = 60,
        rrf_k: int = 60,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Reciprocal Rank Fusion. Return (top_docs, fused_scores_by_id)."""
        rank_scores: Dict[str, float] = {}
        for rank, hit in enumerate(lexical_hits):
            rank_scores[hit["_id"]] = rank_scores.get(hit["_id"], 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, hit in enumerate(vector_hits):
            rank_scores[hit["_id"]] = rank_scores.get(hit["_id"], 0.0) + 1.0 / (rrf_k + rank + 1)

        # Sort by fused score desc
        fused = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)
        id_to_doc = {h["_id"]: h for h in (lexical_hits + vector_hits)}
        docs = []
        for _id, _score in fused:
            if _id in id_to_doc:
                doc = dict(id_to_doc[_id])  # shallow copy
                doc["_fused"] = float(_score)
                docs.append(doc)
            if len(docs) >= k:
                break
        return docs, rank_scores

    def _extract_evidence(self, doc: Dict[str, Any], query: str, max_sentences: int = 3, ev_score: float = 0.0) -> List[Evidence]:
        """Extract sentences containing key terms from doc content.
        ev_score doit être normalisé sur [0,1] (RRF-normalized)."""
        content = doc.get("_source", {}).get("content", "") or ""
        if not content:
            return []
        sentences = self._split_sentences(content)
        # basic term selection
        query_terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 3]
        matched = []
        for s in sentences:
            if any(term in s.lower() for term in query_terms):
                matched.append(s)
            if len(matched) >= max_sentences:
                break
        src_url = doc.get("_source", {}).get("source_url")
        return [
            Evidence(
                doc_id=doc["_id"],
                text=s,
                span=(lambda start: [start, start + len(s)])(
                    (content.find(s) if s in content else -1)
                ) if s and s in content else None,
                date=doc["_source"].get("date"),
                url=src_url,
                source_url=src_url,
                score=float(max(0.0, min(1.0, ev_score))),
            )
            for s in matched
        ]

    def _confidence_from_fused(self, fused_scores: Dict[str, float]) -> float:
        """Compute a simple global confidence in [0,1] from fused scores."""
        if not fused_scores:
            return 0.0
        vals = list(fused_scores.values())
        maxv = max(vals) or 1e-9
        # normalize by max and mean for stability
        meanv = sum(v / maxv for v in vals) / len(vals)
        return round(max(0.0, min(1.0, meanv)), 3)

    # ---------------------------------------------------------------------
    # Main process
    # ---------------------------------------------------------------------

    @timed("fact_finder.process")
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve hybrid evidence for `input_data["query"]`."""
        # ⚠️ Utiliser la requête BRUTE pour le hash (doit matcher l’ingest)
        raw_query = (input_data.get("query") or "").strip()
        if not raw_query:
            return {"facts": [], "evidence": [], "errors": ["Empty query"]}
        qhash = self._query_hash(raw_query)
        # Sanitize uniquement pour la recherche lexicale
        query = self._sanitize_query(raw_query)

        # --- Live ingest (Google CSE -> fetch/parse -> ES) ---
        live_ingested: List[str] = []
        live_requested: bool = bool(input_data.get("live") is True)
        try:
            self._reconnect_if_needed()
            if live_requested and self.google_enabled:
                budget = input_data.get("live_budget") or {}
                max_docs = int(budget.get("max_docs", 5))
                timeout = int(budget.get("timeout", 8))
                max_ms = int(budget.get("max_ms", 4000))
                logger.info("Live ingest requested", extra={"max_docs": max_docs})
                req_id = (input_data.get("request_id") or str(uuid.uuid4()))
                # Ingest avec la REQUÊTE BRUTE + hash déjà calculé
                live_ingested = self._live_ingest(
                    raw_query,
                    max_docs=max_docs,
                    timeout=timeout,
                    request_id=req_id,
                    max_ms=max_ms,
                    query_hash=qhash,
                )
                if live_ingested and self.client:
                    # petite attente de confort (refresh déjà fait)
                    time.sleep(0.2)
        except Exception as e:
            logger.warning(f"Live ingest failed (continuing): {e}")

        trace_id = str(uuid.uuid4())
        logger.info("FactFinder starting hybrid retrieval", extra={"query": query, "trace_id": trace_id})

        k = int(input_data.get("k", 20))
        time_range = input_data.get("time_range")
        retrieval_mode = "lexical_only"
        
        # S'assure que l'index existe avant toute recherche (évite index_not_found au 1er run)
        self._reconnect_if_needed()
        self._ensure_index()

        # --- Step 1: compute embedding (optional) ---
        embedding: Optional[List[float]] = None
        if self.vertex_ready:
            embedding = self._embed_text(query)
        else:
            # Utilise LocalEmbedder si Vertex AI est désactivé
            try:
                embedding = self.embedder.embed([query])[0]
                logger.info("Using local embedding for query vector similarity.")
            except Exception as e:
                logger.warning(f"Local embedding failed: {e}")

        # --- Step 2: Lexical search ---
        lexical_hits_filtered: List[Dict[str, Any]] = []
        if self.client:
            try:
                body: Dict[str, Any] = {
                    "size": max(50, k * 3),
                    "query": {
                        "bool": {
                            "must": [
                                {"multi_match": {"query": query, "fields": ["title^2", "content"]}}
                            ],
                            "filter": [
                                {"term": {"meta.query_hash": qhash}}
                            ],
                            # garde-fous “hors-sujet” basiques (configurables)
                            "must_not": (
                                [{"match_phrase": {"title": b}} for b in self.banned] +
                                [{"match_phrase": {"content": b}} for b in self.banned]
                            ),
                        }
                    },
                    "_source": ["title", "content", "date", "source_url", "tags", "meta"],
                }
                if time_range and (time_range.get("from") or time_range.get("to")):
                    rng: Dict[str, Any] = {}
                    if time_range.get("from"):
                        rng["gte"] = time_range["from"]
                    if time_range.get("to"):
                        rng["lte"] = time_range["to"]
                    body["query"]["bool"]["filter"].append({"range": {"date": rng}})

                res = self.client.search(index=self.index_name, body=body)
                lexical_hits_filtered = res.get("hits", {}).get("hits", []) or []
            except Exception as e:
                logger.error(f"Lexical search failed: {e}")

        # --- Auto live ingest (toujours dans l’espace de la requête) ---
        if (not live_requested) and self.google_enabled and len(lexical_hits_filtered) == 0:
            logger.info("Auto live ingest: zero filtered hits → triggering web discovery")
            self._reconnect_if_needed()
            live_ingested = self._live_ingest(
                raw_query,
                max_docs=5,
                timeout=8,
                query_hash=qhash,
                max_ms=4000,
            ) or live_ingested
            if live_ingested and self.client:
                time.sleep(0.2)
                try:
                    res = self.client.search(index=self.index_name, body=body)
                    lexical_hits_filtered = res.get("hits", {}).get("hits", []) or []
                except Exception:
                    pass

        # Si on a des hits confinés par query_hash, on les garde.
        # Sinon (pas d'ingestion possible ou vide), on tente un fallback thématique indépendant.
        lexical_hits = lexical_hits_filtered
        if len(lexical_hits) == 0:
            lexical_hits = self._fallback_topic_search(query, k=k, time_range=time_range)

        # --- Step 3: Vector search (if embedding & client) ---
        vector_hits: List[Dict[str, Any]] = []
        if embedding and self.client:
            try:
                # Requires a `dense_vector` field named "embedding" in index mapping.
                # ES 8.x: "knn" must be at the top-level of the body (not nested under "query").
                res = self.client.search(
                    index=self.index_name,
                    body={
                        "size": max(50, k * 3),
                        "knn": {
                            "field": "embedding",
                            "query_vector": embedding,
                            "k": max(50, k * 3),
                            "num_candidates": max(100, k * 6),
                        },
                        # Filtre post knn sur notre espace de requête
                        "query": {
                            "bool": {
                                "filter": [
                                    {"term": {"meta.query_hash": qhash}}
                                ],
                                "must_not": (
                                    [{"match_phrase": {"title": b}} for b in self.banned] +
                                    [{"match_phrase": {"content": b}} for b in self.banned]
                                ),
                            }
                        },
                        "_source": ["title", "content", "date", "source_url", "tags", "meta"],
                    },
                )
                vector_hits = res.get("hits", {}).get("hits", []) or []
            except Exception as e:
                logger.warning(f"Vector search skipped: {e}")

        # Decide retrieval mode based on actual vector usage
        used_vector = bool(embedding and vector_hits)
        retrieval_mode = "hybrid" if used_vector else "lexical_only"

        # --- Step 4: Fuse results (RRF) ---
        fused_docs, fused_scores = self._rrf_merge(lexical_hits, vector_hits, k=max(k, 20), rrf_k=60)
        # Si on a des docs marqués (ingestion live), restreindre à la requête courante.
        if any(((h.get("_source", {}) or {}).get("meta", {}) or {}).get("query_hash") == qhash for h in fused_docs):
            fused_docs = [
                d for d in fused_docs
                if ((d.get("_source", {}) or {}).get("meta", {}) or {}).get("query_hash") == qhash
            ]
        # (Le topic guard déjà présent continue de s’appliquer ensuite)
        max_fused = max(fused_scores.values()) if fused_scores else 1.0

        # --- Step 5: Extract evidence sentences ---
        evidence_all: List[Evidence] = []
        for doc in fused_docs[:k]:
            # Score normalisé [0,1] pour chaque doc (utilisé par le mapper)
            doc_norm = float(fused_scores.get(doc["_id"], 0.0)) / max_fused if max_fused > 0 else 0.0
            evidence_all.extend(self._extract_evidence(doc, query, max_sentences=3, ev_score=doc_norm))

        # --- Step 6: Normalize output ---
        facts = []
        for doc in fused_docs[:k]:
            src = doc.get("_source", {})
            facts.append(
                {
                    "id": doc["_id"],
                    "title": src.get("title", ""),
                    "content": (src.get("content", "") or "")[:800],
                    "date": src.get("date"),
                    "source_url": src.get("source_url"),
                    "tags": src.get("tags", []),
                    "score": float(doc.get("_fused") or doc.get("_score") or 0.0),
                }
            )

        confidence_global = self._confidence_from_fused(fused_scores)

        logger.info(
            "FactFinder finished",
            extra={
                "trace_id": trace_id,
                "docs_lexical": len(lexical_hits),
                "docs_vector": len(vector_hits),
                "docs_fused": len(fused_docs),
                "confidence": confidence_global,
                "mode": retrieval_mode,
            },
        )

        result = {
            "query": raw_query,
            "trace_id": trace_id,
            "retrieval_mode": retrieval_mode,
            "live_ingested": live_ingested,
            "facts": facts,
            "evidence": [e.__dict__ for e in evidence_all],
            "confidence_global": confidence_global,
        }

        # Debug utile pour l'UI / logs
        result["retrieval_stats"] = {"lexical": len(lexical_hits), "vector": len(vector_hits), "fused": len(fused_docs)}
        return result

# ---------------------------------------------------------------------
# Patch: fallback endpoint if not defined
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from utils.config import Settings
    s = Settings()
    if not getattr(s, "ES_ENDPOINT", None):
        s.ES_ENDPOINT = "http://localhost:9200"
    ff = FactFinder(s)
    print("Elastic ready:", ff.client is not None)
    print("Index:", ff.index_name)
    print("Vertex ready:", ff.vertex_ready)