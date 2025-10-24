"""
FastAPI entrypoints
-------------------
Endpoints:
- POST /explain -> {nodes, edges, summary, confidence_global, evidence, facts}
- GET /health -> {status: "ok"}
- GET /sources/{node_id} -> placeholder (return sources for a node)
"""

from typing import Any, Dict, Optional, List
from datetime import date
import uuid
import hashlib
from urllib.parse import urlparse
import logging

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, model_validator
from fastapi.middleware.cors import CORSMiddleware

from core.curiosity_core import CuriosityCore
from utils.logger import get_logger
from utils.config import Settings

# Live ingest (Google + simple fetch/parse/index) deps

from ingest.google_search import search_google
from ingest.indexer import index_bulk
from ingest.types import PackagedDoc

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# FastAPI App Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Common Sense Engine API",
    version="1.1",
    description="Conversational causal reasoning engine powered by Elasticsearch + Vertex AI.",
)

# Allow Streamlit / local testing
app.add_middleware(
    CORSMiddleware,
    # NOTE: allow_credentials must not be True with wildcard origins in browsers.
    # For dev, keep '*' and set allow_credentials to False. In prod, restrict origins and flip it to True if needed.
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate the main engine
core = CuriosityCore()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TimeRange(BaseModel):
    # Pydantic v2: parse to date, keep API field name "from"
    from_: Optional[date] = Field(default=None, alias="from", description="Start date (inclusive)")
    to: Optional[date] = Field(default=None, description="End date (inclusive)")

    @model_validator(mode="after")
    def _check_order(self) -> "TimeRange":
        if self.from_ and self.to and self.from_ > self.to:
            raise ValueError("time_range.from must be <= time_range.to")
        return self

class LiveBudget(BaseModel):
    """Budget pour l’ingestion live (Google + fetch)."""
    max_docs: int = Field(default=5, ge=1, le=20, description="Nombre max d'URLs à ingérer en live")
    max_ms: int = Field(default=4000, ge=500, le=20000, description="Budget temps indicatif (ms)")

class ExplainRequest(BaseModel):
    query: str = Field(..., description="User question, e.g., 'Pourquoi le prix du riz augmente ?'")
    time_range: Optional[TimeRange] = Field(
        default=None, description='Example: {"from": "2020-01-01", "to": "2023-12-31"}'
    )
    k: int = Field(default=20, ge=1, le=100, description="Top documents to retrieve")
    live: bool = Field(default=True, description="Activer l’ingestion web live (Google) si configurée")
    live_budget: Optional[LiveBudget] = Field(default=None, description="Budget pour l’ingestion live")

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, Any]:
    """
    Health endpoint providing detailed ElasticSearch status.
    Returns JSON: {status, es_ok, index_exists, embedding_enabled}.
    """
    from utils.config import Settings
    s = Settings()
    es_ok = False
    index_exists = False
    embedding_enabled = False

    try:
        from elasticsearch import Elasticsearch
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Helper pour construire la connexion proprement
        def connect_elasticsearch(settings):
            """Crée une connexion Elasticsearch robuste, avec support HTTPS auto-signé."""
            es_url = settings.ES_ENDPOINT
            if not es_url.startswith(("http://", "https://")):
                es_url = f"http://{es_url}"  # fallback simple

            # 1️⃣ Authentification par API Key
            if getattr(settings, "ES_API_KEY", None):
                return Elasticsearch(
                    hosts=[es_url],
                    api_key=settings.ES_API_KEY,
                    verify_certs=False,
                    request_timeout=10,
                )

            # 2️⃣ Authentification classique (elastic/password)
            elif settings.ES_USER and settings.ES_PASSWORD:
                return Elasticsearch(
                    hosts=[es_url],
                    basic_auth=(settings.ES_USER, settings.ES_PASSWORD),
                    verify_certs=False,
                    request_timeout=10,
                )

            # 3️⃣ Connexion sans sécurité
            else:
                return Elasticsearch(
                    hosts=[es_url],
                    verify_certs=False,
                    request_timeout=10,
                )

        # Connexion à Elasticsearch
        es = connect_elasticsearch(s)

        # Vérification de disponibilité
        es_ok = False
        try:
            es_ok = bool(es.ping())
        except Exception as e:
            logger.warning(f"Elasticsearch ping failed: {e}")
            es_ok = False

        if es_ok:
            logger.info(f"✅ Connected to Elasticsearch at {s.ES_ENDPOINT}")

            # Vérifie si l’index existe et si le mapping est correct
            index_exists = es.indices.exists(index=s.ES_INDEX)
            try:
                mapping = es.indices.get_mapping(index=s.ES_INDEX)
                props = mapping[s.ES_INDEX]["mappings"]["properties"]
                embedding_enabled = (
                    "embedding" in props
                    and props["embedding"].get("type") == "dense_vector"
                )
            except Exception:
                embedding_enabled = False
        else:
            logger.error(f"❌ Unable to connect to Elasticsearch at {s.ES_ENDPOINT}")
            index_exists = False
            embedding_enabled = False

    except Exception as e:
        logger.exception("Failed to initialize Elasticsearch client")
        es = None
        es_ok = False
        index_exists = False
        embedding_enabled = False

    return {
        "status": "ok",
        "es_ok": es_ok,
        "index_exists": index_exists,
        "embedding_enabled": embedding_enabled,
    }


# ---------------------------------------------------------------------------
# Live ingest helper (Google -> fetch+parse -> index)
# ---------------------------------------------------------------------------

def _live_ingest_google(query: str, budget: LiveBudget, request_id: str) -> List[str]:
    """
    Pipeline minimal d’ingestion live :
      1) Recherche Google (Programmable Search)
      2) Fetch & parse (Trafilatura + fallback Readability)
      3) Packaging PackagedDoc
      4) Indexation (Elasticsearch) via ingest.indexer.index_bulk

    Retourne la liste des IDs indexés.
    """
    s = Settings()
    if not s.GOOGLE_ENABLED:
        logger.info("Live ingest skipped: Google not configured.")
        return []

    # 1) Google search
    try:
        urls = search_google(query, s.GOOGLE_API_KEY, s.GOOGLE_CX, n_results=budget.max_docs)
        # De-dupe, keep order
        seen = set()
        urls = [u for u in urls if not (u in seen or seen.add(u))]
        logger.info(f"[live] Google returned {len(urls)} url(s).", extra={"request_id": request_id})
    except Exception as e:
        logger.warning(f"[live] Google search failed: {e}")
        return []

    if not urls:
        return []

    # 2) Fetch + parse texte
    try:
        import trafilatura
        from readability import Document
        from bs4 import BeautifulSoup
        from datetime import datetime
    except Exception as e:
        logger.warning(f"[live] Parser libs missing: {e}")
        return []

    def clean_text(t: str) -> str:
        """Nettoyage des textes pour enlever navigation et pub."""
        lines = t.splitlines()
        keep = [
            l.strip() for l in lines
            if l.strip() and not any(bad in l.lower() for bad in [
                "cookies", "connexion", "newsletter", "derniers articles",
                "accueil", "inscrivez-vous", "abonnez-vous", "mentions légales",
                "abonnement", "partager cet article", "vos commentaires"
            ])
        ]
        return " ".join(keep)

    packs: List[PackagedDoc] = []
    qhash = hashlib.md5((query or "").strip().lower().encode("utf-8")).hexdigest()
    for u in urls[: budget.max_docs]:
        try:
            # Compatibilité versions trafilatura (timeout optionnel selon versions)
            try:
                dl = trafilatura.fetch_url(u, timeout=8)  # respecte robots.txt
            except TypeError:
                dl = trafilatura.fetch_url(u)
            if not dl:
                continue

            # 🧠 Extraction principale avec Trafilatura
            txt = trafilatura.extract(
                dl,
                include_tables=False,
                include_comments=False,
                favor_recall=True,
                include_formatting=False,
            )

            # ⚙️ Fallback avec Readability + BeautifulSoup si texte trop court
            if not txt or len(txt.strip()) < 400:
                try:
                    doc = Document(dl)
                    html_clean = doc.summary()
                    soup = BeautifulSoup(html_clean, "html.parser")
                    txt = soup.get_text(separator=" ", strip=True)
                except Exception as e:
                    logger.debug(f"[live] Fallback readability failed for {u}: {e}")
                    txt = None

            if not txt or len(txt.strip()) < 300:
                continue

            txt = clean_text(txt)

            # 🏷️ Métadonnées
            meta = trafilatura.extract_metadata(dl, url=u) or {}
            title = None
            try:
                title = getattr(meta, "title", None) or (meta.get("title") if isinstance(meta, dict) else None)
            except Exception:
                title = None
            if not title:
                title = urlparse(u).path.rsplit("/", 1)[-1] or urlparse(u).netloc

            pub = None
            try:
                pub = getattr(meta, "date", None) or (meta.get("date") if isinstance(meta, dict) else None)
            except Exception:
                pub = None

            publisher = None
            lang = None
            try:
                publisher = getattr(meta, "sitename", None) or (meta.get("sitename") if isinstance(meta, dict) else None)
                lang = getattr(meta, "language", None) or (meta.get("language") if isinstance(meta, dict) else None)
            except Exception:
                pass

            # 3) Package → PackagedDoc
            doc_id = hashlib.md5(u.encode("utf-8")).hexdigest()
            packs.append(
                PackagedDoc(
                    id=doc_id,
                    title=title or u,
                    content=txt.strip(),
                    date=pub,
                    source_url=u,
                    tags=[],
                    meta={
                        "publisher": publisher or urlparse(u).netloc,
                        "lang": lang,
                        "live": True,
                        "request_id": request_id,
                        "query_hash": qhash,
                    },
                )
            )
        except Exception as e:
            logger.debug(f"[live] Failed to parse {u}: {e}")

    if not packs:
        return []

    # 4) Index
    stats = index_bulk(packs)
    logger.info("[live] Index stats", extra={"stats": stats, "request_id": request_id})
    return [p.id for p in packs]

# ---------------------------------------------------------------------------
# Normalisation de la réponse /explain pour l'UI
# ---------------------------------------------------------------------------
def _stable_id_from_url(url: str) -> str:
    """ID déterministe pour une URL de source."""
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def _normalize_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    - Garantit la présence de `summary`, `confidence_global`, `mode`, `retrieval_mode`
    - Filtre `facts` pour ne garder que les sources réellement utilisées
      (citées dans `evidence` ou référencées par `nodes[*].sources`)
    """
    res: Dict[str, Any] = dict(raw or {})

    # 1) Clés attendues par l'UI
    res["summary"] = (
        raw.get("summary")
        or raw.get("synthesis")
        or raw.get("explanation")
        or raw.get("answer")
        or ""
    )
    res["confidence_global"] = raw.get("confidence_global") or raw.get("confidence") or 0.0
    res.setdefault("mode", raw.get("mode") or "template")
    res["retrieval_mode"] = raw.get("retrieval_mode") or "hybrid"

    # 2) Evidence : harmoniser la clé URL
    evidence = raw.get("evidence") or []
    # Utiliser uniquement le top des évidences pour éviter les fuites "long tail"
    TOPK_EVIDENCE = 20
    evidence_top = evidence[:TOPK_EVIDENCE]
    for ev in evidence_top:
        # Compat: mapper url -> source_url si nécessaire
        if "source_url" not in ev:
            if "url" in ev and ev.get("url"):
                ev["source_url"] = ev.get("url")
            elif "source" in ev and ev.get("source"):
                ev["source_url"] = ev.get("source")
    res["evidence"] = evidence_top

    # 3) Déterminer les sources réellement utilisées
    used_urls = set()
    evidence_ids = set()

    # a) URLs depuis l'évidence
    for ev in evidence_top:
        url = ev.get("source_url") or ev.get("url")
        if url:
            used_urls.add(url)

    # b) Doc IDs vus dans l'évidence
    for ev in evidence_top:
        if ev.get("doc_id"):
            evidence_ids.add(ev["doc_id"])

    # 4) Filtrer/normaliser les facts (seulement celles utilisées dans l'évidence)
    facts = raw.get("facts") or []
    normalized_facts: List[Dict[str, Any]] = []
    seen_fact_ids = set()

    for f in facts:
        url = f.get("source_url") or f.get("url")
        fid = f.get("id")
        if not fid and url:
            fid = _stable_id_from_url(url)
            f["id"] = fid

        # ✅ Garder uniquement si l'URL est citée dans l'évidence
        #    ou si le doc_id figure dans l'évidence
        if ((url in used_urls) or (fid in evidence_ids)) and fid not in seen_fact_ids:
            f.setdefault("title", url or "Document")
            normalized_facts.append(f)
            seen_fact_ids.add(fid)

    # Fallback : si rien filtré mais on a de l'évidence, construire depuis evidence
    if not normalized_facts and evidence_top:
        dedup = set()
        for ev in evidence_top:
            url = ev.get("source_url")
            if url and url not in dedup:
                dedup.add(url)
                normalized_facts.append({
                    "id": _stable_id_from_url(url),
                    "title": url.split("/")[-1] or url,
                    "source_url": url,
                })

    res["facts"] = normalized_facts
    res["nodes"] = raw.get("nodes") or []
    res["edges"] = raw.get("edges") or []
    # Pass-through des es_records si fournis par le CausalityMapper
    if raw.get("es_records"):
        res["es_records"] = raw.get("es_records")
    return res

@app.post("/explain")
async def explain(req: ExplainRequest, response: Response) -> Dict[str, Any]:
    """Run the full causal explanation pipeline.

    Calls CuriosityCore.explain() and returns the structured explanation
    containing:
    - nodes, edges
    - summary, confidence_global
    - evidence, facts
    - optional followup_suggestions
    """
    if not isinstance(req.query, str) or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(req.query.strip()) < 3:
        # Fast validation to avoid meaningless queries
        raise HTTPException(status_code=422, detail="Query is too short (min 3 characters).")

    # Attach a per-request UUID for tracing (also added as response header)
    request_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id

    try:
        # Normalize time_range to ISO strings for downstream components (ES expects strings)
        time_range_payload: Optional[Dict[str, Optional[str]]] = None
        if req.time_range:
            time_range_payload = {
                "from": req.time_range.from_.isoformat() if req.time_range.from_ else None,
                "to": req.time_range.to.isoformat() if req.time_range.to else None,
            }

        logger.info("POST /explain", extra={"query": req.query, "request_id": request_id})

        # ✅ Utiliser le pipeline complet (CuriosityCore → Orchestrator → FactFinder/Causality/Synthesizer)
        #    => produit summary + nodes/edges + evidence cohérente
        result_raw = core.explain(
            req.query,
            time_range=time_range_payload,  # dict {"from": "...", "to": "..."} ou None
            k=req.k,
            live=req.live,
            live_budget=(
                req.live_budget.model_dump() if req.live_budget else None
            ),
            request_id=request_id,
        )

        result = _normalize_result(result_raw)
        result["request_id"] = request_id
        # Assurer quelques champs attendus par l'UI
        # Forcer un défaut "hybrid" si non fourni par l'orchestrateur
        result["retrieval_mode"] = result.get("retrieval_mode") or "hybrid"
        # Pass-through es_records si présents (debug / indexation externe)
        if result_raw.get("es_records") and not result.get("es_records"):
            result["es_records"] = result_raw["es_records"]
            
        # Debug non bloquant : remonter l’info d’ingestion live depuis FactFinder
        if result.get("live_ingested"):
            lids = result.get("live_ingested") or []
            if lids:
                result.setdefault("debug", {})["live_ingest"] = {
                    "count": len(lids),
                    "doc_ids": lids[:10],
                }

        # Stocker le dernier payload pour /sources
        app.state.last_payload = result

        return result

    except HTTPException:
        # Pass-through deliberate HTTP errors
        raise
    except Exception as e:
        logger.exception("Pipeline failed", extra={"request_id": request_id})
        # Don't leak internal details to clients
        raise HTTPException(status_code=500, detail="Internal error. Please try again later.")


@app.get("/sources/{node_id}")
async def sources(node_id: str) -> Dict[str, Any]:
    """
    Returns a list of source URLs associated with the given node_id
    from the last /explain payload.
    """
    payload = getattr(app.state, "last_payload", None)

    urls: list[str] = []
    if isinstance(payload, dict):
        id2url = {
            f.get("id"): f.get("source_url")
            for f in (payload.get("facts") or [])
            if f.get("id") and f.get("source_url")
        }

        # Seules les URLs présentes dans facts filtrés sont autorisées
        allowed_urls = set(id2url.values())

        for n in (payload.get("nodes") or []):
            if n.get("id") == node_id:
                for src in n.get("sources", []):
                    url = None
                    if isinstance(src, str):
                        # si c'est un id connu → mappe vers son URL
                        if src in id2url:
                            url = id2url[src]
                        # si c'est déjà une URL, ne l'accepte que si elle est dans facts filtrés
                        elif src.startswith("http") and src in allowed_urls:
                            url = src
                    if url and url not in urls:
                        urls.append(url)

    return {"node_id": node_id, "sources": urls[:5]}