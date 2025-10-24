"""
Bulk indexer for Elasticsearch (compatible with existing `data/index_data.py`).

Behavior:
- If the official `elasticsearch` client is available AND ES is reachable → index documents.
- Else, gracefully fall back to writing an NDJSON file `ingest_out.jsonl` in the CWD.

Why this fallback?
- To keep the pipeline *directement fonctionnel* even without ES in local dev.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import json
import os
from datetime import datetime
from urllib.parse import urlparse

# Lazy imports for optional deps
try:
    from elasticsearch import Elasticsearch, helpers  # type: ignore
except Exception:  # pragma: no cover
    Elasticsearch = None  # type: ignore
    helpers = None  # type: ignore

try:
    from utils.config import Settings  # your project should provide this
    from utils.elastic import get_elastic_client

except Exception:
    # Minimal fallback Settings if your utils are not available in this environment
    class Settings:  # type: ignore
        ES_ENDPOINT: str = os.environ.get("ES_ENDPOINT", "http://localhost:9200")
        ES_INDEX: str = os.environ.get("ES_INDEX", "facts_index")
        ES_USER: Optional[str] = os.environ.get("ES_USER") or None
        ES_PASSWORD: Optional[str] = os.environ.get("ES_PASSWORD") or None
        ES_API_KEY: Optional[str] = os.environ.get("ES_API_KEY") or None
        # Si défini, on crée le champ embedding dans le mapping
        ES_EMBEDDING_DIM: Optional[int] = (
            int(os.environ.get("ES_EMBEDDING_DIM")) if os.environ.get("ES_EMBEDDING_DIM") else None
        )

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

from .types import PackagedDoc


def _normalize_endpoint(url: str) -> str:
    """Ajoute un schéma si manquant, par défaut http://"""
    if not url.startswith(("http://", "https://")):
        return "http://" + url
    return url


def _connect_es_once(settings: Settings, endpoint: str) -> Tuple[Optional["Elasticsearch"], Optional[str]]:
    """Tente une connexion ES sur un endpoint donné (sans fallback)."""
    if Elasticsearch is None:
        return None, "elasticsearch client not installed"
    try:
        kwargs = dict(verify_certs=False, request_timeout=30)
        # Auth par API key prioritaire si dispo
        if getattr(settings, "ES_API_KEY", None):
            client = Elasticsearch(endpoint, api_key=settings.ES_API_KEY, **kwargs)
        # Sinon basic auth
        elif getattr(settings, "ES_USER", None) and getattr(settings, "ES_PASSWORD", None):
            client = Elasticsearch(endpoint, basic_auth=(settings.ES_USER, settings.ES_PASSWORD), **kwargs)
        else:
            client = Elasticsearch(endpoint, **kwargs)
        # ping
        client.info()
        return client, None
    except Exception as e:  # pragma: no cover
        return None, str(e)


def _connect_es(settings: Settings):
    """
    Etablit une connexion ES robuste :
    - normalise le schéma (http:// si manquant)
    - si échec en http://, retente en https:// (cas ES8 sécurisé)
    """
    endpoint = _normalize_endpoint(settings.ES_ENDPOINT)
    # ✅ Nouvelle version : on centralise la logique via utils.elastic
    try:
        from utils.elastic import get_elastic_client
        client = get_elastic_client()
        return client, None
    except Exception as e:
        # fallback original si utils.elastic n’existe pas dans l’environnement
        logger.warning(f"get_elastic_client() unavailable, fallback to manual connect: {e}")
        client, err = _connect_es_once(settings, endpoint)
        return client, err

def ensure_index(client, index: str, emb_dim: Optional[int] = None) -> None:
    """Create index with minimal mapping if it doesn't exist.
    Ajoute/complète le champ `embedding` si demandé (ES_EMBEDDING_DIM ou emb_dim)."""
    try:
        if client.indices.exists(index=index):
            # Optionnel : si l'index existe, s'assurer que le champ embedding est présent si demandé
            try:
                s = Settings()
                dim = emb_dim or getattr(s, "ES_EMBEDDING_DIM", None)
                if dim:
                    mapping = client.indices.get_mapping(index=index)
                    props = mapping[index]["mappings"].get("properties", {})
                    if "embedding" not in props:
                        client.indices.put_mapping(
                            index=index,
                            properties={
                                "embedding": {
                                    "type": "dense_vector",
                                    "dims": int(dim),
                                    "index": True,
                                    "similarity": "cosine",
                                }
                            },                        )
            except Exception as _:
                pass
            return
        body = {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "dynamic": True,
                "properties": {
                    "title": {"type": "text"},
                    "content": {"type": "text"},
                    "date": {"type": "date", "format": "strict_date||date_optional_time"},
                    "source_url": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    # meta.* fields are dynamic
                },
            },
        }
        # Ajoute le champ embedding si demandé via config
        try:
            s = Settings()
            dim = emb_dim or getattr(s, "ES_EMBEDDING_DIM", None)
            if dim:
                body["mappings"]["properties"]["embedding"] = {  # type: ignore[index]
                    "type": "dense_vector",
                    "dims": int(dim),
                    "index": True,
                    "similarity": "cosine",
                }
        except Exception:
            pass

        client.indices.create(index=index, body=body)
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to ensure index '{index}': {e}")


def _write_ndjson_fallback(packs: List[PackagedDoc], path: str = "ingest_out.jsonl") -> Dict[str, Any]:
    """Write NDJSON file locally as a fallback when ES is not available."""
    try:
        with open(path, "a", encoding="utf-8") as f:
            for p in packs:
                rec = {
                    "id": p.id,
                    "title": p.title,
                    "content": p.content,
                    "date": p.date,
                    "source_url": p.source_url,
                    "tags": p.tags,
                    "meta": p.meta,
                }
                if getattr(p, "embedding", None) is not None:
                    rec["embedding"] = p.embedding
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"ok": len(packs), "failed": 0, "total": len(packs), "fallback_path": os.path.abspath(path)}
    except Exception as e:
        logger.exception("NDJSON fallback write failed.")
        return {"ok": 0, "failed": len(packs), "total": len(packs), "error": str(e)}


def index_bulk(packs: List[PackagedDoc]) -> Dict[str, Any]:
    """
    Index a small batch of PackagedDoc into Elasticsearch.
    Fallback to local NDJSON if ES is not available.
    Returns stats: {'ok': int, 'failed': int, 'total': int, ...}
    """
    s = Settings()
    client, err = _connect_es(s)
    if client is None or helpers is None:
        logger.warning(
            f"Elasticsearch not available → NDJSON fallback. "
            f"reason=({err or 'helpers missing'}) endpoint={getattr(s,'ES_ENDPOINT',None)}"
        )
        return _write_ndjson_fallback(packs)

    # Déduire la dimension d'embedding depuis le lot si non configuré
    batch_dim: Optional[int] = None
    for p in packs:
        if getattr(p, "embedding", None) is not None:
            try:
                batch_dim = int(len(p.embedding))  # type: ignore[arg-type]
                break
            except Exception:
                pass

    ensure_index(client, s.ES_INDEX, emb_dim=batch_dim)

    actions = (
        {
            "_op_type": "index",
            "_index": s.ES_INDEX,
            "_id": p.id,
            "_source": {
                "title": p.title,
                "content": p.content,
                "date": p.date,
                "source_url": p.source_url,
                "tags": p.tags,
                "meta": p.meta,
            },
        }
        for p in packs
    )

    ok = 0
    failed = 0
    try:
        # Injecte `embedding` si présent sur le doc
        def _with_embedding(gen):
            for a, p in zip(gen, packs):
                if getattr(p, "embedding", None) is not None:
                    a["_source"]["embedding"] = p.embedding  # type: ignore[index]
                yield a

        for success, info in helpers.streaming_bulk(  # type: ignore            client,
            _with_embedding(actions),
            chunk_size=200,
            max_retries=2,
            raise_on_error=False,
        ):
            if success:
                ok += 1
            else:
                failed += 1
                logger.warning(f"Bulk item failed: {info}")
    except Exception as e:
        logger.exception("Bulk indexing crashed.")
        failed = len(packs) - ok

    # 🔄 Rendez les docs visibles immédiatement pour la requête qui suit
    try:
        client.indices.refresh(index=s.ES_INDEX)
    except Exception as e:
        logger.debug(f"Index refresh skipped/failed: {e}")

    return {"ok": ok, "failed": failed, "total": len(packs), "index": s.ES_INDEX}