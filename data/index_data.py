"""
Indexing Script (MVP)
---------------------
Purpose: Load JSONL, compute embeddings (placeholder), and index into Elastic with fields required by FactFinder.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional
import json
import hashlib
from pathlib import Path
from datetime import datetime
from functools import lru_cache

from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import BulkIndexError  # type: ignore

from utils.logger import get_logger
from utils.config import Settings
from utils.elastic import get_elastic_client

logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# MUST match the embedding model used by FactFinder if/when enabled.
# textembedding-gecko@003 returns 768-dim vectors.
EMBED_DIMS = 768
BATCH_SIZE = 500

# ---------------------------------------------------------------------
# Embeddings (Vertex AI) — optional but recommended for hybrid search
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_vertex_embedder():
    """Lazy-load Vertex AI TextEmbeddingModel if configured. Returns (model, enabled_flag)."""
    try:
        from utils.config import Settings
        s = Settings()
        if not (s.VERTEX_PROJECT and s.VERTEX_LOCATION):
            logger.info("Vertex embeddings disabled (VERTEX_PROJECT/LOCATION not set).")
            return None
        import vertexai
        from vertexai.language_models import TextEmbeddingModel
        vertexai.init(project=s.VERTEX_PROJECT, location=s.VERTEX_LOCATION)
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        logger.info("Vertex embeddings enabled with textembedding-gecko@003")
        return model
    except Exception as e:
        logger.warning(f"Vertex embeddings not available: {e}")
        return None


def embed_text(text: str) -> Optional[list[float]]:
    """Return a 768-dim embedding using Vertex AI if available, else None (lexical-only)."""
    model = _load_vertex_embedder()
    if not model:
        return None
    try:
        vec = model.get_embeddings([text[:3072]])[0].values
        arr = [float(x) for x in vec]
        if len(arr) != EMBED_DIMS:
            logger.warning(f"Unexpected embedding dims: {len(arr)} (expected {EMBED_DIMS})")
        return arr
    except Exception as e:
        logger.warning(f"Embedding failed, continuing without vector: {e}")
        return None

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _normalize_date(val: Any) -> Optional[str]:
    """Normalize various date formats to ISO 'YYYY-MM-DD'. Returns None on failure."""
    if not val:
        return None
    s = str(val).strip()
    # Already ISO-like
    try:
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            # Validate
            datetime.fromisoformat(s)
            return s
    except Exception:
        pass

    # Try common formats
    fmts = ("%Y/%m/%d", "%d/%m/%Y", "%Y.%m.%d", "%Y-%m", "%Y")
    for fmt in fmts:
        try:
            if fmt == "%Y-%m":
                dt = datetime.strptime(s, "%Y-%m")
                return dt.strftime("%Y-%m-01")
            if fmt == "%Y":
                dt = datetime.strptime(s, "%Y")
                return dt.strftime("%Y-01-01")
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    # Fallback: try generic parse
    try:
        dt = datetime.fromisoformat(s)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def _deterministic_id(doc: Dict[str, Any]) -> str:
    """Return a deterministic ID for the document when none is provided."""
    basis = (
        (doc.get("id") or "")
        + "|"
        + (doc.get("title") or "")
        + "|"
        + (doc.get("date") or "")
        + "|"
        + (doc.get("source_url") or "")
        + "|"
        + (doc.get("content") or "")[:200]
    )
    return hashlib.md5(basis.encode("utf-8")).hexdigest()


def _ensure_index(client: Elasticsearch, index: str) -> None:
    """Create index with expected mappings if it does not exist."""
    try:
        exists = client.indices.exists(index=index)
    except Exception as e:
        logger.exception(f"Failed to check index existence '{index}': {e}")
        raise

    if exists:
        logger.info("Index already exists", extra={"index": index})
        return

    body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "dynamic": True,  # allow extra fields if present
            "properties": {
                "title": {"type": "text"},
                "content": {"type": "text"},
                "date": {"type": "date", "format": "strict_date||date_optional_time"},
                "source_url": {"type": "keyword"},
                "tags": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBED_DIMS,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    }

    try:
        client.indices.create(index=index, body=body)
        logger.info("Index created", extra={"index": index})
    except Exception as e:
        logger.exception(f"Failed to create index '{index}' with mappings: {e}")
        raise


def _prepare_doc(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize/prepare a document before indexing."""
    doc: Dict[str, Any] = dict(raw)

    # Normalize tags to list[str]
    tags = doc.get("tags")
    if tags is None:
        doc["tags"] = []
    elif not isinstance(tags, list):
        doc["tags"] = [str(tags)]
    else:
        # ensure all tags are strings
        doc["tags"] = [str(t) for t in tags]

    # Normalize date
    norm_date = _normalize_date(doc.get("date"))
    if norm_date:
        doc["date"] = norm_date
    else:
        # It's acceptable to omit invalid dates; FactFinder handles optional date filter
        doc.pop("date", None)

    # Compute embedding (optional)
    content = doc.get("content", "") or ""
    emb = embed_text(content[:2000])
    if emb is not None:
        doc["embedding"] = emb
    else:
        # omit field entirely to avoid storing nulls
        doc.pop("embedding", None)

    # Ensure we don't store absurdly large content (keeps index smaller)
    if isinstance(content, str) and len(content) > 100_000:
        doc["content"] = content[:100_000]

    # Ensure source_url type
    if "source_url" in doc and doc["source_url"] is not None:
        doc["source_url"] = str(doc["source_url"])

    return doc


def _action_stream(p: Path, index: str) -> Iterable[Dict[str, Any]]:
    """Yield bulk actions from a JSONL file."""
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(
                    "Skipping invalid JSON line",
                    extra={"line_no": line_no, "error": str(e)},
                )
                continue

            doc = _prepare_doc(raw)
            doc_id = raw.get("id") or _deterministic_id(doc)

            yield {
                "_op_type": "index",  # upsert-style replace
                "_index": index,
                "_id": doc_id,
                "_source": doc,
            }


# ---------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------

def index_jsonl(path: str) -> None:
    """Indexe un fichier JSONL dans Elasticsearch (supporte API key, basic auth ou local)."""
    settings = Settings()
    try:
        client = get_elastic_client()
        if not client.ping():
            raise ConnectionError("Elasticsearch cluster not reachable.")
        logger.info("Connected to Elasticsearch", extra={"endpoint": settings.ES_ENDPOINT})
    except Exception as e:
        logger.exception("Failed to connect to Elasticsearch")
        return
        
        if not client.ping():
            raise ConnectionError("Elasticsearch cluster not reachable.")
        logger.info("Connected to Elasticsearch", extra={"endpoint": settings.ES_ENDPOINT})
    except Exception as e:
        logger.exception("Failed to connect to Elasticsearch")
        return

    p = Path(path)
    if not p.exists():
        logger.error("JSONL file not found", extra={"path": path})
        return

    # Ensure index exists with proper mapping
    try:
        _ensure_index(client, settings.ES_INDEX)
    except Exception:
        # Error already logged inside _ensure_index
        return

    logger.info("Indexing started", extra={"file": str(p), "index": settings.ES_INDEX})

    # Streaming bulk
    total_ok = 0
    total_fail = 0
    try:
        for ok, info in helpers.streaming_bulk(
            client,
            _action_stream(p, settings.ES_INDEX),
            chunk_size=BATCH_SIZE,
            max_retries=2,
            initial_backoff=1,
            max_backoff=8,
        ):
            if ok:
                total_ok += 1
            else:
                total_fail += 1
                logger.warning("Bulk item failed", extra={"item": info})
    except BulkIndexError as bie:
        total_fail += len(bie.errors)  # type: ignore[attr-defined]
        logger.exception("BulkIndexError during indexing")
    except Exception as e:
        logger.exception(f"Unexpected error during streaming_bulk: {e}")

    logger.info(
        "Indexing completed",
        extra={"index": settings.ES_INDEX, "ok": total_ok, "failed": total_fail, "total_docs": total_ok + total_fail},
    )

# ---------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m data.index_data <path_to_jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Indexing from: {path}")
    index_jsonl(path)