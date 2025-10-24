# utils/elastic.py
from __future__ import annotations

from elasticsearch import Elasticsearch
from typing import Any, Dict, List, Optional
from utils.config import Settings
from modules.embedder import LocalEmbedder
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_elastic_client() -> Elasticsearch:
    """Return a configured Elasticsearch client with API key or basic auth."""
    s = Settings()
    es_url = s.ES_ENDPOINT
    if not es_url.startswith(("http://", "https://")):
        es_url = f"https://{es_url}"

    params = dict(
        hosts=[es_url],
        verify_certs=True,
        request_timeout=10,
    )

    if getattr(s, "ES_API_KEY", None):
        params["api_key"] = s.ES_API_KEY
    elif s.ES_USER and s.ES_PASSWORD:
        params["basic_auth"] = (s.ES_USER, s.ES_PASSWORD)

    return Elasticsearch(**params)


# ----------------------------------------------------------------------
# Hybrid Retrieval with Context Filtering + Dense Rerank
# ----------------------------------------------------------------------

def hybrid_search(
    client: Elasticsearch,
    index: str,
    query: str,
    *,
    k: int = 20,
    context_filter: Optional[Dict[str, Any]] = None,
    rerank: bool = True,
    rerank_weight: float = 0.45,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[Dict[str, Any]]:
    """
    Perform a hybrid search: BM25 + dense rerank, with optional context filtering.

    Args:
        client: Elasticsearch client.
        index: target index name.
        query: user query string.
        k: number of results to return (default 20).
        context_filter: dict of filters (e.g., {"lang": "fr", "domain": "economy"}).
        rerank: if True, apply dense rerank on top BM25 hits.
        rerank_weight: blending factor (0..1) between BM25 and semantic similarity.
        model_name: SentenceTransformer model for reranking.

    Returns:
        List of documents with unified score field `hybrid_score`.
    """
    must_filters = []
    if context_filter:
        for field, value in context_filter.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                must_filters.append({"terms": {field: list(value)}})
            else:
                must_filters.append({"term": {field: value}})

    # Base BM25 query
    body = {
        "query": {
            "bool": {
                "must": {"match": {"content": query}},
                "filter": must_filters,
            }
        },
        "_source": ["id", "title", "content", "summary", "lang", "domain", "published_at"],
        "size": max(k * 2, 40),
    }

    res = client.search(index=index, body=body)
    hits = res.get("hits", {}).get("hits", [])
    if not hits:
        logger.warning("No hits found for query: %s", query)
        return []

    # Normalize BM25 scores to 0..1
    bm25_scores = np.array([h["_score"] for h in hits], dtype=float)
    if bm25_scores.ptp() > 0:
        bm25_scores = (bm25_scores - bm25_scores.min()) / bm25_scores.ptp()
    else:
        bm25_scores = np.ones_like(bm25_scores)

    for h, s in zip(hits, bm25_scores):
        h["_norm_score"] = float(s)

    # Optional dense rerank
    if rerank:
        try:
            embedder = LocalEmbedder(model_name)
            docs = [h["_source"].get("content") or h["_source"].get("summary") or "" for h in hits]
            q_emb = np.array(embedder.embed([query])[0])
            d_embs = np.array(embedder.embed(docs))
            sims = (d_embs @ q_emb) / (
                np.linalg.norm(d_embs, axis=1) * np.linalg.norm(q_emb) + 1e-8
            )
            sims = (sims - sims.min()) / (sims.ptp() + 1e-8)
            for h, s in zip(hits, sims):
                h["_sim_score"] = float(s)
                h["hybrid_score"] = (
                    rerank_weight * s + (1.0 - rerank_weight) * h["_norm_score"]
                )
        except Exception as e:
            logger.warning(f"Hybrid rerank failed: {e}")
            for h in hits:
                h["hybrid_score"] = h["_norm_score"]
    else:
        for h in hits:
            h["hybrid_score"] = h["_norm_score"]

    hits.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return hits[:k]