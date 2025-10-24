"""
Ingestion pipeline package for Common Sense Engine (CSE).

Modules:
- types.py     : shared dataclasses used across the pipeline
- seeds.py     : generate discovery 'seeds' (queries, RSS, whitelists)
- fetcher.py   : polite fetching (robots.txt, rate-limit, retries)
- parser.py    : clean text & extract metadata from HTML
- enricher.py  : derive Evidence units (sentences + spans) and light scoring
- causality.py : detect causal candidates (nodes/edges) from evidence
- packager.py  : assemble JSONL-ready documents (v1 + meta)
- indexer.py   : push documents to Elasticsearch in bulk (or local NDJSON fallback)

Quick start (sync):
    from ingest.seeds import build_seeds
    from ingest.fetcher import fetch_seed_sync
    from ingest.parser import parse_page
    from ingest.enricher import extract_evidence
    from ingest.causality import build_causal_candidates
    from ingest.packager import package_jsonl, attach_causality_meta
    from ingest.indexer import index_bulk

    seeds = build_seeds("prix du riz", lang=["fr","en"], since=None, k=10)
    pages = sum((fetch_seed_sync(s) for s in seeds), [])
    docs = list(filter(None, (parse_page("req-1", p) for p in pages)))
    ev   = [e for d in docs for e in extract_evidence(d, "prix du riz", max_units=5)]
    nodes, edges = build_causal_candidates(ev)
    packs = [package_jsonl(d, [e for e in ev if e.doc_id]) for d in docs]
    packs = [attach_causality_meta(p, nodes, edges) for p in packs]
    stats = index_bulk(packs)
    print(stats)

Notes:
- Asynchronous APIs are provided in fetcher; synchronous wrappers are available for convenience.
"""

from .types import (  # re-export common types
    Seed, FetchedPage, ParsedDocument, EvidenceUnit,
    CausalNode, CausalEdge, PackagedDoc, IngestJob,
)

__all__ = [
    "Seed", "FetchedPage", "ParsedDocument", "EvidenceUnit",
    "CausalNode", "CausalEdge", "PackagedDoc", "IngestJob",
]