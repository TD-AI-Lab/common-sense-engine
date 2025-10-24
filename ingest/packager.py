"""
Packager: turn {ParsedDocument, Evidence, Causal graph} into JSONL v1 + meta.

P0 scope:
- Produce PackagedDoc (id, title, content, date, source_url, tags, meta)
- content must be the normalized text used for evidence spans
- meta should include: publisher/lang, evidence (top spans), audit refs if available
- Compatible with the ES mapping used in CSE
"""

from __future__ import annotations

from typing import List, Dict, Any
import hashlib

from .types import ParsedDocument, EvidenceUnit, PackagedDoc, CausalNode, CausalEdge


def package_jsonl(doc: ParsedDocument, evidence: List[EvidenceUnit]) -> PackagedDoc:
    """
    Convert a parsed doc + its evidence into a single JSONL-ready PackagedDoc.
    ID strategy:
      - if canonical_url present → hash(canonical_url)
      - else → hash(title|publisher|published_at|url)
    """
    doc_id = _doc_id(doc)
    tags: List[str] = []  # optional for P0
    meta: Dict[str, Any] = {
        "publisher": doc.publisher,
        "lang": doc.lang,
        "evidence": [{"text": ev.text, "span": ev.span} for ev in evidence[:5]],
        "audit": {"raw_ref": doc.raw_ref, "parsed_ref": doc.parsed_ref},
    }
    return PackagedDoc(
        id=doc_id,
        title=doc.title or (doc.publisher or "Document"),
        content=doc.text,
        date=doc.published_at,
        source_url=doc.canonical_url or doc.url,
        tags=tags,
        meta=meta,
    )


def attach_causality_meta(pack: PackagedDoc, nodes: List[CausalNode], edges: List[CausalEdge]) -> PackagedDoc:
    """Optional: add causal candidates to meta for downstream use / auditing."""
    pack.meta["causality"] = {
        "nodes": [n.__dict__ for n in nodes[:50]],
        "edges": [e.__dict__ for e in edges[:100]],
    }
    return pack


def _doc_id(doc: ParsedDocument) -> str:
    if doc.canonical_url:
        return hashlib.md5(doc.canonical_url.encode("utf-8")).hexdigest()
    basis = (doc.title or "") + "|" + (doc.publisher or "") + "|" + (doc.published_at or "") + "|" + (doc.url or "")
    return hashlib.md5(basis.encode("utf-8")).hexdigest()