"""
Causality candidate extractor from Evidence.

P0 scope:
- Recognize FR/EN causal patterns (spec §9)
- Split sentence around the pattern to produce (cause → effect)
- Build nodes / edges with deterministic IDs
- Score edges with a simple formula (pattern strength only in P0)
"""

from __future__ import annotations

from typing import List, Dict, Tuple
import re
import hashlib

from .types import EvidenceUnit, CausalNode, CausalEdge


# Patterns
_PATTERNS_STRONG = [
    r"\bà cause de\b", r"\bbecause of\b", r"\bdue to\b", r"\bcauses?\b", r"\bentraîne\b", r"\bprovoque\b",
    r"\bleads to\b", r"\bresults in\b", r"\bconduit à\b", r"\bmène à\b", r"\bengendre\b", r"\btriggers?\b"
]
_PATTERNS_MEDIUM = [r"\bfavorise\b", r"\bdrives?\b", r"\best lié à\b", r"\bcorrelates with\b", r"\bcoïncide avec\b"]


def build_causal_candidates(evidence: List[EvidenceUnit]) -> tuple[List[CausalNode], List[CausalEdge]]:
    """
    Convert evidence sentences into a minimal causal graph (nodes + edges).
    Deduplicate nodes by normalized text; remap edges accordingly.
    """
    nodes: List[CausalNode] = []
    edges: List[CausalEdge] = []

    for ev in evidence:
        raw = ev.text
        txt = raw.lower()

        m, weight = _match_first(txt, _PATTERNS_STRONG, 0.6)
        if not m:
            m, weight = _match_first(txt, _PATTERNS_MEDIUM, 0.4)
        if not m:
            continue

        src_txt = raw[:m.start()].strip(" .;,:-")
        dst_txt = raw[m.end():].strip(" .;,:-")
        if len(src_txt.split()) < 3 or len(dst_txt.split()) < 3:
            continue

        src = _make_node(src_txt, [ev.doc_id], time_hint=ev.date)
        dst = _make_node(dst_txt, [ev.doc_id], time_hint=ev.date)

        edges.append(
            CausalEdge(
                src=src.id,
                dst=dst.id,
                relation="causes" if weight >= 0.5 else "correlates",
                confidence=weight,
                evidence=[{"doc_id": ev.doc_id, "span": ev.span}],
            )
        )
        nodes.extend([src, dst])

    nodes, id_map = _dedup_nodes(nodes)
    kept_ids = {n.id for n in nodes}
    for e in edges:
        e.src = id_map.get(e.src, e.src)
        e.dst = id_map.get(e.dst, e.dst)
    edges = [e for e in edges if e.src in kept_ids and e.dst in kept_ids]

    return nodes, edges


# -----------------------------
# Helpers
# -----------------------------

def _match_first(text: str, patterns: List[str], weight: float):
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m, weight
    return None, 0.0


def _make_node(summary: str, sources: List[str], time_hint: str | None) -> CausalNode:
    nid = "n_" + hashlib.md5(summary.lower().encode("utf-8")).hexdigest()[:10]
    label = summary[:60]
    return CausalNode(id=nid, label=label, summary=summary, time=time_hint, sources=list(set(sources)))


def _dedup_nodes(nodes: List[CausalNode]) -> tuple[List[CausalNode], Dict[str, str]]:
    seen: Dict[str, CausalNode] = {}
    id_map: Dict[str, str] = {}
    result: List[CausalNode] = []
    for n in nodes:
        key = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", n.summary.lower()))
        if key in seen:
            keep = seen[key]
            keep.sources = list(set(keep.sources + n.sources))
            id_map[n.id] = keep.id
        else:
            seen[key] = n
            result.append(n)
            id_map[n.id] = n.id
    return result, id_map