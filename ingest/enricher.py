"""
Enricher: derive Evidence units from ParsedDocument and score them.

P0 scope:
- Sentence segmentation (very light): split on [.?!] + newline boundaries
- Scoring:
  S_p = pseudoBM25(query, sentence)
- Offsets are computed on the normalized text used everywhere else
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import re

from .types import ParsedDocument, EvidenceUnit


def extract_evidence(doc: ParsedDocument, query: str, *, max_units: int = 5) -> List[EvidenceUnit]:
    """
    Build EvidenceUnit[] from a single ParsedDocument:
    1) split into sentences
    2) compute a simple relevance score per sentence
    3) keep top-M with constraints (max 2 per paragraph)
    4) compute robust [start,end] offsets on doc.text
    """
    sentences, para_ids = _split_sentences(doc.text)
    if not sentences:
        return []

    q_terms = _tokenize(query)
    idf = _fake_idf(q_terms)

    scored: List[Tuple[int, float]] = []
    for i, s in enumerate(sentences):
        s_terms = set(_tokenize(s))
        overlap = [t for t in q_terms if t in s_terms]
        bm25 = sum(idf.get(t, 0.0) for t in overlap)
        scored.append((i, bm25))

    # top-k by score (desc), oversample then enforce diversity
    top_idx = [i for i, _ in sorted(scored, key=lambda x: x[1], reverse=True)[: max_units * 3]]
    top_idx = _enforce_diversity(top_idx, para_ids, limit=max_units)

    ev: List[EvidenceUnit] = []
    for i in top_idx[:max_units]:
        s = sentences[i]
        start = doc.text.find(s)
        if start == -1:
            # extremely rare if normalization preserved substrings
            continue
        end = start + len(s)
        score = float(next(score for idx, score in scored if idx == i))
        ev.append(
            EvidenceUnit(
                doc_id=_doc_id(doc),
                text=s,
                span=[start, end],
                sentence_id=i,
                para_id=para_ids[i],
                date=doc.published_at,
                source_url=doc.canonical_url or doc.url,
                score=score,
            )
        )
    return ev


# -----------------------------
# Helpers
# -----------------------------

def _split_sentences(text: str) -> tuple[List[str], List[int]]:
    """
    Very light sentence splitting:
    - split on punctuation [.?!] followed by space/newline
    - track paragraph ids (split on blank lines)
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences: List[str] = []
    para_ids: List[int] = []
    for pid, p in enumerate(paras):
        parts = re.split(r"(?<=[\.\?!])\s+", p)
        for s in parts:
            s = s.strip()
            if len(s) < 40:  # drop too short
                continue
            sentences.append(s)
            para_ids.append(pid)
    return sentences, para_ids


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]{3,}", text.lower())]


def _fake_idf(q_terms: List[str]) -> Dict[str, float]:
    """
    Deterministic pseudo-IDF: rarer terms in the query get higher weight.
    For P0, approximate rarity by term length (ok for fast-path).
    """
    uniq = set(q_terms)
    return {t: 1.0 + min(2.0, len(t) / 8.0) for t in uniq}


def _enforce_diversity(idxs: List[int], para_ids: List[int], *, limit: int) -> List[int]:
    """Keep at most 2 evidence units per paragraph for variety."""
    out: List[int] = []
    per_para: Dict[int, int] = {}
    for i in idxs:
        pid = para_ids[i]
        cnt = per_para.get(pid, 0)
        if cnt >= 2:
            continue
        out.append(i)
        per_para[pid] = cnt + 1
        if len(out) >= limit:
            break
    return out


def _doc_id(doc: ParsedDocument) -> str:
    basis = (doc.title or "") + "|" + (doc.publisher or "") + "|" + (doc.published_at or "") + "|" + (doc.url or "")
    import hashlib
    return hashlib.md5(basis.encode("utf-8")).hexdigest()