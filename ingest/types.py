"""
Shared types for the ingestion pipeline.
These dataclasses travel across modules to ensure strongly-typed, testable boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime


SeedType = Literal["query", "rss", "url"]
ContentType = Literal["html", "json", "xml", "rss", "unknown"]
CausalRelation = Literal["causes", "correlates", "contradicts"]


@dataclass
class Seed:
    """A discovery unit that instructs the fetcher where/what to fetch."""
    type: SeedType
    value: str  # query string for 'query', URL for 'rss'/'url'
    provider: Optional[str] = None  # e.g., 'bing', 'programmable-search', 'faorss'
    lang: Optional[str] = None      # 'fr', 'en', etc.
    topic: Optional[str] = None     # user topic (e.g., "prix du riz")
    since: Optional[str] = None     # ISO date hint
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FetchedPage:
    """Raw HTTP response captured by fetcher."""
    url: str
    status: int
    fetched_at: datetime
    headers: Dict[str, str]
    content_type: ContentType
    raw: bytes  # raw bytes (keep as-is for audit)
    error: Optional[str] = None
    robots_allowed: bool = True
    final_url: Optional[str] = None  # after redirects


@dataclass
class ParsedDocument:
    """Normalized text + metadata extracted from a fetched page."""
    request_id: str
    url: str
    canonical_url: Optional[str]
    fetched_at: datetime
    status: int
    publisher: Optional[str]
    author: List[str]
    title: Optional[str]
    lang: Optional[str]
    country: Optional[str]
    published_at: Optional[str]  # ISO date (YYYY-MM-DD)
    section: Optional[str]
    text: str                     # normalized plain text
    char_map_ref: Optional[str]   # reference to char-map file (if externalized)
    raw_ref: Optional[str]        # storage ref to raw page (e.g., gs://…)
    parsed_ref: Optional[str]     # storage ref to parsed JSON (e.g., gs://…)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceUnit:
    """Atomic evidence extracted from ParsedDocument with offsets on normalized text."""
    doc_id: str                   # a stable id (hash/url)
    text: str                     # sentence/paragraph text
    span: List[int]               # [start, end] on ParsedDocument.text
    sentence_id: Optional[int]
    para_id: Optional[int]
    date: Optional[str]           # ISO date if detected near the span
    # Nouveau champ normalisé (utilisé par le CausalityMapper/UI)
    url: Optional[str] = None
    # Ancien alias conservé pour compat
    source_url: Optional[str] = None
    score: float = 0.0            # score normalisé de préférence sur [0,1]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalNode:
    id: str
    label: str
    summary: str
    time: Optional[str] = None    # ISO date (YYYY-MM-DD)
    sources: List[str] = field(default_factory=list)  # doc_ids
    locations: List[str] = field(default_factory=list)
    # Champs additionnels alignés sur le mapper v2.1 (facultatifs)
    text: Optional[str] = None
    title: Optional[str] = None


@dataclass
class CausalEdge:
    src: str
    dst: str
    relation: CausalRelation
    confidence: float
    evidence: List[Dict[str, Any]]  # [{doc_id, span, url?, score?}]
    # Champs UI/pondération (facultatifs)
    weight: float = 0.0
    color: Optional[str] = None
    source_text: Optional[str] = None
    sources: List[str] = field(default_factory=list)  # URLs agrégées
    style: Optional[str] = None       # "dashed" pour liens indirects
    indirect: bool = False
    circular: bool = False

@dataclass
class PackagedDoc:
    """JSONL v1 + meta document ready for Elasticsearch indexing."""
    id: str
    title: str
    content: str
    date: Optional[str]
    source_url: Optional[str]
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    # Optionnel : vecteur prêt à indexer (si déjà calculé)
    embedding: Optional[List[float]] = None

@dataclass
class IngestJob:
    """High-level job context propagated across pipeline for observability."""
    job_id: str
    topic: str
    lang: List[str]
    since: Optional[str]
    k: int
    request_id: str
    started_at: datetime
    seeds: List[Seed] = field(default_factory=list)
    fetched: List[FetchedPage] = field(default_factory=list)
    parsed: List[ParsedDocument] = field(default_factory=list)
    evidence: List[EvidenceUnit] = field(default_factory=list)
    nodes: List[CausalNode] = field(default_factory=list)
    edges: List[CausalEdge] = field(default_factory=list)
    packaged: List[PackagedDoc] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)