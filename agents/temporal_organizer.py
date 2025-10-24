"""
TemporalOrganizer Agent
-----------------------
Purpose: Impose a clear chronology on the causal graph and annotate nodes/edges with normalized temporal information.

Improvements:
- Infer node dates from `facts` (doc_id -> date) and `evidence`.
- Normalize to ISO (YYYY-MM-DD) and add `time_precision`: year|month|day.
- Annotate edges with `time_src`, `time_dst`, and `temporal_order` (forward/backward/unknown).
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import re

from utils.logger import get_logger
from utils.metrics import timed

logger = get_logger(__name__)


class TemporalOrganizer:
    """Adds temporal structure and returns a re-ordered graph.

    Implement `process` to:
    1) Parse `nodes` and `edges` from previous step
    2) Normalize `node.time` to ISO strings; if absent, try to infer from sources/evidence
    3) Sort nodes by time; break ties deterministically (lexicographic id)
    4) Attach timestamps and temporal order onto edges for UI consumption
    """

    # ------------------------- Date helpers -------------------------

    def _normalize_date(self, value: Optional[Any]) -> Optional[str]:
        """Convert loose formats (YYYY, YYYY-MM, ISO with time, etc.) into ISO 8601 date (YYYY-MM-DD)."""
        if value is None:
            return None

        # Accept ints/floats as years
        if isinstance(value, (int, float)):
            year = int(value)
            if 1000 <= year <= 3000:
                return f"{year:04d}-01-01"
            return None

        value = str(value).strip()
        if not value:
            return None

        # If already ISO date
        m = re.match(r"^(\d{4}-\d{2}-\d{2})", value)
        if m:
            return m.group(1)

        # Replace trailing Z for fromisoformat (Python does not parse 'Z')
        iso_candidate = value.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(iso_candidate)
            return dt.date().isoformat()
        except Exception:
            pass

        # Simple year or year-month
        if re.fullmatch(r"\d{4}", value):
            return f"{value}-01-01"
        if re.fullmatch(r"\d{4}-\d{2}", value):
            return f"{value}-01"

        # Try parsing with common fallback formats
        for fmt in ("%Y/%m/%d", "%d/%m/%Y", "%Y.%m.%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m"):
            try:
                dt = datetime.strptime(value, fmt)
                # When only year/month were provided (e.g. %Y/%m), day defaults to 1
                return dt.strftime("%Y-%m-%d")
            except Exception:
                continue

        return None

    def _precision_of(self, value: Optional[str]) -> Optional[str]:
        """Return 'day' | 'month' | 'year' depending on granularity of the normalized value."""
        if not value:
            return None
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return "day"
        if re.fullmatch(r"\d{4}-\d{2}", value):
            return "month"
        if re.fullmatch(r"\d{4}", value):
            return "year"
        # normalized() returns YYYY-MM-DD, keep safe default
        return "day"

    def _infer_time_from_text(self, text: Optional[str]) -> Optional[str]:
        """Very light inference: look for a date-like token in the node text/label."""
        if not text:
            return None
        text = str(text)

        patterns = [
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{4}-\d{2}\b",
            r"\b(19|20)\d{2}\b",
            r"\b\d{4}/\d{2}/\d{2}\b",
            r"\b\d{2}/\d{2}/\d{4}\b",
            r"\b\d{4}\.\d{2}\.\d{2}\b",
            r"\b\d{2}-\d{2}-\d{4}\b",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return self._normalize_date(m.group(0))
        return None

    # ------------------------- Sorting & annotation -------------------------

    def _sort_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort nodes by normalized time, fallback to ID then label for stability."""
        def sort_key(node: Dict[str, Any]):
            t_str = node.get("time")
            try:
                t = datetime.fromisoformat(t_str) if t_str else datetime.max
            except Exception:
                t = datetime.max
            return (t, node.get("id", ""), node.get("label", ""), node.get("summary", ""))
        return sorted(nodes, key=sort_key)

    def _annotate_edges(
        self, edges: List[Dict[str, Any]], node_times: Dict[str, Optional[str]]
    ) -> List[Dict[str, Any]]:
        """Attach time_src/time_dst and temporal_order for UI timeline. Accept both src/dst and source/target."""
        annotated: List[Dict[str, Any]] = []
        for e in edges:
            src_id = e.get("src") or e.get("source")
            dst_id = e.get("dst") or e.get("target")
            ts = node_times.get(src_id)
            td = node_times.get(dst_id)
            new_e = dict(e)
            new_e["time_src"] = ts
            new_e["time_dst"] = td
            if ts and td:
                try:
                    new_e["temporal_order"] = "forward" if ts <= td else "backward"
                except Exception:
                    new_e["temporal_order"] = "unknown"
            else:
                new_e["temporal_order"] = "unknown"
            annotated.append(new_e)
        return annotated

    # ------------------------- Main -------------------------

    @timed("temporal_organizer.process")
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = list(input_data.get("nodes", []))   # shallow copy
        edges: List[Dict[str, Any]] = list(input_data.get("edges", []))   # shallow copy
        facts: List[Dict[str, Any]] = list(input_data.get("facts", []))
        evidence: List[Dict[str, Any]] = list(input_data.get("evidence", []))

        if not nodes:
            return {"nodes": [], "edges": edges, "chronology": []}

        # Build a mapping doc_id -> normalized date (prefer the earliest usable date)
        doc_dates: Dict[str, str] = {}
        # 1) from facts (authoritative at doc level)
        for f in facts:
            did = f.get("id")
            dt = self._normalize_date(f.get("date"))
            if did and dt:
                if (did not in doc_dates) or (dt < doc_dates[did]):
                    doc_dates[did] = dt
        # 2) from evidence (sometimes carries a better-granularity date)
        for ev in evidence:
            did = ev.get("doc_id")
            dt = self._normalize_date(ev.get("date"))
            if did and dt:
                if (did not in doc_dates) or (dt < doc_dates[did]):
                    doc_dates[did] = dt

        # If missing time, try from sources/evidence; then normalize everything
        for n in nodes:
            n_time = n.get("time")
            if not n_time:
                # try from sources (assumed to be doc_ids or URLs; we only map doc_ids here)
                for sid in (n.get("sources") or []):
                    dt = doc_dates.get(sid)
                    if dt:
                        n_time = dt
                        break
            if not n_time:
                inferred = self._infer_time_from_text(n.get("summary") or n.get("label"))
                if inferred:
                    n_time = inferred
            n["time"] = self._normalize_date(n_time)
            n["time_precision"] = self._precision_of(n.get("time"))

        # Sort nodes by time
        nodes_sorted = self._sort_nodes(nodes)

        # Annotate edges with timestamps and order
        node_times = {n.get("id"): n.get("time") for n in nodes_sorted if n.get("id")}
        edges_annotated = self._annotate_edges(edges, node_times)

        # Add stable chronological ordering (useful for UI timeline + narrative)
        ordered_ids = [n["id"] for n in nodes_sorted if n.get("id")]
        chronology = [{"id": nid, "order": i} for i, nid in enumerate(ordered_ids)]

        logger.info(
            "TemporalOrganizer processed chronology",
            extra={"nodes": len(nodes_sorted), "edges": len(edges_annotated)},
        )

        return {
            "nodes": nodes_sorted,
            "edges": edges_annotated,
            "chronology": chronology,
        }