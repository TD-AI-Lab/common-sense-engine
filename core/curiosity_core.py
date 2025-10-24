"""
CuriosityCore Facade
--------------------
Purpose: Provide a simple facade for the API/UI to trigger the full pipeline without exposing agent wiring details.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from datetime import date, datetime

from agents.fact_finder import FactFinder
from agents.causality_mapper import CausalityMapper
from agents.temporal_organizer import TemporalOrganizer
from agents.synthesizer import Synthesizer
from agents.conversational_core import ConversationalCore
from core.orchestrator import Orchestrator

from utils.config import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class CuriosityCore:
    """High-level entry point for the Common Sense Engine.

    Responsibilities:
    - Build and connect all core agents (FactFinder, CausalityMapper, TemporalOrganizer, Synthesizer)
    - Manage conversation memory via ConversationalCore
    - Offer a single `explain()` method for the API or UI layer
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Optionally pass a shared Settings instance so all agents share config."""
        self.settings = settings or Settings()

        # In production, you could inject clients (Elastic, Vertex AI, etc.)
        self.fact_finder = FactFinder(self.settings)
        self.causality_mapper = CausalityMapper()
        self.temporal_organizer = TemporalOrganizer()
        self.synthesizer = Synthesizer(self.settings)
        self.conversation = ConversationalCore()

        self.orchestrator = Orchestrator(
            fact_finder=self.fact_finder,
            causality_mapper=self.causality_mapper,
            temporal_organizer=self.temporal_organizer,
            synthesizer=self.synthesizer,
            conversation=self.conversation,
        )

        logger.info("CuriosityCore initialized with all agents.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_time_range(
        self, time_range: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Optional[str]]]:
        """Accepts dates/strings and returns ISO date strings or None."""
        if not time_range:
            return None

        def to_iso(v: Any) -> Optional[str]:
            if v is None:
                return None
            if isinstance(v, date):
                return v.isoformat()
            if isinstance(v, datetime):
                return v.date().isoformat()
            if isinstance(v, str):
                s = v.strip()
                # keep only YYYY or YYYY-MM or YYYY-MM-DD best-effort
                if not s:
                    return None
                # If a full ISO datetime slips in, cut to date
                if "T" in s:
                    try:
                        return datetime.fromisoformat(s).date().isoformat()
                    except Exception:
                        pass
                # Fallback: trust string (validators happen earlier in API)
                return s
            return None

        return {
            "from": to_iso(time_range.get("from")),
            "to": to_iso(time_range.get("to")),
        }

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def explain(
        self,
        query: str,
        *,
        time_range: Optional[Dict[str, Any]] = None,
        k: int = 20,
        live: bool = True,
        live_budget: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full explanation pipeline for a given query.

        Parameters
        ----------
        query : str
            User question (e.g., "Pourquoi le prix du riz augmente ?")
        time_range : dict | None
            Optional {"from": ISO8601, "to": ISO8601}
        k : int
            Max documents to retrieve (default 20)

        Returns
        -------
        dict : Final payload for UI/API consumption
        """
        if not query or not query.strip():
            return {"error": "Empty query", "nodes": [], "edges": [], "summary": ""}

        # Inject minimal context (based on short-term memory) and remember the turn
        enriched_query = self.conversation.inject_context(query)
        self.conversation.remember(enriched_query)

        logger.info(
            "CuriosityCore starting full pipeline",
            extra={"query": enriched_query, "k": k, "live": live, "request_id": request_id},
        )

        # Normalize time range for downstream components (ES expects strings)
        tr_norm = self._normalize_time_range(time_range)

        try:
            result = self.orchestrator.run_explain(
                enriched_query,
                time_range=tr_norm,
                k=k,
                live=live,
                live_budget=live_budget,
                request_id=request_id,
            )
        except Exception as e:
            logger.exception("CuriosityCore pipeline crashed")
            # Safe fallback payload
            return {
                "query": enriched_query,
                "nodes": [],
                "edges": [],
                "facts": [],
                "evidence": [],
                "summary": "",
                "confidence_global": 0.0,
                "errors": [f"CuriosityCore: {e}"],
            }

        # Store conversation feedback (for follow-up)
        evidence_count = len(result.get("evidence", []))
        confidence = result.get("confidence_global") or 0.0
        self.conversation.update_retrieval_feedback(evidence_count, confidence)

        # Add suggestions if needed
        if self.conversation.follow_up_needed():
            result["followup_suggestions"] = self.conversation.suggest_followup()

        logger.info(
            "CuriosityCore pipeline completed",
            extra={"confidence": confidence, "evidence": evidence_count},
        )

        return result