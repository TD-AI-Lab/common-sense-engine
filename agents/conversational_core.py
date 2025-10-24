"""
ConversationalCore Agent
------------------------
Purpose: Manage multi-turn context, reformulate user questions, and trigger follow-up retrieval when needed.
"""

from typing import Any, Dict, List, Optional
import re

from utils.logger import get_logger

logger = get_logger(__name__)


class ConversationalCore:
    """Lightweight dialogue manager for MVP.

    Responsibilities:
    - Maintain short-term memory of recent turns (user/system)
    - Reformulate queries when user adds temporal or referential hints
    - Detect when a follow-up retrieval is necessary
    """

    def __init__(self, history_limit: int = 5) -> None:
        self.history: List[Dict[str, Any]] = []
        self.history_limit = history_limit
        self.last_confidence: Optional[float] = None
        self.last_evidence_count: Optional[int] = None

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def remember(self, user_msg: str, system_msg: Optional[str] = None) -> None:
        """Store last turn (bounded by `history_limit`)."""
        self.history.append({"user": user_msg, "system": system_msg})
        self.history = self.history[-self.history_limit:]
        logger.debug("Conversation remembered", extra={"len": len(self.history)})

    def get_last_user_query(self) -> Optional[str]:
        """Return the last user query in memory, if any."""
        for turn in reversed(self.history):
            if "user" in turn and turn["user"]:
                return turn["user"]
        return None

    # ------------------------------------------------------------------
    # Context injection
    # ------------------------------------------------------------------

    def inject_context(self, query: str) -> str:
        """Return a slightly reformulated query using recent turns.

        Example:
        - last query: "Pourquoi le prix du riz augmente ?"
        - new user: "Et si on regarde en 2021 ?"
        -> reformulated: "Pourquoi le prix du riz augmente en 2021 ?"
        """
        base = query.strip()
        last = self.get_last_user_query()
        if not last:
            return base

        # If current message is a referential nudge with a year, inject it into the last full question
        ref_pat = r"\b(et si|et maintenant|et pour|si on regarde)\b"
        year_now = re.search(r"\b(19|20)\d{2}\b", base)
        if year_now and re.search(ref_pat, base.lower()):
            y = year_now.group(0)
            if re.search(r"\?\s*$", last):
                reform = re.sub(r"\?\s*$", f" en {y} ?", last)
            else:
                reform = f"{last} en {y}"
            logger.debug("Reformulated contextual query (year injection)", extra={"query": reform})
            return reform

        # Pure referential follow-up without explicit year -> concatenate conservatively
        if re.search(ref_pat, base.lower()):
            reform = f"{last} — {base}"
            logger.debug("Reformulated contextual query (referential)", extra={"query": reform})
            return reform

        # Otherwise, keep the user's current question as-is
        return base

    # ------------------------------------------------------------------
    # Follow-up detection
    # ------------------------------------------------------------------

    def update_retrieval_feedback(self, evidence_count: int, confidence: float) -> None:
        """Store the latest retrieval results to inform next turn."""
        self.last_evidence_count = evidence_count
        self.last_confidence = confidence

    def follow_up_needed(self) -> bool:
        """Heuristic: decide if a follow-up retrieval or clarification is needed."""
        if self.last_evidence_count is None or self.last_confidence is None:
            return False
        # If too few evidence or low confidence, suggest clarifying
        if self.last_evidence_count < 3 or self.last_confidence < 0.3:
            logger.info(
                "Low-confidence retrieval detected",
                extra={
                    "evidence": self.last_evidence_count,
                    "confidence": self.last_confidence,
                },
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Suggestions for user (optional)
    # ------------------------------------------------------------------

    def suggest_followup(self) -> Optional[List[str]]:
        """Suggest possible follow-up queries if confidence is low."""
        if not self.follow_up_needed():
            return None

        last = self.get_last_user_query() or ""
        suggestions = [
            f"{last} en 2024",
            f"{last} dans d'autres régions",
            f"Quelles sont les causes inverses ?",
        ]
        return suggestions