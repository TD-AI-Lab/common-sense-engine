"""
Orchestrator
------------
Coordinates the full multi-agent reasoning pipeline of the Common Sense Engine.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable, Optional, List
import os
from contextlib import suppress

from types import SimpleNamespace
import asyncio
import inspect

from utils.logger import get_logger
from utils.metrics import timed

from agents.node_summarizer import NodeSummarizer
from agents.implicit_inferencer import ImplicitInferencer

logger = get_logger(__name__)


@runtime_checkable
class PipelineAgent(Protocol):
    """Contract for pipeline agents (FactFinder, CausalityMapper, TemporalOrganizer, Synthesizer)."""

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        ...


@runtime_checkable
class ConversationCallbacks(Protocol):
    """Narrow contract used from ConversationalCore without assuming a `process` method."""

    def update_retrieval_feedback(self, evidence_count: int, confidence: float) -> None: ...
    def follow_up_needed(self) -> bool: ...
    def suggest_followup(self) -> Optional[List[str]]: ...


class Orchestrator:
    """Run the multi-agent pipeline and collect partial failures.

    Pipeline sequence:
    1️⃣ FactFinder        → retrieves relevant facts & evidence
    2️⃣ CausalityMapper   → builds causal graph (nodes/edges)
    3️⃣ TemporalOrganizer → orders events chronologically
    4️⃣ Synthesizer       → generates narrative summary
    """

    def __init__(
        self,
        *,
        fact_finder: PipelineAgent,
        causality_mapper: PipelineAgent,
        temporal_organizer: PipelineAgent,
        synthesizer: PipelineAgent,
        conversation: Optional[ConversationCallbacks] = None,
        node_summarizer: Optional[NodeSummarizer] = None,
        implicit_inferencer: Optional[ImplicitInferencer] = None,
    ) -> None:
        self.fact_finder = fact_finder
        self.causality_mapper = causality_mapper
        self.temporal_organizer = temporal_organizer
        self.synthesizer = synthesizer
        self.conversation = conversation
        self._last_payload: dict | None = None
        # Allow DI of a configured NodeSummarizer (e.g., Gemini client/model/budget)
        self.node_summarizer = node_summarizer or NodeSummarizer()
        self.implicit_inferencer = implicit_inferencer or ImplicitInferencer()

    # Try to auto-attach a Gemini client if none provided and an API key exists.
    def _ensure_llm_on_summarizer(self) -> None:
        try:
            if not self.node_summarizer or getattr(self.node_summarizer, "llm", None):
                return
            # 1) tenter d'emprunter le client du Synthesizer (même modèle/config)
            for attr in ("llm", "model", "client", "gemini", "_client"):
                cand = getattr(self.synthesizer, attr, None)
                if cand and (hasattr(cand, "generate_content") or hasattr(cand, "generate") or callable(cand)):
                    self.node_summarizer.llm = cand
                    logger.info("Attached summarizer LLM from Synthesizer", extra={"attr": attr})
                    return
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            if not api_key:
                logger.warning("No GEMINI_API_KEY/GOOGLE_API_KEY found for NodeSummarizer; using local fallback")
                return
            with suppress(ImportError):
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.node_summarizer.llm = genai.GenerativeModel(model_name)
                logger.info("Attached default Gemini client to NodeSummarizer", extra={"model": model_name})
        except Exception as e:
            logger.warning(f"Could not attach a Gemini client automatically: {e}")

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _await_if_needed(value: Any) -> Any:
        """
        If `value` is awaitable, try to resolve it synchronously.
        If we are already inside a running loop, raise a clear error to force
        the caller to use an async pipeline rather than silently failing.
        """
        if inspect.isawaitable(value):
            try:
                return asyncio.run(value)
            except RuntimeError as e:
                raise RuntimeError(
                    "NodeSummarizer.summarize_nodes is async; run the pipeline in async "
                    "or inject a sync NodeSummarizer."
                ) from e
        return value
        
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_update(dst: Dict[str, Any], src: Any) -> List[str]:
        """Update dict if `src` is a dict. Returns any `errors` contained in `src`."""
        collected_errors: List[str] = []
        if isinstance(src, dict):
            # collect nested errors first, then update payload
            src_errors = src.get("errors")
            if isinstance(src_errors, list):
                collected_errors.extend(str(e) for e in src_errors)
            dst.update(src)
        else:
            collected_errors.append("Agent returned a non-dict payload")
        return collected_errors

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    @timed("orchestrator.run_explain")
    def run_explain(
        self,
        query: str,
        *,
        time_range: Optional[Dict[str, Any]] = None,
        k: int = 20,
        request_id: Optional[str] = None,
        live: bool = True,
        live_budget: Optional[Dict[str, Any]] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the multi-agent pipeline.

        Args:
            query: user query to explain
            time_range: optional {"from": ISO8601, "to": ISO8601}
            k: top docs to retrieve
            request_id: optional correlation id for logs/tracing
            extras: optional additional keys merged into the initial payload
        """
        payload: Dict[str, Any] = {
            "query": query,
            "time_range": time_range,
            "k": k,
            "live": live,
            "live_budget": (live_budget or {}),
        }
        if request_id:
            payload["request_id"] = request_id
        if isinstance(extras, dict) and extras:
            # Allow upstream to attach out-of-band data (e.g., live_ingested)
            payload.update(extras)
        errors: List[str] = []

        # --- Step 1: FactFinder ---
        try:
            logger.info(
                "Step 1/4: FactFinder",
                extra={
                    "query": query,
                    "k": k,
                    "time_range": time_range,
                    "request_id": request_id,
                    "live": live,
                },
            )
            ff = self.fact_finder.process(payload)
            errors += self._safe_update(payload, ff)
        except Exception as e:
            logger.exception("FactFinder failed")
            errors.append(f"FactFinder: {e}")
            payload.update({"facts": [], "evidence": [], "confidence_global": 0.0})

        # --- Step 2: CausalityMapper ---
        try:
            logger.info("Step 2/4: CausalityMapper", extra={"request_id": request_id})
            cm = self.causality_mapper.process(payload)
            errors += self._safe_update(payload, cm)
        except Exception as e:
            logger.exception("CausalityMapper failed")
            errors.append(f"CausalityMapper: {e}")
            payload.update({"nodes": [], "edges": []})

        # Log es_records (si fournis par le CausalityMapper)
        try:
            if isinstance(payload.get("es_records"), list):
                logger.info("Causality Mapper ES records", extra={"count": len(payload["es_records"])})
        except Exception:
            pass

        if payload.get("nodes"):
            logger.info(f"[DEBUG] Exemple du premier nœud avant résumés : {payload['nodes'][0]}")

        # --- NEW STEP: NodeSummarizer ---
        try:
            logger.info(
                "Step 2b/4: NodeSummarizer",
                extra={
                    "request_id": request_id,
                    "nodes_in": len(payload.get("nodes", []) or []),
                },
            )
            nodes = payload.get("nodes") or []
            if nodes:
                # si aucun client LLM n'a été passé, on tente un attachement auto (GEMINI_API_KEY)
                self._ensure_llm_on_summarizer()
                if not getattr(self.node_summarizer, "llm", None):
                    logger.warning("NodeSummarizer will use local fallback (no LLM configured)", extra={"nodes_in": len(nodes)})
                logger.info("Before summarizer sample", extra={"node0": nodes[0] if nodes else None})
                result = self._await_if_needed(
                    self.node_summarizer.summarize_nodes(nodes)
                )
                payload["_nodes_summarized"] = True

                # Accept either: list[str] (labels), list[dict], or {"nodes": ...}
                summarized_nodes = (
                    result.get("nodes")
                    if isinstance(result, dict) and "nodes" in result
                    else result
                )

                if not summarized_nodes:
                    logger.info("NodeSummarizer returned empty output; keeping original nodes")
                elif isinstance(summarized_nodes, list) and summarized_nodes and isinstance(summarized_nodes[0], str):
                    # list[str] -> set as 'label' by position
                    for n, label in zip(nodes, summarized_nodes):
                        if isinstance(n, dict):
                            n["label"] = label
                elif isinstance(summarized_nodes, list) and summarized_nodes and isinstance(summarized_nodes[0], dict):
                    # list[dict] -> merge by 'id' without destroying required fields
                    by_id = {n["id"]: n for n in nodes if isinstance(n, dict) and "id" in n}
                    updated = 0
                    for sn in summarized_nodes:
                        if not isinstance(sn, dict):
                            continue
                        nid = sn.get("id")
                        if nid is None or nid not in by_id:
                            continue
                        target = by_id[nid]
                        # Prefer explicit 'label' then 'summary'
                        if "label" in sn and sn["label"] is not None:
                            target["label"] = sn["label"]
                        elif "summary" in sn and sn["summary"] is not None:
                            target["label"] = sn["summary"]
                        # Merge remaining fields conservatively (preserve 'id' and 'text')
                        for k, v in sn.items():
                            if k not in ("id", "text") and v is not None:
                                target[k] = v
                        updated += 1
                    logger.info("NodeSummarizer merged nodes", extra={"updated": updated})
                else:
                    raise TypeError("Unsupported NodeSummarizer output format")

                payload["nodes"] = nodes
                if nodes:
                    logger.info("After summarizer sample", extra={"node0": nodes[0]})
        except Exception as e:
            logger.warning(f"NodeSummarizer failed: {e}")
            errors.append(f"NodeSummarizer: {e}")

        # --- Post-summarization cleanup: factoriser nœuds & remapper arêtes ---
        try:
            if payload.get("nodes"):
                cleaner = getattr(self.causality_mapper, "cleanup_after_summarization", None)
                if callable(cleaner):
                    n_before = len(payload.get("nodes") or [])
                    e_before = len(payload.get("edges") or [])
                    new_nodes, new_edges = cleaner(payload.get("nodes") or [], payload.get("edges") or [])
                    payload["nodes"], payload["edges"] = new_nodes, new_edges
                    logger.info(
                        "Post-summarization cleanup",
                        extra={"nodes": f"{n_before}->{len(new_nodes)}", "edges": f"{e_before}->{len(new_edges)}"},
                    )
        except Exception as e:
            logger.warning(f"Post-summarization cleanup failed: {e}")

        # --- Step 2c: ImplicitInferencer (optionnel) ---
        try:
            if getattr(self, "implicit_inferencer", None):
                logger.info(
                    "Step 2c/4: ImplicitInferencer",
                    extra={
                        "request_id": request_id,
                        "nodes_in": len(payload.get("nodes", []) or []),
                        "edges_in": len(payload.get("edges", []) or []),
                    },
                )
                ii = self.implicit_inferencer.process(payload)
                if isinstance(ii, dict):
                    added = ii.get("edges") or ii.get("implicit_edges") or []
                    if added:
                        payload.setdefault("edges", []).extend(added)
                        logger.info(
                            "ImplicitInferencer merged edges",
                            extra={"added": len(added), "total_edges": len(payload.get("edges") or [])},
                        )
                        # Déduplication (garde l'arête au meilleur 'confidence')
                        try:
                            dedup: dict[tuple[str, str, str], dict] = {}
                            for e in payload.get("edges") or []:
                                s = e.get("from") or e.get("src")
                                d = e.get("to") or e.get("dst")
                                r = (e.get("relation_type") or e.get("relation") or "causes").lower()
                                if not s or not d:
                                    continue
                                key = (str(s), str(d), r)
                                cur = dedup.get(key)
                                score = float(e.get("confidence", e.get("weight", 0.0)) or 0.0)
                                if cur is None or score > float(cur.get("confidence", cur.get("weight", 0.0)) or 0.0):
                                    dedup[key] = e
                            payload["edges"] = list(dedup.values())
                        except Exception as de:
                            logger.warning(f"Edge dedup failed: {de}")
        except Exception as e:
            logger.warning(f"ImplicitInferencer failed: {e}")

        # --- Step 3: TemporalOrganizer ---
        try:
            logger.info("Step 3/4: TemporalOrganizer", extra={"request_id": request_id})
            to = self.temporal_organizer.process(payload)
            # ⚠️ ne pas écraser les nœuds déjà résumés
            if isinstance(to, dict) and payload.get("_nodes_summarized") and "nodes" in to:
                logger.info("TemporalOrganizer returned nodes; keeping summarized nodes already in payload")
                to = dict(to)
                to.pop("nodes", None)

            # ⚠️ préserver les edges construits par le CausalityMapper
            if isinstance(to, dict) and "edges" in to and payload.get("edges"):
                logger.info("TemporalOrganizer returned edges; keeping existing causal edges")
                to = dict(to)
                to.pop("edges", None)

            errors += self._safe_update(payload, to)
        except Exception as e:
            logger.exception("TemporalOrganizer failed")
            errors.append(f"TemporalOrganizer: {e}")

        # --- Step 4: Synthesizer ---
        try:
            logger.info("Step 4/4: Synthesizer", extra={"request_id": request_id})
            sy = self.synthesizer.process(payload)
            # par sûreté : on ignore d'éventuels `nodes` renvoyés ici aussi
            if isinstance(sy, dict) and payload.get("_nodes_summarized") and "nodes" in sy:
                logger.info("Synthesizer returned nodes; keeping summarized nodes already in payload")
                sy = dict(sy)
                sy.pop("nodes", None)
            # et on préserve aussi les edges causaux
            if isinstance(sy, dict) and "edges" in sy and payload.get("edges"):
                logger.info("Synthesizer returned edges; keeping existing causal edges")
                sy = dict(sy)
                sy.pop("edges", None)
            errors += self._safe_update(payload, sy)
        except Exception as e:
            logger.exception("Synthesizer failed")
            errors.append(f"Synthesizer: {e}")
            payload.update({"summary": "", "confidence_global": 0.0})

        # --- Optional: conversational follow-up ---
        if self.conversation:
            try:
                evidence_count = len(payload.get("evidence", []))
                confidence = float(payload.get("confidence_global", 0.0) or 0.0)
                self.conversation.update_retrieval_feedback(evidence_count, confidence)
                if self.conversation.follow_up_needed():
                    payload["followup_suggestions"] = self.conversation.suggest_followup()
            except Exception as e:
                logger.warning(f"Conversational feedback failed: {e}")
                errors.append(f"Conversation: {e}")

        # --- Finalize payload ---
        if errors:
            # merge with any existing errors on payload (if agents set it)
            existing = payload.get("errors", [])
            if isinstance(existing, list):
                payload["errors"] = existing + errors
            else:
                payload["errors"] = errors

        logger.info(
            "Pipeline completed",
            extra={
                "evidence": len(payload.get("evidence", [])),
                "nodes": len(payload.get("nodes", [])),
                "edges": len(payload.get("edges", [])),
                "confidence": payload.get("confidence_global", 0.0),
                "errors": len(payload.get("errors", [])) if isinstance(payload.get("errors"), list) else 0,
                "request_id": request_id,
            },
        )

        # --- Cache the last pipeline result for inspection via /sources/{node_id} ---
        try:
            # shallow copy only, avoids large memory use with embeddings
            self._last_payload = dict(payload)
        except Exception as e:
            logger.warning(f"Failed to store last payload: {e}")
            self._last_payload = None

        return payload