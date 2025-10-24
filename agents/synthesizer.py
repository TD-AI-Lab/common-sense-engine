"""
Synthesizer Agent
-----------------
Purpose:
Generate a short, grounded explanation (4–6 sentences) that references nodes/edges,
synthesizes causal relationships, and cites sources.

This agent operates in two modes:
- LLM mode (Gemini reasoning) if API key available
- Template fallback mode (offline) otherwise
"""

from typing import Any, Dict, List, Optional
import random
import re

from utils.logger import get_logger
from utils.metrics import timed
from utils.config import Settings

logger = get_logger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Deterministic randomness for stable tests/outputs
_RNG = random.Random(0)


class Synthesizer:
    """Hybrid causal reasoning agent.
    
    Combines LLM-based explanation (Gemini) with a deterministic template fallback.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        debug = bool(getattr(self.settings, "DEBUG", False))

        self.gemini_enabled = (
            GEMINI_AVAILABLE
            and getattr(self.settings, "GEMINI_API_KEY", None)
            and len(self.settings.GEMINI_API_KEY) > 5
        )

        if self.gemini_enabled:
            try:
                logger.info(f"Gemini SDK version: {getattr(genai, '__version__', '?')}")
                logger.info(f"Gemini module path: {getattr(genai, '__file__', '?')}")
                _key = (self.settings.GEMINI_API_KEY or "").strip().strip('"').strip("'")
                logger.info(f"GEMINI_API_KEY provided: {bool(_key)}")

                genai.configure(api_key=_key)
                logger.info("genai.configure() done.")

                if debug:
                    try:
                        models = [
                            m.name for m in genai.list_models()
                            if 'generateContent' in getattr(m, 'supported_generation_methods', [])
                        ]
                        logger.info(f"Models with generateContent (subset): {models[:8]}")
                    except Exception as e:
                        logger.warning(f"genai.list_models() failed: {e}")

                model_name = "gemini-2.5-flash"
                self.model = genai.GenerativeModel(model_name)
                logger.info(f"Using Gemini model: {model_name}")

                if debug:
                    try:
                        ping = self.model.generate_content("Ping").text
                        logger.info(f"Warmup OK, sample response: {ping[:60]}…")
                    except Exception as e:
                        logger.warning(f"Warmup generate_content failed: {e}")

                logger.info("Synthesizer connected to Gemini model.")
            except Exception as e:
                logger.error(f"Gemini initialization failed: {e}")
                self.gemini_enabled = False
        else:
            logger.info("Synthesizer running in template-only mode (no Gemini key detected).")

    # ----------------------------------------------------------------------

    def _edge_confidence_mean(self, edges: List[Dict[str, Any]]) -> float:
        """Compute mean confidence across all edges."""
        vals = [float(e.get("confidence", 0.0)) for e in edges if e.get("confidence") is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    def _fact_confidence_mean(self, facts: List[Dict[str, Any]]) -> float:
        """Compute mean confidence from ElasticSearch document scores."""
        if not facts:
            return 0.0
        vals = [float(f.get("score", 0.0)) for f in facts if f.get("score") is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    def _pick_sources(
        self, nodes: List[Dict[str, Any]], facts: List[Dict[str, Any]], max_sources=4
    ) -> List[str]:
        """Select representative source URLs, mapping node doc_ids -> URLs when possible."""
        urls: List[str] = []
        id2url = {}
        for f in facts:
            fid = f.get("id")
            surl = f.get("source_url")
            if fid:
                id2url[fid] = surl
            if surl and surl not in urls:
                urls.append(surl)
            if len(urls) >= max_sources:
                return urls[:max_sources]

        for n in nodes:
            for src in n.get("sources", []):
                url = id2url.get(src)
                if not url and isinstance(src, str) and src.startswith("http"):
                    url = src
                if url and url not in urls:
                    urls.append(url)
                if len(urls) >= max_sources:
                    return urls[:max_sources]

        return urls[:max_sources]

    # ----------------------------------------------------------------------

    def _generate_template_summary(
        self, query: str, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> str:
        """Generate a simple fallback summary from nodes and edges."""
        if not nodes:
            return "Not enough information to generate an explanation."

        intro_templates = [
            f"To answer the question \"{query}\", the analysis of the evidence reveals a chain of causes and effects:",
            f"Regarding the question \"{query}\", several elements appear to be connected:",
            f"Based on the available facts, a coherent sequence of events emerges:",
        ]
        intro = _RNG.choice(intro_templates)

        node_labels = [n.get("label") or n.get("summary") for n in nodes]
        node_labels = [l.strip(". ") for l in node_labels if l]

        sentences = []
        if len(node_labels) == 1:
            sentences.append(f"The main identified event is: {node_labels[0]}.")
        else:
            for i in range(len(node_labels) - 1):
                src = node_labels[i]
                dst = node_labels[i + 1]
                sentences.append(f"{src} likely led to {dst}.")

        outro_templates = [
            "This sequence highlights the causal logic between these phenomena.",
            "Overall, it suggests a consistent chain of events over time.",
            "These relationships could help explain the observed situation.",
        ]
        outro = _RNG.choice(outro_templates)
        body = " ".join(sentences[:4])
        return f"{intro} {body} {outro}"

    # ----------------------------------------------------------------------

    def _edge_lines(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[str]:
        """Format edges as readable causal lines using node labels."""
        id2label = {n["id"]: (n.get("label") or n.get("summary") or n["id"]) for n in nodes}
        lines: List[str] = []
        for e in edges[:10]:
            src_id = e.get("src") or e.get("source")
            dst_id = e.get("dst") or e.get("target")
            if not (src_id and dst_id):
                continue
            src = id2label.get(src_id, src_id)
            dst = id2label.get(dst_id, dst_id)
            conf = e.get("confidence", 0.0)
            lines.append(f"- {src} → {dst} (confidence: {conf})")
        return lines

    def _build_gemini_prompt(
        self, query: str, facts: List[Dict[str, Any]], edges: List[Dict[str, Any]], nodes: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Construct a clean, structured Gemini prompt (English version)."""
        prompt = (
            "You are a causal reasoning engine integrated into a scientific system.\n"
            "Your role is to produce a short, factual explanation (4–6 sentences) based only on the provided evidence.\n"
            "Do not invent anything — instead, logically connect facts together and cite relevant sources.\n"
            "Respond in clear, neutral, and concise English.\n\n"
            f"Question: {query}\n\n"
        )

        if facts:
            prompt += "Available facts:\n"
            for f in facts[:10]:
                prompt += f"- {f.get('title', '')} — {f.get('content', '')[:200]} (score: {f.get('score')}, source: {f.get('source_url')})\n"
            prompt += "\n"

        if edges and nodes:
            prompt += "Detected causal relations:\n"
            for line in self._edge_lines(nodes or [], edges):
                prompt += line + "\n"
            prompt += "\n"

        prompt += (
            "Based only on this information:\n"
            "1. Explain the probable cause of the phenomenon asked in the question.\n"
            "2. Describe the causal logic without adding new assumptions.\n"
            "3. End with an overall confidence score (0 to 1).\n\n"
            "Expected output format:\n"
            "Answer: [4–6 clear sentences]\n"
            "Confidence: [value between 0 and 1]\n"
            "Sources: [list of URLs or doc_ids]\n"
        )

        return prompt

    # ----------------------------------------------------------------------

    @timed("synthesizer.process")
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main synthesis pipeline."""
        query = input_data.get("query", "").strip()
        facts = input_data.get("facts", [])
        nodes = input_data.get("nodes", [])
        edges = input_data.get("edges", [])

        if not query and not nodes:
            return {
                "summary": "No information available to produce an explanation.",
                "confidence_global": 0.0,
                "sources": [],
                "mode": "template",
            }

        # Compute base confidences
        confidence_edges = self._edge_confidence_mean(edges)
        confidence_facts = self._fact_confidence_mean(facts)
        confidence_global = round(0.7 * confidence_edges + 0.3 * confidence_facts, 3)

        sources = self._pick_sources(nodes, facts)
        mode = "template"
        summary = ""

        # Try LLM reasoning if available
        if self.gemini_enabled:
            try:
                prompt = self._build_gemini_prompt(query, facts, edges, nodes)
                result = self.model.generate_content(prompt)
                text = result.text.strip() if result and result.text else ""

                # Parse response
                match_answer = re.search(r"Answer\s*:(.*?)(Confidence|Sources|$)", text, re.S)
                match_conf = re.search(r"Confidence\s*:\s*([0-9.,]+)", text)
                match_src = re.search(r"Sources\s*:\s*(.*)", text)

                summary = match_answer.group(1).strip() if match_answer else text[:400]
                if match_conf:
                    confidence_global = float(match_conf.group(1).replace(",", "."))
                if match_src:
                    extra_sources = [s.strip() for s in match_src.group(1).split(",")]
                    sources.extend([s for s in extra_sources if s not in sources])

                mode = "gemini"

            except Exception as e:
                logger.warning(f"Gemini reasoning failed: {e}")
                summary = self._generate_template_summary(query, nodes, edges)
                mode = "template"
        else:
            summary = self._generate_template_summary(query, nodes, edges)

        if sources:
            summary += " Sources: " + ", ".join(sources[:4])

        logger.info(
            "Synthesizer produced explanation",
            extra={"mode": mode, "confidence": confidence_global, "sources": len(sources)},
        )

        return {
            "summary": summary.strip(),
            "confidence_global": round(confidence_global, 3),
            "sources": sources[:4],
            "mode": mode,
        }