from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple, Sequence, cast 
import re
from utils.logger import get_logger

logger = get_logger(__name__)

class NodeSummarizer:
    """
    Automatic node label summarization using an LLM (Gemini or heuristic fallback).
    - Selects the best source text and always goes through the LLM (local fallback on failure).
    - Tolerates different LLM output formats and falls back to a deterministic local summary if needed.
    - Mirrors the final label into renderer-friendly fields (summary/title/text).
    """

    # Potential fields used as text sources, ordered by priority
    DEFAULT_TEXT_FIELDS: List[str] = [
        "text", "raw", "sentence", "content",  # rich text first
        "summary", "title", "label",           # short fields later
    ]

    # Editorial noise prefixes to remove (FR/EN)
    NOISE_PREFIX_RE = re.compile(
        r"^\s*(?:à\s*lire\s*aussi|a\s*lire\s*aussi|lire\s*aussi|japan\s*data|opinion|analyse|tribune|édito|edito|podcast)\s*:?\s*",
        flags=re.I,
    )

    # Separators and causal connectors to exclude from final labels
    _SEP_RE = re.compile(r"\s*[-—–:]\s*")
    _CAUSAL_CONNECTORS = re.compile(
        r"(?:à\s+cause\s+de|en\s+raison\s+de|due?\s+à|par\s+manque\s+de|because\s+of|caused\s+by|due\s+to|thanks\s+to|gr[aà]ce\s+à)",
        flags=re.I,
    )

    def __init__(
        self,
        llm_client: Any = None,
        *,
        output_field: str = "label",
        text_fields: Optional[List[str]] = None,
        max_words: int = 8,
        max_chars: int = 80,
        mirror_fields: Optional[Sequence[str]] = ("title",),
    ) -> None:
        self.llm = llm_client
        self.output_field = output_field
        self.text_fields = text_fields or list(self.DEFAULT_TEXT_FIELDS)
        self.max_words = max_words
        self.max_chars = max_chars
        self.mirror_fields = tuple(mirror_fields or ())
        self.max_src_chars = 500
        self._cache: dict[str, str] = {}

    # ----------------------------- helpers -----------------------------
    _TAIL_DANGLING_RE = re.compile(
        r"(?:\b(?:à|de|du|des|la|le|les|au|aux|pour|par|avec|sans|sur|dans|chez|vers|en|d’|d')\s*)+$", re.I
    )

    @classmethod
    def _strip_noise(cls, s: str) -> str:
        """Remove editorial prefixes such as 'À lire aussi:' or 'Japan Data:'."""
        if not s:
            return s
        s2 = cls.NOISE_PREFIX_RE.sub("", s)
        s2 = re.sub(r"^[\s:;–—-]+", "", s2)
        return s2.strip()

    @staticmethod
    def _lang_heuristic(text: str) -> str:
        """Very lightweight FR/EN heuristic to select prompt language."""
        t = (text or "").lower()
        fr = len(re.findall(r"\b(le|la|les|des|une?|de|du|et|à|pour|pas|dans|avec)\b", t))
        en = len(re.findall(r"\b(the|a|an|and|to|for|not|of|in|on|with|by)\b", t))
        return "fr" if fr >= en else "en"

    @classmethod
    def _strip_causal_connectors(cls, s: str) -> str:
        """Remove causal connectors like 'because of', 'due to', etc."""
        if not isinstance(s, str):
            return s
        s2 = cls._CAUSAL_CONNECTORS.sub("", s)
        return re.sub(r"\s+", " ", s2).strip()

    @classmethod
    def _enforce_role(cls, label: str, role: str, source_text: str) -> str:
        """
        If the label looks like 'effect — cause', keep only the relevant half
        depending on the node role (cause/effect).
        """
        if not isinstance(label, str) or not label.strip():
            return label
        parts = cls._SEP_RE.split(label)
        if len(parts) >= 2:
            left, right = parts[0].strip(), parts[1].strip()
            if role == "cause":
                label = right or label
            elif role == "effect":
                label = left or label
            else:
                label = max([left, right], key=len)
        return cls._strip_causal_connectors(label).strip()

    def _complete_cause_from_source(self, source: str) -> str:
        """Extract a short cause (1–5 words) from the source sentence when possible."""
        if not isinstance(source, str):
            return ""
        text = " ".join(source.split())
        patterns = [
            r"à\s+cause\s+de\s+([^.,;:()\n]+)",
            r"en\s+raison\s+de\s+([^.,;:()\n]+)",
            r"due?\s+à\s+([^.,;:()\n]+)",
            r"par\s+manque\s+de\s+([^.,;:()\n]+)",
            r"because\s+of\s+([^.,;:()\n]+)",
            r"caused\s+by\s+([^.,;:()\n]+)",
            r"due\s+to\s+([^.,;:()\n]+)",
        ]
        for p in patterns:
            m = re.search(p, text, flags=re.I)
            if m:
                phrase = m.group(1).strip()
                words = phrase.split()
                if words:
                    return " ".join(words[:5]).strip(" ,;:.")
        return ""

    def _fix_dangling(self, label: str, source: str) -> str:
        """Fix or complete dangling prepositions in the label."""
        if not isinstance(label, str) or not label:
            return label
        lab = label.strip()
        if self._TAIL_DANGLING_RE.search(lab):
            addon = self._complete_cause_from_source(source or "")
            if addon:
                return (lab.rstrip() + " " + addon).strip()
            lab = self._TAIL_DANGLING_RE.sub("", lab).rstrip()
        return lab
 
    def _pick_text(self, node: dict) -> Tuple[str, str]:
        """
        Choose the best text field on the node.
        Prefers higher priority fields; if tie, takes the longest.
        Returns (text, field_name); empty if none found.
        """
        candidates: List[Tuple[str, str]] = []
        for key in self.text_fields:
            val = node.get(key)
            if isinstance(val, str):
                val = val.strip()
                if val:
                    candidates.append((val, key))
        if not candidates:
            return "", ""

        priority = {k: i for i, k in enumerate(self.text_fields)}
        best_val, best_key = sorted(
            candidates,
            key=lambda vk: (priority.get(vk[1], 10**6), -len(vk[0])),
        )[0]

        if len(best_val.split()) < 8 and isinstance(node.get("text"), str) and len(node["text"].split()) >= 12:
            return node["text"].strip(), "text"
        return best_val, best_key

    @staticmethod
    def _clean(s: str) -> str:
        """Light cleaning for standalone label strings."""
        if not s:
            return s
        s = s.strip().strip("«»\"'“”‘’")
        s = NodeSummarizer._strip_noise(s)
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"[.。…]+$", "", s).strip()
        return s

    def _shorten(self, s: str) -> str:
        """Trim string by word/char limits, avoiding unfinished prepositions."""
        words = s.split()
        if len(words) > self.max_words:
            s = " ".join(words[: self.max_words])
        if len(s) > self.max_chars:
            s = s[: self.max_chars].rstrip()
        s = self._TAIL_DANGLING_RE.sub("", s).rstrip()
        return s

    @staticmethod
    def _extract_text_from_llm_response(resp: Any) -> str:
        """Extract plain text only, ignoring structured SDK objects."""
        if resp is None:
            return ""
        try:
            t = getattr(resp, "text", None)
            if isinstance(t, str) and t.strip():
                return t.strip()
        except Exception:
            pass
        try:
            candidates = getattr(resp, "candidates", []) or []
            texts = []
            for c in candidates:
                content = getattr(c, "content", None)
                parts = getattr(content, "parts", []) if content else []
                for p in parts:
                    pt = getattr(p, "text", None)
                    if isinstance(pt, str) and pt.strip():
                        texts.append(pt.strip())
            if texts:
                return " ".join(texts)
        except Exception:
            pass
        return ""

    @staticmethod
    def _extract_finish_reason(resp: Any) -> Optional[int]:
        """Return Gemini finish_reason if available (1=STOP, 2=MAX_TOKENS, 3=SAFETY)."""
        try:
            cands = getattr(resp, "candidates", None) or []
            if not cands:
                return None
            fr = getattr(cands[0], "finish_reason", None)
            if isinstance(fr, int):
                return fr
            if isinstance(fr, str):
                return {"STOP":1, "MAX_TOKENS":2, "SAFETY":3}.get(fr.upper(), None)
        except Exception:
            pass
        return None

    def _call_llm(self, prompt: str) -> str:
        """Safely call the LLM client, tolerant to SDK differences."""
        if not self.llm:
            return ""
        try:
            if hasattr(self.llm, "generate"):
                resp = self.llm.generate(prompt, max_tokens=128)
            elif hasattr(self.llm, "generate_content"):
                try:
                    safety_settings = None
                    try:
                        from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore
                        safety_settings = [
                            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT,        "threshold": HarmBlockThreshold.BLOCK_NONE},
                            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,       "threshold": HarmBlockThreshold.BLOCK_NONE},
                            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                        ]
                    except Exception:
                        pass
                    out_tokens = 128 if len(prompt) < 500 else 256
                    resp = self.llm.generate_content(
                        prompt,
                        max_output_tokens=out_tokens,
                        temperature=0,
                        safety_settings=safety_settings,
                        generation_config={"response_mime_type": "text/plain"},
                    )
                except TypeError:
                    resp = self.llm.generate_content(prompt)
            elif callable(self.llm):
                resp = self.llm(prompt)
            else:
                logger.warning("LLM client has no supported interface; skipping.")
                return ""
            txt = self._extract_text_from_llm_response(resp)
            if txt:
                return txt
            fr = self._extract_finish_reason(resp)
            if fr == 2:
                logger.warning("NodeSummarizer: empty LLM text (MAX_TOKENS) -> retry compact prompt")
                short_prompt = prompt[:400] + "…" if len(prompt) > 400 else prompt
                try:
                    resp2 = self.llm.generate_content(
                        short_prompt,
                        generation_config={
                            "max_output_tokens": 256,
                            "temperature": 0,
                            "response_mime_type": "text/plain",
                        },
                    )
                    txt2 = self._extract_text_from_llm_response(resp2)
                    if txt2:
                        return txt2
                except Exception as e:
                    logger.warning(f"NodeSummarizer retry failed: {e}")
            else:
                logger.warning(f"NodeSummarizer: empty LLM text (finish_reason={fr})")
            return ""
        except Exception as e:
            logger.warning(f"NodeSummarizer LLM call failed: {e}")
            return ""

    # ----------------------------- public -----------------------------
    def summarize_nodes(self, nodes: list[dict]) -> list[dict]:
        summarized: list[dict] = []
        total = len(nodes or [])
        updated = 0
        if not self.llm:
            logger.warning(
                "NodeSummarizer: no LLM configured -> using local fallback shortening",
                extra={"nodes_in": total},
            )

        for n in nodes:
            text, src_key = self._pick_text(n)
            if not text:
                summarized.append(n)
                continue

            original_text = text
            cleaned_for_prompt = self._strip_noise(text)
            role = (n.get("role") or "unknown").lower()

            if len(cleaned_for_prompt) > self.max_src_chars:
                cleaned_for_prompt = cleaned_for_prompt[: self.max_src_chars] + "…"

            existing = (n.get(self.output_field) or "").strip()
            if existing and len(existing.split()) <= self.max_words and not self._TAIL_DANGLING_RE.search(existing):
                label = self._clean(existing)
            else:
                lang = self._lang_heuristic(cleaned_for_prompt)
                if lang == "fr":
                    # Keep FR fallback for bilingual robustness
                    if role == "cause":
                        prompt = (
                            "Lis le texte et donne une expression nominale courte (3–8 mots) "
                            "décrivant la CAUSE uniquement. "
                            "Exemples: 'canicule prolongée', 'offre limitée', 'demande accrue'.\n"
                            f"Texte: {cleaned_for_prompt[:500]}{'…' if len(cleaned_for_prompt) > 500 else ''}"
                        )
                    elif role == "effect":
                        prompt = (
                            "Lis le texte et donne une expression nominale courte (3–8 mots) "
                            "décrivant l'EFFET uniquement. "
                            "Exemples: 'hausse des prix du riz', 'pénurie de riz'.\n"
                            f"Texte: {cleaned_for_prompt[:500]}{'…' if len(cleaned_for_prompt) > 500 else ''}"
                        )
                    else:
                        prompt = (
                            "Résume le phénomène mentionné en une expression nominale courte (3–8 mots).\n"
                            f"Texte: {cleaned_for_prompt[:500]}{'…' if len(cleaned_for_prompt) > 500 else ''}"
                        )
                else:
                    if role == "cause":
                        prompt = (
                            "Read the text and output a short noun phrase (3–8 words) describing the CAUSE only. "
                            "No verbs, no dashes, no effects. Examples: 'prolonged heatwave', 'limited supply', 'increased demand'.\n"
                            f"Text: {cleaned_for_prompt[:500]}{'…' if len(cleaned_for_prompt) > 500 else ''}"
                        )
                    elif role == "effect":
                        prompt = (
                            "Read the text and output a short noun phrase (3–8 words) describing the EFFECT only. "
                            "No verbs, no dashes, no causes. Examples: 'rice price increase', 'rice shortage'.\n"
                            f"Text: {cleaned_for_prompt[:500]}{'…' if len(cleaned_for_prompt) > 500 else ''}"
                        )
                    else:
                        prompt = (
                            "Summarize the phenomenon as a short noun phrase (3–8 words). "
                            "No dashes, no combined cause-effect forms.\n"
                            f"Text: {cleaned_for_prompt[:500]}{'…' if len(cleaned_for_prompt) > 500 else ''}"
                        )

                label = self._cache.get(f"{role}::{cleaned_for_prompt}")
                if not label:
                    label = self._clean(self._call_llm(prompt))
                    if not label:
                        label = self._clean(self._shorten(cleaned_for_prompt))
                    self._cache[f"{role}::{cleaned_for_prompt}"] = label or ""

            if not label:
                label = self._clean(self._shorten(cleaned_for_prompt))

            label = self._enforce_role(label, role, cleaned_for_prompt or original_text)
            fixed = self._fix_dangling(label, cleaned_for_prompt or original_text)
            if fixed != label:
                logger.info("NodeSummarizer fixed dangling tail", extra={"before": label, "after": fixed})
            label = fixed

            if label:
                n[self.output_field] = label
                if isinstance(n.get("text"), str) and "full_text" not in n and n["text"] != label:
                    n["full_text"] = n["text"]
                for mf in self.mirror_fields:
                    if mf == self.output_field:
                        continue
                    try:
                        if mf not in ("summary", "text"):
                            n[mf] = label
                    except Exception:
                        pass
                updated += 1
            summarized.append(n)

        logger.info("NodeSummarizer done", extra={"nodes_in": total, "updated": updated})
        return summarized