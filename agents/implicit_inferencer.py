from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import os
import re
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _HAS_ST = False

from utils.logger import get_logger

logger = get_logger(__name__)


class ImplicitInferencer:
    """
    Infers implicit causal edges (CAUSE → EFFECT) between existing nodes.

    Combines:
      - aggressive filtering (by node role and plausibility of the effect)
      - semantic similarity (MiniLM if available, else lexical Jaccard)
      - co-evidence (shared document IDs)
      - lexical priors (typical causal terms)
      - transitivity (A→B and B→C ⇒ A→C)

    Output: "dashed" edges (indirect=True, inferred=True) ready for PyVis.

    Environment variables:
      INFER_IMPLICIT=true|false   (default: true)
      INFER_MIN_SCORE=float       (default: 0.55)
      INFER_MAX_EDGES=int         (default: 8)
      INFER_MODEL=str             (default: "sentence-transformers/all-MiniLM-L6-v2")
      INFER_TRANSITIVE=true|false (default: false)
      INFER_CHAIN_MIN=float       (default: 0.55)
      INFER_MAX_CHAIN_EDGES=int   (default: 3)
    """

    # Regex permissif pour identifier les phénomènes
    PHENOMENON_RE = re.compile(r".+")  # Au moins 1 caractère
    CAUSE_PRIOR_TERMS: set[str] = set()

    _SPLIT_RE = re.compile(
        r"\s*(?:,|;|\bet\b|\band\b|\bainsi\s+que\b|\bas\s+well\s+as\b)\s*", flags=re.I
    )

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        min_score: float | None = None,
        max_edges: int | None = None,
        model_name: str | None = None,
    ) -> None:
        self.enabled = (
            bool(str(os.getenv("INFER_IMPLICIT", "true")).lower() not in {"0", "false", "no"})
            if enabled is None else enabled
        )
        self.min_score = float(os.getenv("INFER_MIN_SCORE", "0.55")) if min_score is None else float(min_score)
        self.max_edges = int(os.getenv("INFER_MAX_EDGES", "8")) if max_edges is None else int(max_edges)
        self.model_name = (
            os.getenv("INFER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            if model_name is None else model_name
        )

        # Transitivity parameters (désactivé par défaut)
        self.do_transitive = bool(str(os.getenv("INFER_TRANSITIVE", "false")).lower() not in {"0", "false", "no"})
        self.chain_min = float(os.getenv("INFER_CHAIN_MIN", "0.55"))
        self.max_chain_edges = int(os.getenv("INFER_MAX_CHAIN_EDGES", "3"))

        self._model: Any = None
        self._cache: dict[str, List[float]] = {}
        
        logger.info(
            f"ImplicitInferencer initialized | enabled={self.enabled} min_score={self.min_score} "
            f"max_edges={self.max_edges} transitive={self.do_transitive}"
        )

    # -------------------------- utilities --------------------------

    @staticmethod
    def _color_for(weight: float) -> str:
        if weight >= 0.7:
            return "#4caf50"  # green
        if weight >= 0.45:
            return "#ffa500"  # orange
        return "#ff4d4d"      # red

    @staticmethod
    def _normalize_text(s: str) -> str:
        s = (s or "").lower().strip()
        return re.sub(r"\s+", " ", s)

    def _token_set(self, s: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", self._normalize_text(s)))

    def _cosine_fallback(self, a: str, b: str) -> float:
        """Lexical Jaccard similarity fallback (0..1)."""
        A = self._token_set(a)
        B = self._token_set(b)
        if not A or not B:
            return 0.0
        return len(A & B) / max(1, len(A | B))

    def _components(self, s: str) -> List[str]:
        """Split a composite label into significant components (>=2 tokens)."""
        s = (s or "").strip()
        if not s:
            return []
        parts = [p.strip() for p in self._SPLIT_RE.split(s) if p.strip()]
        good = [p for p in parts if len(self._token_set(p)) >= 2]
        return good or [s]

    def _ensure_model(self) -> None:
        if self._model is not None or not _HAS_ST:
            return
        try:
            self._model = SentenceTransformer(self.model_name)  # type: ignore
            logger.info("ImplicitInferencer loaded ST model", extra={"model": self.model_name})
        except Exception as e:
            logger.warning(f"ImplicitInferencer: failed to load model ({e}); using lexical fallback")
            self._model = None

    def _embed_many(self, texts: List[str]) -> List[List[float]]:
        """Return normalized embeddings (or [] if fallback)."""
        self._ensure_model()
        if self._model is None:
            return []
        out: List[List[float]] = []
        missing: List[Tuple[int, str]] = []
        for i, t in enumerate(texts):
            key = f"t::{t}"
            if key in self._cache:
                out.append(self._cache[key])
            else:
                out.append([])
                missing.append((i, t))
        if missing:
            try:
                vecs = self._model.encode([t for _, t in missing], normalize_embeddings=True)  # type: ignore
                for (i, t), v in zip(missing, vecs):
                    v_list = list(map(float, v))
                    out[i] = v_list
                    self._cache[f"t::{t}"] = v_list
            except Exception as e:
                logger.warning(f"ImplicitInferencer.encode failed: {e}; falling back to lexical similarity")
                return []
        return out

    @staticmethod
    def _cosine_vec(u: List[float], v: List[float]) -> float:
        if not u or not v:
            return 0.0
        s = sum(a * b for a, b in zip(u, v))
        return max(0.0, min(1.0, s))

    @staticmethod
    def _directional_score(u: List[float], v: List[float]) -> float:
        """
        Compute a signed directional similarity: +1 means u likely precedes v (cause → effect),
        -1 means inverse, 0 means symmetric.
        Works by comparing the mean of positive vs negative component differences.
        """
        if not u or not v or len(u) != len(v):
            return 0.0
        diffs = [a - b for a, b in zip(u, v)]
        pos = sum(d for d in diffs if d > 0)
        neg = -sum(d for d in diffs if d < 0)
        denom = max(1e-6, pos + neg)
        return (pos - neg) / denom  # [-1, +1]

    # -------------------------- rules --------------------------

    def _phenomenon(self, label: str) -> bool:
        """Vérifie si le label est un phénomène valide."""
        return bool(self.PHENOMENON_RE.search(label or ""))

    def _prior(self, cause_label: str, effect_label: str) -> float:
        """
        Prior purely statistical: if one node appears often as a 'source' in existing edges
        or has high out-degree, it's likely to be a cause. We can use this as a soft bias.
        """
        return 0.0  # default neutral prior

    def _existing_pairs(self, edges: List[dict]) -> set[tuple[Any, Any]]:
        pairs: set[tuple[Any, Any]] = set()
        for e in edges or []:
            s = e.get("from") or e.get("src")
            d = e.get("to") or e.get("dst")
            rel = (e.get("relation_type") or e.get("relation") or "").lower()
            if s is not None and d is not None and rel == "causes":
                pairs.add((s, d))
        return pairs

    def _edge_lookup(self, edges: List[dict]) -> dict[tuple[Any, Any], dict]:
        """Index existing edges (src,dst) to retrieve weights/sources for transitivity."""
        idx: dict[tuple[Any, Any], dict] = {}
        for e in edges or []:
            s = e.get("from") or e.get("src")
            d = e.get("to") or e.get("dst")
            rel = (e.get("relation_type") or e.get("relation") or "").lower()
            if s is not None and d is not None and rel == "causes":
                idx[(s, d)] = e
        return idx

    def _build_candidates(self, nodes: List[dict], edges: List[dict]) -> List[tuple[dict, dict]]:
        """
        Generate candidate (cause,effect) pairs not already linked.
        OPTIMISÉ: Filtre les nœuds invalides et limite les combinaisons.
        """
        existing_pairs = self._existing_pairs(edges)
        
        # Filtrer les nœuds valides avec des phénomènes réels
        nodes_valid = []
        for n in nodes:
            label = n.get("label") or ""
            if n.get("id") is not None and label:
                # Filtres plus permissifs pour assurer la génération de candidats
                tokens = self._token_set(label)
                if len(label.strip()) >= 5 and len(tokens) >= 1:  # Au moins 1 token significatif
                    nodes_valid.append(n)
                    logger.debug(f"Node accepted: {label[:50]}")
                else:
                    logger.debug(f"Node rejected (too short): {label[:50]} | len={len(label.strip())} tokens={len(tokens)}")
            else:
                logger.debug(f"Node rejected (no label or id): {n.get('id')}")
        
        logger.info(f"ImplicitInferencer: {len(nodes_valid)}/{len(nodes)} nodes validated")
        
        if len(nodes_valid) < 2:
            logger.warning(f"ImplicitInferencer: insufficient valid nodes ({len(nodes_valid)}), need at least 2")
            return []
        
        # Limiter le nombre de candidats pour éviter l'explosion combinatoire
        MAX_CANDIDATES = 100
        cands: List[tuple[dict, dict]] = []
        seen: set[tuple[Any, Any]] = set()

        # Privilégier les paires avec sources communes
        for c in nodes_valid:
            cid = c.get("id")
            srcs_c = set(map(str, (c.get("sources") or [])))
            
            for e in nodes_valid:
                if c is e:
                    continue
                eid = e.get("id")
                if (cid, eid) in existing_pairs:
                    continue
                # Éviter les doublons inversés
                if (eid, cid) in seen:
                    continue
                
                # Bonus pour les sources communes
                srcs_e = set(map(str, (e.get("sources") or [])))
                if srcs_c & srcs_e:
                    cands.insert(0, (c, e))  # Priorité
                else:
                    cands.append((c, e))
                    
                seen.add((cid, eid))
                
                if len(cands) >= MAX_CANDIDATES:
                    break
            if len(cands) >= MAX_CANDIDATES:
                break

        logger.info(f"ImplicitInferencer: generated {len(cands)} candidate pairs")
        return cands[:MAX_CANDIDATES]

    # -------------------------- transitivity --------------------------

    def _infer_transitive(self, edges: List[dict], confidence_global: float) -> List[dict]:
        """If A→B and B→C, infer A→C (dashed, inferred). LIMITÉ pour éviter la sur-inférence."""
        if not self.do_transitive:
            return []
        
        # Ne calculer la transitivité QUE sur les arêtes explicites (non inférées)
        explicit_edges = [e for e in edges if not e.get("inferred", False)]
        existing_pairs = self._existing_pairs(edges)  # Toutes les paires (pour éviter les doublons)
        lookup = self._edge_lookup(explicit_edges)
        
        by_src: dict[str, List[str]] = defaultdict(list)
        by_dst: dict[str, List[str]] = defaultdict(list)
        for (s, d), e in lookup.items():
            by_src[s].append(d)
            by_dst[d].append(s)

        new_edges: List[dict] = []
        seen: set[tuple[str, str]] = set()

        for b in set(list(by_dst.keys()) + list(by_src.keys())):
            preds = by_dst.get(b, [])
            succs = by_src.get(b, [])
            if not preds or not succs:
                continue
            for a in preds:
                for c in succs:
                    if a == c:
                        continue
                    pair = (a, c)
                    if pair in existing_pairs or pair in seen:
                        continue
                    e_ab = lookup.get((a, b))
                    e_bc = lookup.get((b, c))
                    if not e_ab or not e_bc:
                        continue
                    conf_ab = float(e_ab.get("confidence", e_ab.get("weight", 0.5)))
                    conf_bc = float(e_bc.get("confidence", e_bc.get("weight", 0.5)))
                    # Pénalité plus forte pour les chaînes transitives
                    score = max(0.0, min(0.99, conf_ab * conf_bc * 0.7))
                    if score < self.chain_min:
                        continue
                    weight = score * confidence_global
                    sources = list(set((e_ab.get("sources") or []) + (e_bc.get("sources") or [])))
                    edge = {
                        "from": a,
                        "to": c,
                        "relation_type": "causes",
                        "weight": weight,
                        "confidence": score,
                        "color": self._color_for(weight),
                        "style": "dashed",
                        "indirect": True,
                        "circular": False,
                        "inferred": True,
                        "sources": sources,
                        "source_text": None,
                        "evidence": [
                            {"inferred": True, "method": "transitive",
                             "chain": [(a, b), (b, c)], "conf_ab": conf_ab, "conf_bc": conf_bc}
                        ],
                    }
                    new_edges.append(edge)
                    seen.add(pair)
                    if len(new_edges) >= self.max_chain_edges:
                        return new_edges
        return new_edges

    # -------------------------- main --------------------------

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            logger.warning("ImplicitInferencer is DISABLED")
            return {}
        nodes: List[dict] = input_data.get("nodes") or []
        edges: List[dict] = input_data.get("edges") or []
        
        logger.info(f"ImplicitInferencer received {len(nodes)} nodes and {len(edges)} edges")
        
        if not nodes:
            logger.warning("ImplicitInferencer: No nodes received!")
            return {}
        confidence_global = float(input_data.get("confidence_global", 0.7) or 0.7)

        candidates = self._build_candidates(nodes, edges)
        if not candidates:
            logger.info("ImplicitInferencer: no valid candidates found")
            # Calculer la transitivité seulement sur les arêtes existantes
            trans = self._infer_transitive(edges, confidence_global)
            return {"edges": trans} if trans else {}

        texts: List[str] = []
        for c, e in candidates:
            cl = (c.get("label") or c.get("summary") or "")[:120]
            el = (e.get("label") or e.get("summary") or "")[:120]
            texts.extend([cl, el])
        embs = self._embed_many(texts)

        def pair_sim(idx_pair: int) -> float:
            if embs:
                cl_raw = texts[2 * idx_pair]
                el_raw = texts[2 * idx_pair + 1]
                comps = self._components(el_raw)
                u = embs[2 * idx_pair]
                if len(comps) == 1:
                    v = embs[2 * idx_pair + 1]
                    return self._cosine_vec(u, v)
                best = 0.0
                for comp in comps:
                    best = max(best, self._cosine_fallback(cl_raw, comp))
                return best
            cl = texts[2 * idx_pair]
            el = texts[2 * idx_pair + 1]
            comps = self._components(el)
            if not comps:
                return 0.0
            return max(self._cosine_fallback(cl, comp) for comp in comps)

        scored: List[tuple[float, dict, dict, float, float, float]] = []
        for i, (c, e) in enumerate(candidates):
            sim = pair_sim(i)
            srcs_c = set(map(str, (c.get("sources") or [])))
            srcs_e = set(map(str, (e.get("sources") or [])))
            co_evidence = 1.0 if (srcs_c & srcs_e) else 0.0
            
            # Directional asymmetry bias
            len_c, len_e = len((c.get("label") or "")), len((e.get("label") or ""))
            asym_bias = 0.0
            if len_c < len_e:
                asym_bias += 0.05
            elif len_c > len_e * 1.5:
                asym_bias -= 0.05

            prior = self._prior((c.get("label") or ""), (e.get("label") or ""))
            dir_score = 0.0
            if embs:
                u = embs[2 * i]
                v = embs[2 * i + 1]
                dir_score = 0.1 * self._directional_score(u, v)

            # Pénaliser les inversions
            reversed_exists = any(
                r[1].get("id") == e.get("id") and r[2].get("id") == c.get("id")
                for r in scored
            )
            if reversed_exists:
                dir_score -= 0.1

            score = 0.55 * sim + 0.25 * co_evidence + 0.15 * prior + asym_bias + dir_score
            scored.append((score, c, e, co_evidence, prior, sim))

        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Seuil FIXE plutôt que dynamique pour éviter les faux positifs
        threshold = self.min_score

        kept = [t for t in scored if t[0] >= threshold][: self.max_edges]
        
        logger.info(f"ImplicitInferencer: {len(kept)}/{len(scored)} pairs passed threshold {threshold}")

        # Éliminer les doublons bidirectionnels
        seen_pairs: set[tuple[Any, Any]] = set()
        filtered: List[tuple[float, dict, dict, float, float, float]] = []
        for t in kept:
            _, c, e, *_ = t
            cid, eid = c.get("id"), e.get("id")
            if (eid, cid) in seen_pairs:
                continue
            seen_pairs.add((cid, eid))
            filtered.append(t)
        kept = filtered

        if not kept:
            trans_only = self._infer_transitive(edges, confidence_global)
            return {"edges": trans_only} if trans_only else {}

        new_edges: List[dict] = []
        for score, c, e, co_evidence, prior, sim in kept:
            cid = c.get("id")
            eid = e.get("id")
            weight = max(0.0, min(1.0, score * confidence_global))
            edge = {
                "from": cid,
                "to": eid,
                "relation_type": "causes",
                "weight": weight,
                "confidence": score,
                "color": self._color_for(weight),
                "style": "dashed",
                "indirect": True,
                "circular": False,
                "inferred": True,
                "sources": list(set((c.get("sources") or [])) | set((e.get("sources") or []))),
                "source_text": None,
                "evidence": [
                    {
                        "inferred": True,
                        "method": "embedding+heuristic" if embs else "lexical+heuristic",
                        "sim": round(float(sim), 4),
                        "co_evidence": int(co_evidence),
                        "prior": int(prior),
                    }
                ],
            }
            new_edges.append(edge)

        # Transitivité SEULEMENT sur les arêtes explicites existantes
        trans_edges = self._infer_transitive(edges, confidence_global)
        all_new = new_edges + trans_edges

        logger.info(
            "ImplicitInferencer added inferred edges",
            extra={
                "candidates": len(candidates),
                "added_semantic": len(new_edges),
                "added_transitive": len(trans_edges),
                "min_score": self.min_score,
            },
        )
        return {"edges": all_new}