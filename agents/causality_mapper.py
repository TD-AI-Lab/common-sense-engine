"""
Turn evidence sentences into a compact causal graph – v2.1 compliant.

Coverage:
- FR/EN causal patterns (L→R et R→L) + corrélations/contradictions
- Multi-relations par phrase (pas de break après 1er match)
- Négations et formulations spéculatives (hedges) → pénalités de confiance
- Lemmatisation & ID stables (md5(lemme canonisé))
- Dédoublonnage sémantique (lemma + unidecode + stopwords)
- Agrégation pondérée par evidence: weight = P_edge × mean(evidence_scores) × confidence_global
- Chaînage automatique A→C (dashed, poids=produit)
- Détection de cycles (NetworkX) → marquage circular
- Validation temporelle (dates rudimentaires) → pénalité si t_cause > t_effet
- Sortie prête pour PyVis (weight/color/source_text/sources/style)
- Préparation d’objets pour index ES `causal_relations`
"""

from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, field
import hashlib
import re
import math
import itertools
from collections import defaultdict
from datetime import datetime

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None

try:
    import spacy
    _NLP_FR = spacy.load("fr_core_news_sm")  # type: ignore
except Exception:  # pragma: no cover
    _NLP_FR = None
try:
    import spacy
    _NLP_EN = spacy.load("en_core_web_sm")  # type: ignore
except Exception:  # pragma: no cover
    _NLP_EN = None

try:
    from unidecode import unidecode  # type: ignore
except Exception:  # pragma: no cover
    def unidecode(x: str) -> str:
        return x

from utils.logger import get_logger
from utils.metrics import timed

logger = get_logger(__name__)

# En-têtes éditoriaux/titres à ignorer
NOISE_PREFIX_RE = re.compile(
    r"^\s*(?:à\s*lire\s*aussi|a\s*lire\s*aussi|lire\s*aussi|japan\s*data|podcast|édito|edito|tribune|analyse|breaking|le\s*fait\s*du\s*jour|\[vid[eé]o\]|newsletter)\s*:?\s*",
    flags=re.I,
)


@dataclass
class Node:
    id: str
    label: str               # short, UI-facing (will be rewritten by NodeSummarizer)
    summary: str             # clause-level summary (pre-LLM)
    text: str | None = None  # full evidence sentence (rich source for summarization)
    time: str | None = None
    locations: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    title: str | None = None  # tooltip/title for renderers
    role: str = "unknown"     # "cause" | "effect" | "mixed" | "unknown"

@dataclass
class Edge:
    src: str
    dst: str
    relation: str              # "causes" | "correlates" | "contradicts"
    confidence: float          # probabilité locale (avant × confidence_global)
    evidence: list[dict]       # [{doc_id, span, url?, score?}]
    weight: float = 0.0        # confidence_finale (après agrégation)
    color: str = "#ff4d4d"     # rouge/orange/vert selon seuils
    source_text: str | None = None
    sources: list[str] = field(default_factory=list)  # urls
    style: str | None = None    # "dashed" pour liens indirects
    indirect: bool = False
    circular: bool = False


class CausalityMapper:
    """Transforms evidence into a compact causal graph.

    Implement `process` to:
    1) Parse evidence sentences -> events (Node)
    2) Generate candidate edges with simple temporal/pattern rules
    3) Score edges and prune low-confidence (<0.3) for MVP
    4) Return nodes/edges in a normalized structure

    Keep the algorithm deterministic and small to ease testing; you can layer LLM checks later.
    """

    # ----- Patterns FR/EN (précompilés) -----
    # LEFT -> RIGHT  (X leads to Y)
    L2R_PATTERNS = [
        r"\bprovoqu\w+\b", r"\bentra[iî]n\w+\b", r"\bengendr\w+\b",
        r"\bm[èe]ne\s+à\b", r"\bconduit\s+à\b", r"\bcause(?:s|d)?\b",
        r"\blead(?:s|ing)?\s+to\b", r"\bresults?\s+in\b", r"\bbrings\s+about\b",
        r"\bgives\s+rise\s+to\b", r"\binduces?\b", r"\btriggers?\b",
        r"\bexplique\b", r"\bexplains?\b", r"\bcontributes?\s+to\b",
    ]
    # RIGHT -> LEFT  (X because of Y)  => Y causes X
    R2L_PATTERNS = [
        r"\bà\s+cause\s+de\b", r"\ben\s+raison\s+de\b", r"\bdu\s+fait\s+de\b",
        r"\bsuite\s+à\b", r"\best\s+d[ûu]e?\s+à\b", r"\br[ée]sulte\s+de\b",
        r"\bcons[ée]quence\s+de\b", r"\bs[’']explique\s+par\b",
        r"\bbecause\s+of\b", r"\bdue\s+to\b", r"\bowing\s+to\b",
        r"\bcaused\s+by\b", r"\bdriven\s+by\b", r"\bfueled\s+by\b",
        r"\bresult(?:ing)?\s+from\b", r"\bstems\s+from\b", r"\barises\s+from\b",
        r"\battributed\s+to\b", r"\bthanks\s+to\b", r"\bgr[aà]ce\s+à\b",
    ]
    CORRELATE_PATTERNS = [
        r"\best\s+li[ée]?\s+à\b", r"\bcorrelates?\s+with\b", r"\bco[ïi]ncide\s+avec\b",
        r"\bassoci[ée]?\s+à\b", r"\bassociated\s+with\b"
    ]
    CONTRADICT_PATTERNS = [
        r"\bcependant\b", r"\btoutefois\b", r"\bmais\b", r"\bhowever\b",
        r"\bcontradict(?:s|ed|ing)?\b", r"\boppose\b", r"\bnot\s+caused\s+by\b",
        r"\bne\s+.*pas\s+d[ûu]?\s+à\b"
    ]
    HEDGES = re.compile(r"\b(peut|pourrait|possible|probable|probablement|susceptible|might|may|likely|possible)\b", re.I)
    NEGATIONS = re.compile(r"\b(ne\s+\w+\s+pas|not|no|never|aucun[e]?|pas\s+de)\b", re.I)

    def __init__(self):
        self.node_counter = 0  # conservé (non utilisé pour l'id final)

    # ---------- NLP utils ----------
    @staticmethod
    def _lang(text: str) -> str:
        """Heuristique FR/EN simple via stopwords fréquents (évite dépendances lourdes)."""
        t = text.lower()
        fr_hits = len(re.findall(r"\b(le|la|les|des|une?|de|du|et|à|pour|pas)\b", t))
        en_hits = len(re.findall(r"\b(the|a|an|and|to|for|not|of|in|on)\b", t))
        return "fr" if fr_hits >= en_hits else "en"

    @staticmethod
    def _lemma(text: str, lang: str) -> List[str]:
        """Retourne la liste de lemmes filtrés (stopwords + ponctuations)."""
        text = unidecode(text.lower())
        nlp = _NLP_FR if lang == "fr" else _NLP_EN
        if nlp is None:
            # Fallback simple
            toks = re.findall(r"[a-z0-9]+", text)
            stop_fr = {"le", "la", "les", "des", "de", "du", "un", "une", "et", "à", "pour", "dans", "en", "au", "aux", "par", "pas"}
            stop_en = {"the", "a", "an", "and", "to", "for", "in", "on", "of", "by", "with", "not"}
            stops = stop_fr if lang == "fr" else stop_en
            return [t for t in toks if t not in stops]
        doc = nlp(text)
        return [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha]

    def _canon(self, text: str) -> str:
        lang = self._lang(text)
        lemmas = self._lemma(text, lang)
        return " ".join(lemmas).strip() or unidecode(text.lower()).strip()

    def _stable_id(self, text: str) -> str:
        canon = self._canon(text)
        return hashlib.md5(canon.encode("utf-8")).hexdigest()[:12]

    def _make_node(
        self,
        label: str,
        summary: str,
        sources: list[str],
        *,
        text: str | None = None,
        role: str = "unknown",
    ) -> Node:        
        canon = self._canon(summary or label)
        node_id = self._stable_id(canon)
        return Node(
            id=node_id,
            label=label,
            summary=summary,
            text=(text or summary),
            sources=sources,
            title=label,
            role=role,
        )

    def _extract_events(self, evidence: List[dict]) -> List[Node]:
        """Extract rough event candidates from evidence text (fallback)."""
        nodes = []
        for ev in evidence:
            ev_text = (ev.get("text", "") or "").strip()
            if not ev_text:
                continue
            # Normalisation basique : retirer les titres/encarts, compacter espaces
            ev_text = NOISE_PREFIX_RE.sub("", ev_text)
            ev_text = re.sub(r"\s+", " ", ev_text).strip()
            if not ev_text:
                continue

            # Heuristic: split on cause verbs and punctuation
            parts = re.split(r"[.;]|(?:parce que|because)", ev_text)
            for p in parts:
                p = p.strip()
                if len(p.split()) < 3:
                    continue
                # label = short clause, summary = clause, text = full evidence sentence
                node = self._make_node(
                    p[:80],
                    p,
                    [ev.get("doc_id", "unknown")],
                    text=ev.get("text", "") or p,
                    role="unknown",
                )
                nodes.append(node)
        return nodes

    def _clean_phrase(self, s: str) -> str:
        s = s.strip(" .;,:-—–()[]")
        # remove leading discourse markers
        s = re.sub(r"^(?:and|but|however|cependant|toutefois|mais)\s+", "", s, flags=re.I)
        # collapse whitespace
        s = re.sub(r"\s+", " ", s)
        return s

    def _split_once(self, raw: str, pattern: str) -> Tuple[str, str] | None:
        m = re.search(pattern, raw, flags=re.I)
        if not m:
            return None
        left = self._clean_phrase(raw[:m.start()])
        right = self._clean_phrase(raw[m.end():])
        if len(left.split()) < 2 or len(right.split()) < 2:
            return None
        return left, right

    def _split_all(self, raw: str, pattern: str) -> List[Tuple[str, str]]:
        """Retourne TOUTES les paires (gauche, droite) pour un pattern donné dans la phrase."""
        res: List[Tuple[str, str]] = []
        for m in re.finditer(pattern, raw, flags=re.I):
            left = self._clean_phrase(raw[:m.start()])
            right = self._clean_phrase(raw[m.end():])
            if len(left.split()) >= 2 and len(right.split()) >= 2:
                res.append((left, right))
        return res

    def _extract_relations(self, evidence: List[dict]) -> Tuple[List[Node], List[Edge]]:
        """Return (nodes, edges) built from causal/correlate/contradict sentences."""
        nodes_by_key: Dict[str, Node] = {}
        edges_acc: Dict[Tuple[str, str, str], Edge] = {}  # (src_id, dst_id, relation)

        def get_node(clause: str, doc_id: str, full_sentence: str, role: str = "unknown") -> Node:
            key = self._canon(clause)
            if key not in nodes_by_key:
                node = self._make_node(
                    clause[:80],
                    clause,         # summary (clause)
                    [doc_id],
                    text=full_sentence,  # full evidence sentence
                    role=role,
                )
                nodes_by_key[key] = node
            else:
                node = nodes_by_key[key]
                if doc_id not in node.sources:
                    node.sources.append(doc_id)
                # Upgrade du rôle si possible (unknown -> known) ou conflit -> mixed
                if role and role != "unknown":
                    if node.role == "unknown":
                        node.role = role
                    elif node.role != role and node.role != "mixed":
                        node.role = "mixed"
            return node

        for ev in evidence:
            raw_text = (ev.get("text") or "").strip()
            if not raw_text:
                continue
            # Nettoie les en-têtes éditoriaux et compresse les espaces
            raw_text = NOISE_PREFIX_RE.sub("", raw_text)
            raw_text = re.sub(r"\s+", " ", raw_text).strip()
            if len(raw_text.split()) < 5:
                continue
            doc_id = ev.get("doc_id", "unknown")
            url = ev.get("url")
            ev_score = float(ev.get("score", 1.0))
            span = ev.get("span")

            # Flags négation/spéculation
            neg_penalty = 0.0
            if self.NEGATIONS.search(raw_text):
                neg_penalty = 0.25
            hedge_penalty = 0.15 if self.HEDGES.search(raw_text) else 0.0

            # Helper d'ajout d'arête
            def add_edge(cause: str, effect: str, relation: str, base_conf: float):
                cnode = get_node(cause, doc_id, raw_text)
                enode = get_node(effect, doc_id, raw_text)
                key = (cnode.id, enode.id, relation)
                local = max(0.01, base_conf - hedge_penalty)
                if relation == "contradicts":
                    local = min(local, 0.4)
                if key not in edges_acc:
                    edges_acc[key] = Edge(
                        src=cnode.id, dst=enode.id, relation=relation,
                        confidence=local,
                        evidence=[{"doc_id": doc_id, "span": span, "url": url, "score": ev_score}],
                        source_text=raw_text,
                        sources=[u for u in [url] if u],
                    )
                else:
                    e = edges_acc[key]
                    e.evidence.append({"doc_id": doc_id, "span": span, "url": url, "score": ev_score})
                    # prob cumulée  p = 1 - Π(1 - p_i)  (on cumule sur la confidence locale)
                    e.confidence = min(0.99, 1 - (1 - e.confidence) * (1 - local))
                    if url and url not in e.sources:
                        e.sources.append(url)

            # 1) CONTRADICTIONS explicites (prend le pas)
            for pat in self.CONTRADICT_PATTERNS:
                for left, right in self._split_all(raw_text, pat):
                    # Ambigu : on encode comme "contradicts" de left envers right (non-orienté visuellement)
                    add_edge(left, right, "contradicts", base_conf=0.6 - neg_penalty)

            # 2) CORRELATIONS
            for pat in self.CORRELATE_PATTERNS:
                for left, right in self._split_all(raw_text, pat):
                    add_edge(left, right, "correlates", base_conf=0.55 - hedge_penalty)

            # 3) CAUSALITÉ L->R
            for pat in self.L2R_PATTERNS:
                for left, right in self._split_all(raw_text, pat):
                    # left leads to right
                    base = 0.6 if re.search(r"(explique|explains|contributes?\s+to)", pat, re.I) else 0.7
                    # marquer rôle: left=cause, right=effet
                    cnode = get_node(left, doc_id, raw_text, role="cause")
                    enode = get_node(right, doc_id, raw_text, role="effect")
                    key = (cnode.id, enode.id, "causes")
                    local_base = base - neg_penalty
                    if key not in edges_acc:
                        edges_acc[key] = Edge(
                            src=cnode.id, dst=enode.id, relation="causes",
                            confidence=local_base,
                            evidence=[{"doc_id": doc_id, "span": span, "url": url, "score": ev_score}],
                            source_text=raw_text,
                            sources=[u for u in [url] if u],
                        )
                    else:
                        e = edges_acc[key]
                        e.evidence.append({"doc_id": doc_id, "span": span, "url": url, "score": ev_score})
                        e.confidence = min(0.99, 1 - (1 - e.confidence) * (1 - local_base))
                        if url and url not in e.sources:
                            e.sources.append(url)

            # 4) CAUSALITÉ R->L
            for pat in self.R2L_PATTERNS:
                for left, right in self._split_all(raw_text, pat):
                    # right caused by left  =>  left causes right
                    # "thanks to / grâce à" = cause facilitatrice -> base un peu plus faible
                    facilitator = bool(re.search(r"(thanks\s+to|gr[aà]ce\s+à)", pat, re.I))
                    base = 0.75 if not facilitator else 0.55
                    # négation explicite du type "not caused by" -> contradictoire
                    if re.search(r"(not\s+caused\s+by|ne\s+.*pas\s+d[ûu]?\s+à)", raw_text, re.I):
                        add_edge(right, left, "contradicts", base_conf=0.6)
                    else:
                        # marquer rôle: left=cause, right=effet  (car left causes right)
                        cnode = get_node(left, doc_id, raw_text, role="cause")
                        enode = get_node(right, doc_id, raw_text, role="effect")
                        key = (cnode.id, enode.id, "causes")
                        local_base = base - neg_penalty
                        if key not in edges_acc:
                            edges_acc[key] = Edge(
                                src=cnode.id, dst=enode.id, relation="causes",
                                confidence=local_base,
                                evidence=[{"doc_id": doc_id, "span": span, "url": url, "score": ev_score}],
                                source_text=raw_text,
                                sources=[u for u in [url] if u],
                            )
                        else:
                            e = edges_acc[key]
                            e.evidence.append({"doc_id": doc_id, "span": span, "url": url, "score": ev_score})
                            e.confidence = min(0.99, 1 - (1 - e.confidence) * (1 - local_base))
                            if url and url not in e.sources:
                                e.sources.append(url)

        # Sort stable
        nodes = list(nodes_by_key.values())
        edges = list(edges_acc.values())
        edges.sort(key=lambda e: (e.src, e.dst, e.relation))
        return nodes, edges

    def _deduplicate_nodes(self, nodes: List[Node]) -> tuple[List[Node], dict[str, str]]:
        """Merge near-duplicate nodes using simple label normalization.
        Returns (deduped_nodes, id_map old->kept)."""
        seen: dict[str, Node] = {}
        id_map: dict[str, str] = {}
        result: List[Node] = []
        for n in nodes:
            base = self._canon(n.summary or n.label or "")
            key = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", base))
            if key in seen:
                kept = seen[key]
                kept.sources = list(set(kept.sources + n.sources))
                # conserve le texte le plus riche comme source
                try:
                    if (n.text and (not kept.text or len(n.text) > len(kept.text or ""))):
                        kept.text = n.text
                except Exception:
                    pass
                # fusion du rôle
                if kept.role == "unknown" and n.role != "unknown":
                    kept.role = n.role
                elif kept.role != n.role and n.role != "unknown":
                    kept.role = "mixed"
                id_map[n.id] = kept.id
            else:
                seen[key] = n
                # Réécrit l'ID avec la version stable canonique
                stable = self._stable_id(base)
                n.id = stable
                result.append(n)
                id_map[n.id] = n.id
        return result, id_map

    def _deduplicate_edges(self, edges: List[Edge]) -> List[Edge]:
        """Fusionne les arêtes (src,dst,relation) en agrégeant l'évidence et la confiance (p cumulée)."""
        grouped: Dict[Tuple[str, str, str], Edge] = {}
        for e in edges:
            key = (e.src, e.dst, e.relation)
            if key not in grouped:
                grouped[key] = e
            else:
                g = grouped[key]
                g.evidence.extend(e.evidence)
                g.sources = list(set((g.sources or []) + (e.sources or [])))
                g.source_text = g.source_text or e.source_text
                # p cumulée
                g.confidence = min(0.99, 1 - (1 - g.confidence) * (1 - e.confidence))
        return list(grouped.values())

    @staticmethod
    def _extract_year(text: str) -> int | None:
        """Heuristique: retourne la première année plausible (1900-2100)."""
        m = re.search(r"(19|20)\d{2}", text)
        if m:
            y = int(m.group(0))
            if 1900 <= y <= 2100:
                return y
        return None

    def _temporal_validate(self, nodes: List[Node], edges: List[Edge]) -> None:
        """Règle simple: si année(cause) > année(effet) ⇒ pénalité 50% sur la confidence locale."""
        years: Dict[str, int] = {}
        for n in nodes:
            y = self._extract_year(n.text or n.summary or n.label or "")
            if y:
                years[n.id] = y
                n.time = str(y)
        for e in edges:
            yc = years.get(e.src)
            ye = years.get(e.dst)
            if yc and ye and yc > ye:
                e.confidence = max(0.01, e.confidence * 0.5)

    @staticmethod
    def _color_for(weight: float) -> str:
        if weight >= 0.7:
            return "#4caf50"  # vert
        if weight >= 0.45:
            return "#ffa500"  # orange
        return "#ff4d4d"      # rouge

    def _chain_indirect_edges(self, edges: List[Edge]) -> List[Edge]:
        """Si A→B et B→C (causes), créer A→C indirect (dashed), poids = produit."""
        direct = [(e.src, e.dst, e) for e in edges if e.relation == "causes"]
        by_src = defaultdict(list)
        by_dst = defaultdict(list)
        for s, d, e in direct:
            by_src[s].append((d, e))
            by_dst[d].append((s, e))
        add: List[Edge] = []
        for b in set(d for _, d, _ in direct):
            preds = by_dst.get(b, [])
            succs = by_src.get(b, [])
            for (a, ea), (c, ec) in itertools.product(preds, succs):
                if a == c:
                    continue
                # éviter doublons si déjà présent
                if any(x.src == a and x.dst == c and x.relation == "causes" for x in edges):
                    continue
                w = max(0.01, ea.weight * ec.weight)
                e = Edge(
                    src=a, dst=c, relation="causes",
                    confidence=min(0.95, ea.confidence * ec.confidence),
                    evidence=[{"chain": [(ea.src, ea.dst), (ec.src, ec.dst)]}],
                    weight=w, style="dashed", indirect=True, source_text=None
                )
                add.append(e)
        return edges + add

    def _mark_cycles(self, edges: List[Edge]) -> None:
        if nx is None:
            return
        G = nx.DiGraph()
        for e in edges:
            G.add_edge(e.src, e.dst)
        try:
            for cycle in nx.simple_cycles(G):
                # marquer toutes les arêtes appartenant aux cycles détectés
                cycle_set = set()
                for u, v in zip(cycle, cycle[1:] + cycle[:1]):
                    cycle_set.add((u, v))
                for e in edges:
                    if (e.src, e.dst) in cycle_set:
                        e.circular = True
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    # Post-summarization: factoriser par label final et agréger les arêtes
    # ------------------------------------------------------------------
    def cleanup_after_summarization(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Regroupe les nœuds par leur label (après passage du NodeSummarizer),
        reconstruit des IDs stables, remappe les arêtes, agrège les doublons,
        et recalcule la couleur depuis le poids/la confiance.
        """
        if not nodes:
            return nodes, edges

        def canon_label(lbl: str) -> str:
            return re.sub(r"\s+", " ", self._canon(lbl or "").strip())

        # 1) Fusion des nœuds par label
        merged: Dict[str, Dict[str, Any]] = {}
        id_map: Dict[str, str] = {}
        for n in nodes:
            old_id = n.get("id")
            label = n.get("label") or n.get("summary") or n.get("title") or n.get("text") or old_id
            c = canon_label(label)
            new_id = hashlib.md5(c.encode("utf-8")).hexdigest()[:12]
            if c not in merged:
                m = dict(n)
                m["id"] = new_id
                m["label"] = label
                m["title"] = label
                m["sources"] = list({*(n.get("sources") or [])})
                merged[c] = m
            else:
                m = merged[c]
                m["sources"] = list({*(m.get("sources") or []), *(n.get("sources") or [])})
                # rôle : unknown -> known ; conflit -> mixed
                r0 = (m.get("role") or "unknown").lower()
                r1 = (n.get("role") or "unknown").lower()
                if r0 == "unknown" and r1 != "unknown":
                    m["role"] = r1
                elif r1 != "unknown" and r1 != r0 and r0 != "mixed":
                    m["role"] = "mixed"
                # label : conserve le plus concis
                lab0 = m.get("label") or ""
                if lab0 and label and len(label) < len(lab0):
                    m["label"] = label
                    m["title"] = label
                # full_text le plus informatif
                ft0 = (m.get("full_text") or m.get("text") or "")
                ft1 = (n.get("full_text") or n.get("text") or "")
                if len(ft1) > len(ft0):
                    if "full_text" in n:
                        m["full_text"] = n["full_text"]
                    elif "text" in n:
                        m["full_text"] = n["text"]
            if old_id:
                id_map[old_id] = new_id

        new_nodes = list(merged.values())

        # 2) Remap & agrégation des arêtes (src,dst,relation)
        def _get_src(e: Dict[str, Any]) -> str | None:
            return e.get("from") or e.get("src")
        def _get_dst(e: Dict[str, Any]) -> str | None:
            return e.get("to") or e.get("dst")
        def _set_src(e: Dict[str, Any], v: str) -> None:
            if "from" in e:
                e["from"] = v
            else:
                e["src"] = v
        def _set_dst(e: Dict[str, Any], v: str) -> None:
            if "to" in e:
                e["to"] = v
            else:
                e["dst"] = v

        grouped: Dict[tuple, Dict[str, Any]] = {}
        for e in edges or []:
            s = _get_src(e)
            d = _get_dst(e)
            if not s or not d:
                continue
            if s in id_map:
                s = id_map[s]
            if d in id_map:
                d = id_map[d]
            _set_src(e, s)
            _set_dst(e, d)
            rel = e.get("relation_type") or e.get("relation") or "rel"
            key = (s, d, rel)
            if key not in grouped:
                g = dict(e)
                g["relation_type"] = rel
                grouped[key] = g
            else:
                g = grouped[key]
                # prob cumulée
                p0 = float(g.get("confidence", 0.0) or 0.0)
                p1 = float(e.get("confidence", 0.0) or 0.0)
                g["confidence"] = min(0.99, 1 - (1 - p0) * (1 - p1))
                # weight = max
                g["weight"] = max(float(g.get("weight", 0.0) or 0.0), float(e.get("weight", 0.0) or 0.0))
                # merge sources/evidence
                g["sources"] = list({*(g.get("sources") or []), *(e.get("sources") or [])})
                g["evidence"] = (g.get("evidence") or []) + (e.get("evidence") or [])
                # flags
                g["indirect"] = bool(g.get("indirect")) or bool(e.get("indirect"))
                g["circular"] = bool(g.get("circular")) or bool(e.get("circular"))

        for g in grouped.values():
            w = float(g.get("weight", 0.0) or 0.0) or float(g.get("confidence", 0.0) or 0.0)
            g["color"] = self._color_for(w)

        new_edges = list(grouped.values())
        return new_nodes, new_edges

    @timed("causality_mapper.process")
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build a causal graph from `input_data["evidence"]`.

        Expected input structure (from FactFinder):
        {
          "query": str,
          "confidence_global": float,
          "evidence": [
            {"doc_id": str, "text": str, "span": [start,end]|None, "url": str?, "score": float?},
            ...
          ]
        }
        """
        evidence = input_data.get("evidence", [])
        if not evidence:
            return {"nodes": [], "edges": [], "es_records": []}
        confidence_global = float(input_data.get("confidence_global", 0.8))
        query_ctx = input_data.get("query", "")

        # 1) Try to build nodes/edges from explicit causal patterns
        nodes, edges = self._extract_relations(evidence)

        # 2) Fallback: if no edges extracted, show rough events (better than empty graph)
        if not edges:
            nodes = self._extract_events(evidence)

        # Deduplicate nodes and remap edges
        nodes, id_map = self._deduplicate_nodes(nodes)
        for e in edges:
            if e.src in id_map:
                e.src = id_map[e.src]
            if e.dst in id_map:
                e.dst = id_map[e.dst]
        # Edge dedup après remap
        edges = self._deduplicate_edges(edges)

        # Validation temporelle simple (pénalités si incohérence)
        self._temporal_validate(nodes, edges)

        # ---- Pondération finale (weight) et champs visuels ----
        # Combine les scores d'ev + confidence_global
        for e in edges:
            ev_scores = [float(x.get("score", 1.0)) for x in e.evidence] or [1.0]
            mean_ev = sum(ev_scores) / max(1, len(ev_scores))
            e.weight = max(0.0, min(1.0, e.confidence * mean_ev * confidence_global))
            e.color = self._color_for(e.weight)
            if not e.sources:
                e.sources = list({x.get("url") for x in e.evidence if x.get("url")})
                e.sources = [s for s in e.sources if s]
            # fallback tooltip
            if not e.source_text and e.evidence:
                e.source_text = (e.evidence[0].get("text") or "") if "text" in e.evidence[0] else None

        # Prune low-confidence edges and discard edges with unknown endpoints
        edges = [e for e in edges if e.confidence >= 0.3 and e.src and e.dst]
        node_ids = {n.id for n in nodes}
        edges = [e for e in edges if (e.src in node_ids and e.dst in node_ids)]

        # Chaînage A→C (dashed / indirect)
        edges = self._chain_indirect_edges(edges)
        # Uniformise la couleur sur la base du poids (y compris liens indirects)
        for e in edges:
            e.color = self._color_for(e.weight if e.weight else e.confidence * confidence_global)

        # Détection de cycles
        self._mark_cycles(edges)

        logger.info(
            "CausalityMapper produced graph",
            extra={"nodes": len(nodes), "edges": len(edges)},
        )

        # ----- Documents pour Elasticsearch (optionnel en sortie) -----
        es_records = []
        timestamp = datetime.utcnow().isoformat() + "Z"
        for e in edges:
            rec = {
                "cause": next((n.summary for n in nodes if n.id == e.src), e.src),
                "effet": next((n.summary for n in nodes if n.id == e.dst), e.dst),
                "relation_type": e.relation,
                "embedding": None,  # à remplir par pipeline d'embedding (384d) si dispo
                "confidence": e.weight,
                "query_context": query_ctx,
                "timestamp": timestamp,
                "sources": e.sources,
            }
            es_records.append(rec)

        # Format prêt PyVis (from/to/weight/color/style)
        pyvis_edges = []
        for e in edges:
            pyvis_edges.append({
                "from": e.src, "to": e.dst, "relation_type": e.relation,
                "weight": e.weight, "color": e.color, "style": e.style,
                "source_text": e.source_text, "sources": e.sources,
                "indirect": e.indirect, "circular": e.circular,
                "confidence": e.confidence,
                "evidence": e.evidence,
            })

        return {
            "nodes": [n.__dict__ for n in nodes],
            "edges": pyvis_edges,
            "es_records": es_records,
        }