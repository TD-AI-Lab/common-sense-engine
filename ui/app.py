"""
Streamlit UI (Prototype)
------------------------
Layout:
- Left: chat-like input and summarized answer
- Right: interactive causal graph (pyvis)
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components
from math import ceil

# --------------------------------------------------------------------
# Tokenization and filtering helpers
# --------------------------------------------------------------------
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+")

STOPWORDS_EN = {
    "why", "how", "what", "which", "who", "whom", "whose", "the", "a", "an", "in", "on", "at",
    "to", "for", "of", "by", "and", "or", "is", "are", "was", "were", "be", "been", "being",
    "from", "with", "as", "it", "this", "that", "these", "those"
}

def tokenize(text: str) -> set[str]:
    return {w.lower() for w in WORD_RE.findall(text or "")}

def query_tokens(query: str) -> set[str]:
    q = tokenize(query)
    return {w for w in q if len(w) >= 3 and w not in STOPWORDS_EN}

GENERIC_NOISE = {
    "price", "increase", "rise", "cost", "inflation", "market", "world", "year", "month", "day",
    "cause", "reason", "effect", "level", "strong", "weak", "high", "low", "record", "index", "rate"
}

def significant_tokens(text: str) -> set[str]:
    toks = tokenize(text)
    return {t for t in toks if len(t) >= 3 and t not in STOPWORDS_EN and t not in GENERIC_NOISE and not t.isdigit()}

def dominant_topic_tokens(evidence_list: List[Dict[str, Any]], min_df_ratio: float = 0.3, max_tokens: int = 30) -> set[str]:
    """
    Determine a dominant topic from evidence texts:
    tokens whose document frequency >= min_df_ratio.
    Fallback: top-N by DF if no tokens exceed the threshold.
    """
    df: Dict[str, int] = {}
    total = 0
    for ev in (evidence_list or []):
        total += 1
        toks = significant_tokens(ev.get("text", ""))
        for t in toks:
            df[t] = df.get(t, 0) + 1
    if total == 0 or not df:
        return set()
    min_df = max(1, ceil(total * min_df_ratio))
    dom = {t for t, c in df.items() if c >= min_df}
    if not dom:
        dom = {t for t, _ in sorted(df.items(), key=lambda kv: kv[1], reverse=True)[:max_tokens]}
    return dom

# --------------------------------------------------------------------
# Streamlit Config
# --------------------------------------------------------------------
st.set_page_config(page_title="🧠 Common Sense Engine", layout="wide")

# --------------------------------------------------------------------
# API Config
# --------------------------------------------------------------------
try:
    API_URL = st.secrets["API_URL"]
except Exception:
    API_URL = "http://localhost:8000"

# --------------------------------------------------------------------
# Header
# --------------------------------------------------------------------
st.title("🧠 Common Sense Engine")
st.caption("Ask a complex question and let the AI connect causes and effects.")

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _build_graph_html(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> str:
    """Create a PyVis network graph and return rendered HTML."""
    net = Network(height="650px", width="100%", directed=True, notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120, spring_strength=0.02, damping=0.09)

    seen = set()
    for n in nodes:
        nid = n.get("id")
        if not nid or nid in seen:
            continue
        seen.add(nid)
        label = n.get("label") or n.get("summary") or n.get("title") or n.get("text") or nid
        label_disp = (label[:60] + "…") if len(label) > 60 else label
        time = n.get("time") or "—"
        sources = n.get("sources") or []
        long_txt = n.get("full_text") or n.get("title") or n.get("text") or label
        title = f"<b>{label}</b><br/>{long_txt}<br/>time: {time}<br/>sources: {', '.join(sources[:4])}"
        net.add_node(
            nid,
            label=label_disp,
            title=title,
            shape="dot",
            size=14,
            borderWidth=1,
        )

    def _fallback_color_for(rel: str) -> str:
        r = (rel or "").lower()
        if r == "causes":
            return "#2E7D32"
        if r == "contradicts":
            return "#C62828"
        if r == "correlates":
            return "#F57C00"
        return "#607D8B"

    for e in edges:
        src = e.get("from") or e.get("src")
        dst = e.get("to") or e.get("dst")
        if not src or not dst:
            continue
        rel = e.get("relation_type") or e.get("relation") or "rel"
        weight = float(e.get("weight", 0.0) or 0.0)
        conf = float(e.get("confidence", weight) or 0.0)
        w_for_width = weight if weight > 0 else conf
        width = max(1.0, (w_for_width ** 0.5) * 4.0)
        color = e.get("color") or _fallback_color_for(rel)
        tooltip = f"{rel} — {e.get('source_text','')}"
        srcs = [s for s in (e.get("sources") or []) if isinstance(s, str)]
        if srcs:
            safe = "</br>".join([f'<a href="{s}" target="_blank">{s}</a>' for s in srcs[:3]])
            tooltip += f"<br/>sources:<br/>{safe}"
        dashes = (e.get("style") == "dashed") or bool(e.get("indirect"))
        net.add_edge(
            src, dst,
            value=w_for_width,
            width=width,
            color=color,
            title=tooltip,
            arrows="to",
            smooth=True,
            dashes=dashes,
        )

    return net.generate_html()

def _call_api(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.post(f"{API_URL}/explain", json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling the API: {e}")
        return None
    except ValueError:
        st.error("Error: the API response is not valid JSON.")
        return None

# --------------------------------------------------------------------
# Input section
# --------------------------------------------------------------------
with st.form("query_form", clear_on_submit=False):
    query = st.text_input(
        "💬 Ask a question (e.g. How did the Russian gas supply cuts affect electricity costs in Germany?)",
        placeholder="Type your question here...",
        key="query_text",
    )
    c1, c2 = st.columns([2, 1])
    k = c2.number_input("Top documents (k)", min_value=1, max_value=100, value=20, step=1, key="k_value")
    submitted = st.form_submit_button("🔍 Explain", use_container_width=True)

if submitted:
    if not query.strip():
        st.warning("Please enter a question before starting the analysis.")
    else:
        payload = {
            "query": query.strip(),
            "time_range": None,  # removed start/end date
            "k": int(k),
            "live": True,
        }
        with st.spinner("⏳ Analyzing..."):
            data = _call_api(payload)

        if data:
            if "errors" in data and data["errors"]:
                with st.expander("⚠️ Pipeline Errors", expanded=False):
                    for err in data["errors"]:
                        st.error(f"- {err}")

            left, right = st.columns([1, 1])

            with left:
                st.subheader("🧾 Summary")
                summary = data.get("summary", "")
                if summary:
                    st.write(summary)
                else:
                    st.info("No summary generated.")

                st.metric("Global confidence", f"{data.get('confidence_global', 0.0):.2f}")

                meta_cols = st.columns(2)
                meta_cols[0].write(f"**Synthesis mode** : `{data.get('mode', 'template')}`")
                meta_cols[1].write(f"**Retrieval mode** : `{data.get('retrieval_mode', 'lexical_only')}`")

                sugg = data.get("followup_suggestions") or []
                if sugg:
                    st.subheader("💡 Suggestions")
                    for s in sugg:
                        if st.button(s, key=f"sugg_{s}"):
                            st.session_state["query_text"] = s

                evidence_all = data.get("evidence", []) or []
                dom_tokens = dominant_topic_tokens(evidence_all)

                if dom_tokens:
                    evidence = [ev for ev in evidence_all if significant_tokens(ev.get("text", "")).intersection(dom_tokens)]
                else:
                    evidence = evidence_all

                evidence_top = evidence[:15]
                facts = data.get("facts", []) or []

                used_urls = {
                    (ev.get("source_url") or ev.get("url"))
                    for ev in evidence_top
                    if ev.get("source_url") or ev.get("url")
                }
                used_doc_ids = {ev.get("doc_id") for ev in evidence_top if ev.get("doc_id")}

                filtered_facts = []
                seen_urls = set()
                for f in facts:
                    url = f.get("source_url") or f.get("url")
                    fid = f.get("id")
                    if url and url in used_urls and url not in seen_urls:
                        ftoks = significant_tokens((f.get("title") or "") + " " + (f.get("source_url") or ""))
                        if not dom_tokens or ftoks.intersection(dom_tokens):
                            seen_urls.add(url)
                            filtered_facts.append(f)
                    elif fid and fid in used_doc_ids:
                        ftoks = significant_tokens((f.get("title") or "") + " " + (f.get("source_url") or ""))
                        if not dom_tokens or ftoks.intersection(dom_tokens):
                            url_key = url or fid
                            if url_key not in seen_urls:
                                seen_urls.add(url_key)
                            seen_urls.add(url)
                            filtered_facts.append(f)

                if not filtered_facts and evidence_top:
                    for ev in evidence_top:
                        url = ev.get("source_url") or ev.get("url") or ""
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            filtered_facts.append({"title": url.split("/")[-1] or url, "source_url": url})

                facts = filtered_facts

                if facts:
                    st.subheader("📚 Sources")
                    for f in facts[:10]:
                        title = f.get("title", "Document")
                        url = f.get("source_url") or ""
                        if url:
                            st.markdown(f"- **{title}** — {url}")
                        else:
                            st.markdown(f"- **{title}**")

                if evidence:
                    with st.expander("🔎 Top Evidence Excerpts", expanded=False):
                        for ev in evidence[:15]:
                            txt = ev.get("text", "")
                            date = ev.get("date") or "—"
                            url = ev.get("source_url") or ""
                            st.markdown(f"- _{txt}_  \n  — date: {date} | source: {url}")

                st.download_button(
                    "⬇️ Download JSON response",
                    data=json.dumps(data, ensure_ascii=False, indent=2),
                    file_name="explain_response.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with right:
                st.subheader("🌐 Causal Graph (MVP)")
                st.caption("Interactive visualization of causal relations (pyvis).")

                nodes_raw = data.get("nodes", []) or []
                edges_raw = data.get("edges", []) or []

                allowed_urls = {
                    (ev.get("source_url") or ev.get("url"))
                    for ev in evidence_top
                    if ev.get("source_url") or ev.get("url")
                }
                allowed_doc_ids = {ev.get("doc_id") for ev in evidence_top if ev.get("doc_id")}

                def node_has_allowed_source(n: Dict[str, Any]) -> bool:
                    srcs = n.get("sources") or []
                    for s in srcs:
                        if isinstance(s, str):
                            if s in allowed_doc_ids:
                                return True
                            if s.startswith("http") and s in allowed_urls:
                                return True
                    return False

                nodes = [n for n in nodes_raw if node_has_allowed_source(n)]
                valid_ids = {n.get("id") for n in nodes}

                def _endpoints_ok(e: Dict[str, Any]) -> bool:
                    s = e.get("from") or e.get("src")
                    d = e.get("to") or e.get("dst")
                    return (s in valid_ids) and (d in valid_ids)

                edges = [e for e in edges_raw if _endpoints_ok(e)]

                c1, c2, _ = st.columns([1, 1, 1])
                show_indirect = c1.toggle("Show indirect links (dashed)", value=True)
                show_noncausal = c2.toggle("Include correlations/contradictions", value=True)
                if not show_indirect:
                    edges = [e for e in edges if not (e.get("indirect") or e.get("style") == "dashed")]
                if not show_noncausal:
                    def _is_causal(e: Dict[str, Any]) -> bool:
                        r = (e.get("relation_type") or e.get("relation") or "").lower()
                        return r == "causes"
                    edges = [e for e in edges if _is_causal(e)]

                if nodes or edges:
                    html = _build_graph_html(nodes, edges)
                    components.html(html, height=680, scrolling=True)
                else:
                    st.info("No nodes or edges to display for this query.")
