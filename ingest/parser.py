"""
Parser module: convert FetchedPage → ParsedDocument.

P0 scope:
- HTML/RSS only. JSON API payloads are skipped for now.
- Use Trafilatura if present, else a naive cleaner (regex) as fallback.
- Extract common metadata (title, author, published_at, lang, canonical_url, publisher).
- Normalize text consistently — Evidence spans refer to this normalized text.

Design notes:
- Keep parsing fast (timeouts already enforced by fetcher).
- Discard ultra-short pages (<200 chars after normalization).
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import re

from .types import FetchedPage, ParsedDocument


def parse_page(request_id: str, page: FetchedPage) -> Optional[ParsedDocument]:
    """
    Return a ParsedDocument or None if the page is unusable.
    Skips non-200/no content; focuses on HTML/RSS.
    """
    if page.status != 200 or not page.raw:
        return None

    if page.content_type not in ("html", "rss", "xml"):
        # P0: skip JSON/unknown
        return None

    html = page.raw.decode(errors="replace")

    # Try Trafilatura first
    text, meta1 = _extract_with_trafilatura(html)
    if not text:
        # fallback naive
        text = _naive_text_clean(html)
    meta2 = _extract_meta_from_html(html)

    # Merge meta (prefer explicit HTML meta over Trafilatura when present)
    meta = {**meta1, **{k: v for k, v in meta2.items() if v}}

    title = meta.get("title")
    lang = meta.get("lang")
    published_at = _normalize_date(meta.get("published_at"))
    canonical = meta.get("canonical_url")
    publisher = meta.get("publisher")
    author_list: List[str] = []
    if meta.get("author"):
        author_list = [meta["author"]] if isinstance(meta["author"], str) else list(meta["author"])

    text = _normalize_text(text)
    if len(text) < 200:
        return None

    return ParsedDocument(
        request_id=request_id,
        url=page.final_url or page.url,
        canonical_url=canonical,
        fetched_at=page.fetched_at,
        status=page.status,
        publisher=publisher,
        author=author_list,
        title=title,
        lang=lang,
        country=None,
        published_at=published_at,
        section=None,
        text=text,
        char_map_ref=None,
        raw_ref=None,
        parsed_ref=None,
        meta=meta,
    )


# -----------------------------
# Helpers
# -----------------------------

def _extract_with_trafilatura(html: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract main text + meta with Trafilatura if available.
    Returns (text, meta). On failure, returns ("", {}).
    """
    try:
        import trafilatura  # type: ignore
        from trafilatura.settings import use_config  # type: ignore

        cfg = use_config()
        cfg.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")  # avoid long blocking
        txt = trafilatura.extract(html, include_comments=False, output_format="txt", include_links=False, config=cfg) or ""

        md = trafilatura.extract_metadata(html) or None
        meta = {}
        if md:
            meta["title"] = getattr(md, "title", None)
            meta["author"] = getattr(md, "author", None)
            meta["published_at"] = getattr(md, "date", None)
            meta["lang"] = getattr(md, "language", None)
            # canonical may be unreliable; let HTML meta override
            meta["canonical_url"] = None
            meta["publisher"] = getattr(md, "site", None) or getattr(md, "sitename", None)

        return txt, meta
    except Exception:
        return "", {}


def _extract_meta_from_html(html: str) -> Dict[str, Any]:
    """
    Quick meta extraction using regex:
    - <title>, <meta property="og:*">, <meta name="author">,
    - article:published_time, og:url (as canonical), <html lang="..">, og:site_name
    """
    def _find1(pat: str) -> Optional[str]:
        m = re.search(pat, html, flags=re.I | re.S)
        return m.group(1).strip() if m else None

    title = _find1(r"<title[^>]*>(.*?)</title>")
    author = _find1(r'<meta[^>]+name=["\']author["\'][^>]+content=["\'](.*?)["\']')
    published = _find1(r'<meta[^>]+(?:name|property)=["\'](?:article:published_time|pubdate|datePublished)["\'][^>]+content=["\'](.*?)["\']')
    canonical = _find1(r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\'](.*?)["\']') or \
               _find1(r'<meta[^>]+property=["\']og:url["\'][^>]+content=["\'](.*?)["\']')
    lang = _find1(r'<html[^>]+lang=["\'](.*?)["\']')
    publisher = _find1(r'<meta[^>]+property=["\']og:site_name["\'][^>]+content=["\'](.*?)["\']')

    return {
        "title": title,
        "author": author,
        "published_at": published,
        "canonical_url": canonical,
        "lang": lang,
        "publisher": publisher,
    }


def _naive_text_clean(html: str) -> str:
    """Remove scripts/styles, drop tags, and collapse whitespace."""
    html = re.sub(r"<!--.*?-->", " ", html, flags=re.S)
    html = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<style.*?</style>", " ", html, flags=re.S | re.I)
    txt = re.sub(r"<[^>]+>", " ", html)
    return _normalize_text(txt)


def _normalize_text(text: str) -> str:
    """Collapse whitespace, normalize quotes, keep readable Unicode, preserve newlines sparingly."""
    text = text.replace("\xa0", " ").replace("\r", " ")
    # keep paragraph boundaries
    text = re.sub(r"\n{3,}", "\n\n", text)
    # collapse spaces
    text = re.sub(r"[ \t]+", " ", text)
    # normalize spaces around newlines
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return text.strip()


def _normalize_date(val: Optional[str]) -> Optional[str]:
    """Convert various date strings to ISO YYYY-MM-DD where possible."""
    if not val:
        return None
    s = val.strip()
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", s)
    if m:
        return m.group(1)
    for fmt in ("%Y/%m/%d", "%d/%m/%Y", "%Y.%m.%d", "%Y-%m", "%Y"):
        try:
            from datetime import datetime
            if fmt == "%Y-%m":
                return datetime.strptime(s, "%Y-%m").strftime("%Y-%m-01")
            if fmt == "%Y":
                return datetime.strptime(s, "%Y").strftime("%Y-01-01")
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    # Try to capture a date-like substring (very permissive)
    m2 = re.search(r"(\d{4})[-/\.](\d{1,2})[-/\.](\d{1,2})", s)
    if m2:
        y, mo, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"
    return None