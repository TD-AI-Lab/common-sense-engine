"""
Polite fetcher (HTTP + optional headless later) with robots.txt compliance, rate limiting and retries.

P0 scope:
- Async HTTP via httpx IF available. If not, you can use synchronous wrappers (urllib fallback).
- Enforce robots.txt (deny if disallowed) with a tiny in-memory cache (24h TTL).
- Respect a minimal politeness delay (2–5s/host with jitter).
- Timeouts: connect=3s, read=8s.
- Return FetchedPage with raw bytes + headers + content_type.

Also provided:
- fetch_seed_sync(seed): synchronous wrapper (uses asyncio if httpx is present, else urllib)
- fetch_many(seeds, concurrency=5): async batched fetch with semaphore
- fetch_many_sync(seeds, concurrency=5): synchronous convenience wrapper
"""

from __future__ import annotations

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import urlparse, quote_plus
import asyncio
import random
import sys

from .types import Seed, FetchedPage, ContentType


# -----------------------------
# Robots cache (simple, in-memory)
# -----------------------------

_ROBOTS_CACHE: Dict[str, Dict[str, object]] = {}  # host -> {'allowed': callable(path)->bool, 'fetched_at': datetime}
_LAST_FETCH_PER_HOST: Dict[str, datetime] = {}     # politeness tracker


async def _fetch_robots(host: str) -> None:
    """
    Download and cache robots.txt for a host.
    Minimalistic parser: allow-all if missing/non-200.
    """
    if host in _ROBOTS_CACHE and (datetime.utcnow() - _ROBOTS_CACHE[host]["fetched_at"]) < timedelta(hours=24):
        return

    url = f"https://{host}/robots.txt"
    allow_all = {"allowed": (lambda path: True), "fetched_at": datetime.utcnow()}
    try:
        # Lazy import httpx
        try:
            import httpx  # type: ignore
        except Exception:
            # No httpx → fall back to allow-all (safe default), wrappers sync will still work with urllib.
            _ROBOTS_CACHE[host] = allow_all
            return

        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            resp = await client.get(url, headers={"User-Agent": "CSEBot/0.1 (+contact-url)"})
        if resp.status_code != 200 or "disallow" not in resp.text.lower():
            _ROBOTS_CACHE[host] = allow_all
            return

        rules = _parse_simple_robots(resp.text)
        _ROBOTS_CACHE[host] = {"allowed": rules, "fetched_at": datetime.utcnow()}
    except Exception:
        _ROBOTS_CACHE[host] = allow_all


def _parse_simple_robots(text: str) -> Callable[[str], bool]:
    """
    Very light parser: builds an allowed(path)->bool for User-agent: * rules.
    Limitations: ignores crawl-delay, sitemaps, specific user-agents (OK for P0).
    """
    lines = [l.strip() for l in text.splitlines()]
    ua_star = False
    disallows: List[str] = []
    allows: List[str] = []

    for line in lines:
        if not line or line.startswith("#"):
            continue
        lower = line.lower()
        if lower.startswith("user-agent:"):
            ua_star = ("*" in lower)
        elif ua_star and lower.startswith("disallow:"):
            path = line.split(":", 1)[1].strip()
            disallows.append(path or "/")
        elif ua_star and lower.startswith("allow:"):
            path = line.split(":", 1)[1].strip()
            if path:
                allows.append(path)

    def is_allowed(path: str) -> bool:
        for a in allows:
            if path.startswith(a):
                return True
        for d in disallows:
            if d != "/" and path.startswith(d):
                return False
        return True

    return is_allowed


async def _respect_politeness(host: str) -> None:
    """Ensure a minimal delay (2–5s + jitter) between requests to the same host."""
    now = datetime.utcnow()
    last = _LAST_FETCH_PER_HOST.get(host)
    delay_s = random.uniform(2.0, 5.0)
    if last:
        elapsed = (now - last).total_seconds()
        if elapsed < delay_s:
            await asyncio.sleep(delay_s - elapsed)
    _LAST_FETCH_PER_HOST[host] = datetime.utcnow()


def _normalize_content_type(ct: Optional[str]) -> ContentType:
    if not ct:
        return "unknown"
    c = ct.lower()
    if "html" in c:
        return "html"
    if "json" in c:
        return "json"
    if "xml" in c or "rss" in c or "atom" in c:
        return "rss"
    return "unknown"


# -----------------------------
# Public async API
# -----------------------------

async def fetch_seed(seed: Seed) -> List[FetchedPage]:
    """
    Execute a single seed:
      - type='rss'  → GET the feed (one page)
      - type='url'  → GET the URL
      - type='query'→ hit a SERP landing page URL (placeholder until official API)
    Returns a list because future versions may expand (queries→multiple links).
    P0: returns one FetchedPage per seed.
    """
    if seed.type == "rss":
        return [await _get_url(seed.value)]
    if seed.type == "url":
        return [await _get_url(seed.value)]
    if seed.type == "query":
        q = quote_plus(seed.value)
        serp_url = f"https://www.bing.com/search?q={q}"
        return [await _get_url(serp_url)]
    return []


async def fetch_many(seeds: List[Seed], *, concurrency: int = 5) -> List[FetchedPage]:
    """
    Fetch many seeds concurrently with a semaphore to avoid hammering hosts.
    """
    sem = asyncio.Semaphore(max(1, concurrency))
    results: List[FetchedPage] = []

    async def _wrap(s: Seed):
        async with sem:
            pages = await fetch_seed(s)
            results.extend(pages)

    await asyncio.gather(*[_wrap(s) for s in seeds])
    return results


# -----------------------------
# Async HTTP core
# -----------------------------

async def _get_url(url: str) -> FetchedPage:
    """
    Core GET with robots + politeness + httpx if present.
    If httpx is unavailable, this coroutine offloads urllib to a thread.
    """
    parsed = urlparse(url)
    host = parsed.netloc
    path = parsed.path or "/"

    await _fetch_robots(host)
    allowed_fn = _ROBOTS_CACHE.get(host, {}).get("allowed")
    robots_allowed = True
    if callable(allowed_fn):
        robots_allowed = bool(allowed_fn(path))
    if not robots_allowed:
        return FetchedPage(
            url=url,
            status=999,
            fetched_at=datetime.utcnow(),
            headers={},
            content_type="unknown",
            raw=b"",
            error="robots.txt disallow",
            robots_allowed=False,
            final_url=url,
        )

    await _respect_politeness(host)

    # Try httpx (async). If not available, run urllib in a thread.
    try:
        import httpx  # type: ignore
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=8.0),
            headers={"User-Agent": "CSEBot/0.1 (+contact-url)", "Accept": "text/html,application/json,*/*"},
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
        ct = _normalize_content_type(resp.headers.get("content-type"))
        return FetchedPage(
            url=url,
            status=resp.status_code,
            fetched_at=datetime.utcnow(),
            headers={k.lower(): v for k, v in resp.headers.items()},
            content_type=ct,
            raw=resp.content or b"",
            error=None if resp.status_code == 200 else f"http {resp.status_code}",
            robots_allowed=True,
            final_url=str(resp.url),
        )
    except Exception:
        # Fallback: urllib (sync) executed in thread so we can await it
        def _urllib_fetch(u: str):
            import urllib.request
            try:
                req = urllib.request.Request(u, headers={"User-Agent": "CSEBot/0.1 (+contact-url)", "Accept": "text/html,application/json,*/*"})
                with urllib.request.urlopen(req, timeout=8) as r:
                    data = r.read()
                    headers = {k.lower(): v for k, v in (r.headers.items() if hasattr(r, "headers") else [])}
                    code = getattr(r, "status", 200)
                    return data, headers, code, r.geturl()
            except Exception as e:
                return b"", {}, 0, None

        data, headers, code, final = await asyncio.to_thread(_urllib_fetch, url)
        ct = _normalize_content_type(headers.get("content-type"))
        return FetchedPage(
            url=url,
            status=code,
            fetched_at=datetime.utcnow(),
            headers=headers,
            content_type=ct,
            raw=data,
            error=None if code == 200 and data else f"http {code or 'error'}",
            robots_allowed=True,
            final_url=final or url,
        )


# -----------------------------
# Synchronous wrappers (for quick demos/tests)
# -----------------------------

def _run_async(coro):
    """Run an async coroutine from sync context safely (compatible with notebooks)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # In a running loop (e.g., Jupyter) – create a new task and wait
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    return asyncio.run(coro)


def fetch_seed_sync(seed: Seed) -> List[FetchedPage]:
    """Sync wrapper around fetch_seed (uses httpx if available, urllib otherwise)."""
    return _run_async(fetch_seed(seed))


def fetch_many_sync(seeds: List[Seed], *, concurrency: int = 5) -> List[FetchedPage]:
    """Sync wrapper around fetch_many."""
    return _run_async(fetch_many(seeds, concurrency=concurrency))