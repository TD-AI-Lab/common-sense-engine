"""
Seed generation (discovery) for open-web ingestion.

Goal (P0):
- Given (topic, lang[], since, k), produce ~N seeds mixing:
  - Search API queries (e.g., Bing, Programmable Search) — *optional if no key*
  - RSS feeds (Reuters, FAO, World Bank, AP, UN)
  - Whitelist URLs (publisher homepages/sections)

Design:
- Providers are pluggable.
- Never call the network here — just produce 'intent' objects.
- Deduplicate (type,value).
"""

from __future__ import annotations

from typing import List, Optional
from .types import Seed


def build_seeds(topic: str, *, lang: List[str], since: Optional[str], k: int) -> List[Seed]:
    """
    Compose seeds from multiple lightweight providers.
    Order by (RSS first → reliable), then queries, then whitelists.

    Implementation notes:
    - RSS: 3–5 stable feeds
    - Query templates: per language, neutral phrasing around causality
    - Whitelist: safe orgs
    - Cap to 20 seeds (fast-path)
    """
    seeds: List[Seed] = []
    seeds += _rss_seeds()
    seeds += _query_seeds(topic, lang, since)
    seeds += _whitelist_seeds(topic)

    # Deduplicate by (type, value)
    seen = set()
    unique: List[Seed] = []
    for s in seeds:
        key = (s.type, s.value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)

    return unique[:max(5, min(20, k))]


# -----------------------------
# Providers (internal)
# -----------------------------

def _rss_seeds() -> List[Seed]:
    """Return a small, high-signal set of RSS seeds (static URLs)."""
    return [
        Seed(type="rss", value="https://www.fao.org/newsroom/rss", provider="fao"),
        Seed(type="rss", value="https://www.reuters.com/rss", provider="reuters"),
        Seed(type="rss", value="https://www.worldbank.org/en/news/all?format=rss", provider="worldbank"),
        Seed(type="rss", value="https://apnews.com/hub/ap-top-news?utm_source=apnews.com&utm_medium=referral", provider="ap"),
        Seed(type="rss", value="https://www.un.org/press/en/rss.xml", provider="un"),
    ]


def _query_seeds(topic: str, langs: List[str], since: Optional[str]) -> List[Seed]:
    """
    Build search-intent seeds using language templates.
    Value contains the query string; the fetcher will turn it into a SERP URL or API call later.
    """
    templates_fr = [
        f"pourquoi {topic}",
        f"{topic} causes",
        f"raison {topic}",
        f"{topic} augmentation",
        f"impact {topic}",
    ]
    templates_en = [
        f"why {topic}",
        f"{topic} causes",
        f"due to {topic}",
        f"{topic} increase",
        f"impact of {topic}",
    ]
    seeds: List[Seed] = []
    if "fr" in langs:
        for q in templates_fr:
            seeds.append(Seed(type="query", value=q, provider="bing", lang="fr", topic=topic, since=since))
    if "en" in langs:
        for q in templates_en:
            seeds.append(Seed(type="query", value=q, provider="bing", lang="en", topic=topic, since=since))
    return seeds


def _whitelist_seeds(topic: str) -> List[Seed]:
    """Add safe whitelisted sources."""
    return [
        Seed(type="url", value="https://en.wikipedia.org/wiki/Main_Page", provider="wikipedia", topic=topic),
        Seed(type="url", value="https://www.oecd.org/", provider="oecd", topic=topic),
        Seed(type="url", value="https://www.worldbank.org/", provider="worldbank", topic=topic),
    ]