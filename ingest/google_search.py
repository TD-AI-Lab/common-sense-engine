"""
Google Custom Search API wrapper for ingestion seeds.
Robuste + paginé + fallback HTTP si googleapiclient indisponible.
"""

from typing import List
import logging

from utils.logger import get_logger

logger = get_logger(__name__)

GOOGLE_JSON_API = "https://www.googleapis.com/customsearch/v1"


def _search_with_sdk(topic: str, api_key: str, cx: str, n_results: int) -> List[str]:
    """Utilise googleapiclient.discovery.build avec pagination (start)."""
    try:
        from googleapiclient.discovery import build
    except Exception as e:
        logger.debug(f"googleapiclient indisponible: {e}")
        raise

    service = build("customsearch", "v1", developerKey=api_key, cache_discovery=False)

    urls: List[str] = []
    start = 1  # 1-based index
    per_page = 10
    seen = set()

    while len(urls) < n_results and start <= 91:  # CSE limite ~100 résultats
        num = min(per_page, n_results - len(urls))
        try:
            res = service.cse().list(q=topic, cx=cx, num=num, start=start, safe="off").execute()
        except Exception as e:
            logger.warning(f"[google] SDK call failed at start={start}: {e}")
            break
        items = res.get("items", []) or []
        if not items:
            break
        for it in items:
            link = it.get("link")
            if link and link not in seen:
                seen.add(link)
                urls.append(link)
        start += len(items)

    return urls


def _search_with_http(topic: str, api_key: str, cx: str, n_results: int) -> List[str]:
    """Fallback HTTP direct sur l'endpoint JSON API (sans googleapiclient)."""
    import requests

    urls: List[str] = []
    start = 1
    per_page = 10
    seen = set()

    while len(urls) < n_results and start <= 91:
        num = min(per_page, n_results - len(urls))
        params = {
            "q": topic,
            "cx": cx,
            "key": api_key,
            "num": num,
            "start": start,
            "safe": "off",
            # astuce: réduire la payload renvoyée
            "fields": "items(link),searchInformation/totalResults",
        }
        try:
            r = requests.get(GOOGLE_JSON_API, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning(f"[google] HTTP call failed at start={start}: {e}")
            break

        items = (data or {}).get("items", []) or []
        if not items:
            break
        for it in items:
            link = it.get("link")
            if link and link not in seen:
                seen.add(link)
                urls.append(link)
        start += len(items)

    return urls


def search_google(topic: str, api_key: str, cx: str, n_results: int = 10) -> List[str]:
    """
    Perform a Google Custom Search for the given topic (paginated, deduped).
    Returns a list of URLs (strings). In case of error, returns [].
    """
    if not api_key or not cx:
        logger.warning("[google] Missing API key or CX.")
        return []
    n_results = max(1, min(int(n_results or 10), 50))  # garde-fou simple

    try:
        return _search_with_sdk(topic, api_key, cx, n_results)
    except Exception:
        # Fallback HTTP direct si le SDK n'est pas dispo ou échoue
        return _search_with_http(topic, api_key, cx, n_results)