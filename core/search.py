"""Tavily — the independent source behind C's VERIFIED/CONTRADICTED verdicts.

Two things that make this defensible rather than decorative:
  1. Every result keeps its URL + snippet, so a verdict can cite something.
  2. Results are UNTRUSTED (a founder can plant a page) — callers must route
     snippets through llm.complete(untrusted=...), never raw into a prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel

from core.config import settings

CACHE_DIR = Path("data/raw/tavily")

# Domains where a founder controls the content. Corroboration from here is weak.
SELF_PUBLISHED_HINTS = ("linkedin.com", "medium.com", "substack.com", "twitter.com", "x.com")


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    score: float = 0.0
    published_at: str | None = None  # often absent — see B.md on date_inferred
    self_published: bool = False


def search(query: str, *, max_results: int = 5, days: int | None = None) -> list[SearchResult]:
    """Cached web search. Empty results mean UNVERIFIABLE, never CONTRADICTED."""
    key = "".join(c if c.isalnum() else "_" for c in query)[:80]
    cache_file = CACHE_DIR / f"{key}_{max_results}.json"
    if cache_file.exists():
        return [SearchResult(**r) for r in json.loads(cache_file.read_text())]

    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.tavily_api_key)
    raw = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        **({"days": days} if days else {}),
    )

    results = [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=r.get("content", ""),
            score=r.get("score", 0.0),
            published_at=r.get("published_date"),
            self_published=_is_self_published(r.get("url", "")),
        )
        for r in raw.get("results", [])
    ]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps([r.model_dump() for r in results]))
    return results


def _is_self_published(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(h in host for h in SELF_PUBLISHED_HINTS)
