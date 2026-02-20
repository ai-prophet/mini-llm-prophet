"""Perplexity Search API integration for mini-llm-prophet.

Uses the Perplexity Search API (https://docs.perplexity.ai/docs/search/quickstart)
which returns structured web results with pre-extracted content in the snippet field.
No separate content extraction step (like trafilatura) is needed.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from miniprophet.environment.source_board import Source
from miniprophet.exceptions import SearchAuthError, SearchNetworkError, SearchRateLimitError
from miniprophet.search import SearchResult

logger = logging.getLogger("miniprophet.search.perplexity")

PERPLEXITY_API_ENDPOINT = "https://api.perplexity.ai/search"
# pricing from: https://docs.perplexity.ai/docs/getting-started/pricing
PERPLEXITY_PER_SEARCH_COST = 5 / 1000


class PerplexitySearchTool:
    """Search tool backed by the Perplexity Search API.

    Perplexity returns ranked web results with substantial extracted content
    already included in the snippet field, so no additional scraping is needed.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_tokens_per_page: int = 4096,
        max_tokens: int = 10000,
        country: str = "US",
    ) -> None:
        self._api_key = os.getenv("PERPLEXITY_API_KEY", "")
        self._timeout = timeout
        self._max_tokens_per_page = max_tokens_per_page
        self._max_tokens = max_tokens
        self._country = country

    def search(self, query: str, limit: int = 5) -> SearchResult:
        if not self._api_key:
            raise SearchAuthError("PERPLEXITY_API_KEY environment variable is not set")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "query": query,
            "max_results": min(limit, 20),
            "max_tokens_per_page": self._max_tokens_per_page,
            "max_tokens": self._max_tokens,
        }
        if self._country:
            payload["country"] = self._country

        try:
            resp = requests.post(
                PERPLEXITY_API_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            if resp.status_code == 401:
                raise SearchAuthError(
                    "Perplexity API authentication failed. Check PERPLEXITY_API_KEY."
                )
            if resp.status_code == 429:
                raise SearchRateLimitError("Perplexity API rate limit exceeded")
            raise SearchNetworkError(f"Perplexity API HTTP {resp.status_code}: {resp.text[:300]}")
        except requests.exceptions.RequestException as exc:
            raise SearchNetworkError(f"Perplexity API request failed: {exc}") from exc

        data = resp.json()
        sources: list[Source] = []
        for item in data.get("results", []):
            url = item.get("url", "")
            snippet = item.get("snippet", "")
            if url and snippet:
                sources.append(
                    Source(
                        url=url,
                        title=item.get("title", ""),
                        snippet=snippet[:200],
                        text=snippet,
                    )
                )

        logger.info(f"Perplexity search '{query}': {len(sources)} source(s)")
        # For perplexity, the cost is fixed for each request, regardless of the number of sources returned
        return SearchResult(sources=sources, cost=PERPLEXITY_PER_SEARCH_COST)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "search": {
                        "search_class": "perplexity",
                        "timeout": self._timeout,
                        "max_tokens_per_page": self._max_tokens_per_page,
                        "max_tokens": self._max_tokens,
                        "country": self._country,
                    }
                }
            }
        }
