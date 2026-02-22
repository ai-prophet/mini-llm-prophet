"""Search tool: web search with source ID assignment."""

from __future__ import annotations

import logging

from pydantic import BaseModel

from miniprophet.environment.source_board import Source
from miniprophet.exceptions import SearchAuthError, SearchError
from miniprophet.search import SearchResult, SearchTool

logger = logging.getLogger("miniprophet.tools.search")

SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search the web for information relevant to the forecasting problem. "
            "Returns a list of sources with titles, snippets, and article content. "
            "Each source is assigned a global ID (S1, S2, ...) that persists across searches."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant information.",
                },
            },
            "required": ["query"],
        },
    },
}


class SearchToolConfig(BaseModel):
    search_results_limit: int = 5
    max_source_text_chars: int = 2000


class SearchForecastTool:
    """Wraps a SearchTool backend into a forecast Tool with source ID tracking."""

    def __init__(
        self,
        search_backend: SearchTool,
        source_registry: dict[str, Source],
        *,
        search_limit: int = 10,
        config: SearchToolConfig | None = None,
    ) -> None:
        self._backend = search_backend
        self._source_registry = source_registry
        self._search_limit = search_limit
        self._config = config or SearchToolConfig()
        self._next_source_id: int = 1
        self.n_searches: int = 0
        self.last_search_results: list[tuple[str, Source]] = []

    @property
    def name(self) -> str:
        return "search"

    def get_schema(self) -> dict:
        return SEARCH_SCHEMA

    def _assign_source_id(self, source: Source) -> str:
        sid = f"S{self._next_source_id}"
        self._next_source_id += 1
        self._source_registry[sid] = source
        return sid

    def execute(self, args: dict) -> dict:
        query = args.get("query", "").strip()
        if not query:
            return {"output": "Error: 'query' is required for the search tool.", "error": True}

        if self.n_searches >= self._search_limit:
            return {
                "output": (
                    f"Search limit reached ({self._search_limit} queries). "
                    "Use your existing sources to submit a forecast."
                ),
                "error": True,
            }

        try:
            result: SearchResult = self._backend.search(
                query, limit=self._config.search_results_limit
            )
        except SearchAuthError:
            raise
        except SearchError as exc:
            return {
                "output": f"Search failed: {exc}. Try again or use existing sources.",
                "error": True,
            }

        self.n_searches += 1
        self.last_search_results = [(self._assign_source_id(src), src) for src in result.sources]

        if not self.last_search_results:
            body = "No sources found for this query."
        else:
            lines: list[str] = [f"Found {len(self.last_search_results)} source(s):\n"]
            for sid, src in self.last_search_results:
                text_preview = src.text[: self._config.max_source_text_chars]
                lines.append(
                    f'[{sid}] "{src.title}" ({src.url})\n'
                    f"    Snippet: {src.snippet}\n"
                    f"    Content: {text_preview}\n"
                )
            body = "\n".join(lines)

        return {"output": body, "search_cost": result.cost}

    def display(self, output: dict) -> None:
        from miniprophet.cli.components.observation import print_observation
        from miniprophet.cli.components.search_results import print_search_observation

        raw = output.get("output", "")
        if not output.get("error") and raw.startswith("Found ") and "source(s):" in raw[:40]:
            print_search_observation(raw)
        else:
            print_observation(output)
