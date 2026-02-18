"""ForecastEnvironment: dispatches tool calls to search, source board, and submit."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

from miniprophet.environment.source_board import Source, SourceBoard
from miniprophet.exceptions import (
    SearchAuthError,
    SearchError,
    Submitted,
)
from miniprophet.search import SearchResult, SearchTool

logger = logging.getLogger("miniprophet.environment")

# ---------------------------------------------------------------------------
# OpenAI function-calling tool schemas
# ---------------------------------------------------------------------------

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search the web for information relevant to the forecasting problem. "
            "Returns a list of sources with titles, snippets, and article content."
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

ADD_SOURCE_TOOL = {
    "type": "function",
    "function": {
        "name": "add_source",
        "description": (
            "Add a source from the most recent search results to your source board. "
            "Include an analytical note about the source's relevance, reliability, "
            "and key insights for the forecast."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_index": {
                    "type": "integer",
                    "description": "Zero-based index of the source in the latest search results.",
                },
                "note": {
                    "type": "string",
                    "description": "Your analytical note about this source.",
                },
            },
            "required": ["source_index", "note"],
        },
    },
}

EDIT_NOTE_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_note",
        "description": (
            "Edit the note of a previously added source on the board. "
            "Use this to update your analysis as new information becomes available "
            "(e.g. mark a source as unreliable based on contradicting evidence)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "board_id": {
                    "type": "integer",
                    "description": "The board entry ID (shown as #N on the board).",
                },
                "new_note": {
                    "type": "string",
                    "description": "The updated analytical note.",
                },
            },
            "required": ["board_id", "new_note"],
        },
    },
}

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": (
            "Submit your final probabilistic forecast. Provide a probability "
            "(between 0 and 1) for EVERY listed outcome. The probabilities should "
            "reflect the balance of evidence gathered on your source board."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "probabilities": {
                    "type": "object",
                    "description": (
                        "A JSON object mapping each outcome name (exactly as listed) "
                        "to a probability value between 0 and 1."
                    ),
                },
            },
            "required": ["probabilities"],
        },
    },
}

ALL_TOOLS = [SEARCH_TOOL, ADD_SOURCE_TOOL, EDIT_NOTE_TOOL, SUBMIT_TOOL]


class ForecastEnvConfig(BaseModel):
    search_results_limit: int = 5
    max_source_text_chars: int = 2000


class ForecastEnvironment:
    """Dispatches tool-call actions and manages internal state (source board)."""

    def __init__(
        self,
        search_tool: SearchTool,
        outcomes: list[str],
        *,
        search_limit: int = 10,
        config_class: type = ForecastEnvConfig,
        **kwargs: Any,
    ) -> None:
        self.config = config_class(**kwargs)
        self.search_tool = search_tool
        self.outcomes = outcomes
        self.search_limit = search_limit
        self.board = SourceBoard()
        self.last_search_results: list[Source] = []
        self.n_searches = 0

    def execute(self, action: dict) -> dict:
        tool_name = action.get("name", "")
        try:
            raw_args = action.get("arguments", "{}")
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError as exc:
            return {"output": f"Invalid JSON in tool arguments: {exc}", "error": True}

        match tool_name:
            case "search":
                return self._handle_search(args)
            case "add_source":
                return self._handle_add_source(args)
            case "edit_note":
                return self._handle_edit_note(args)
            case "submit":
                return self._handle_submit(args)
            case _:
                return {"output": f"Unknown tool: {tool_name}", "error": True}

    def get_tool_schemas(self) -> list[dict]:
        return list(ALL_TOOLS)

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def _handle_search(self, args: dict) -> dict:
        query = args.get("query", "").strip()
        if not query:
            return {"output": "Error: 'query' is required for the search tool.", "error": True}

        if self.n_searches >= self.search_limit:
            return {
                "output": (
                    f"Search limit reached ({self.search_limit} queries). "
                    "Use your existing sources to submit a forecast."
                ),
                "error": True,
            }

        try:
            result: SearchResult = self.search_tool.search(
                query, limit=self.config.search_results_limit
            )
        except SearchAuthError:
            raise
        except SearchError as exc:
            return {"output": f"Search failed: {exc}. Try again or use existing sources.", "error": True}

        self.n_searches += 1
        self.last_search_results = result.sources

        if not result.sources:
            body = "No sources found for this query."
        else:
            lines: list[str] = [f"Found {len(result.sources)} source(s):\n"]
            for i, src in enumerate(result.sources):
                text_preview = src.text[: self.config.max_source_text_chars]
                lines.append(
                    f"[{i}] \"{src.title}\" ({src.url})\n"
                    f"    Snippet: {src.snippet}\n"
                    f"    Content: {text_preview}\n"
                )
            body = "\n".join(lines)

        output = body + "\n\n" + self.board.render()
        return {"output": output, "search_cost": result.cost}

    def _handle_add_source(self, args: dict) -> dict:
        source_index = args.get("source_index")
        note = args.get("note", "").strip()

        if source_index is None:
            return {"output": "Error: 'source_index' is required.", "error": True}
        if not note:
            return {"output": "Error: 'note' is required.", "error": True}
        if not self.last_search_results:
            return {"output": "Error: no search results available. Run a search first.", "error": True}
        if not (0 <= source_index < len(self.last_search_results)):
            return {
                "output": (
                    f"Error: source_index {source_index} out of range. "
                    f"Valid range: 0-{len(self.last_search_results) - 1}."
                ),
                "error": True,
            }

        source = self.last_search_results[source_index]
        entry = self.board.add(source, note)
        body = f"Source added to board as #{entry.id}."
        output = body + "\n\n" + self.board.render()
        return {"output": output}

    def _handle_edit_note(self, args: dict) -> dict:
        board_id = args.get("board_id")
        new_note = args.get("new_note", "").strip()

        if board_id is None:
            return {"output": "Error: 'board_id' is required.", "error": True}
        if not new_note:
            return {"output": "Error: 'new_note' is required.", "error": True}

        try:
            entry = self.board.edit_note(board_id, new_note)
        except KeyError:
            return {"output": f"Error: no board entry with id #{board_id}.", "error": True}

        body = f"Note for #{entry.id} updated."
        output = body + "\n\n" + self.board.render()
        return {"output": output}

    def _handle_submit(self, args: dict) -> dict:
        probabilities = args.get("probabilities")
        if not isinstance(probabilities, dict):
            return {"output": "Error: 'probabilities' must be a JSON object mapping outcomes to values.", "error": True}

        errors: list[str] = []
        for outcome in self.outcomes:
            if outcome not in probabilities:
                errors.append(f"Missing probability for outcome: '{outcome}'.")
        for key, val in probabilities.items():
            if key not in self.outcomes:
                errors.append(f"Unknown outcome: '{key}'.")
            elif not isinstance(val, (int, float)) or not (0 <= val <= 1):
                errors.append(f"Probability for '{key}' must be a number between 0 and 1, got {val}.")

        if errors:
            return {"output": "Submission rejected:\n" + "\n".join(errors), "error": True}

        raise Submitted(
            {
                "role": "exit",
                "content": "Forecast submitted successfully.",
                "extra": {
                    "exit_status": "submitted",
                    "submission": probabilities,
                    "board": self.board.serialize(),
                },
            }
        )

    # ------------------------------------------------------------------

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "outcomes": self.outcomes,
                },
                "board": self.board.serialize(),
            },
        }
