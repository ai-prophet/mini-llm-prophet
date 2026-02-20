"""ForecastEnvironment: dispatches tool calls to search, source board, and submit."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

from miniprophet.environment.source_board import VALID_SENTIMENTS, Source, SourceBoard
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

ADD_SOURCE_TOOL = {
    "type": "function",
    "function": {
        "name": "add_source",
        "description": (
            "Add a source from search results to your source board. "
            "Include an analytical note and optionally a reaction per outcome."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "The global source ID (e.g. 'S3') from search results.",
                },
                "note": {
                    "type": "string",
                    "description": "Your analytical note about this source.",
                },
                "reaction": {
                    "type": "object",
                    "description": (
                        "Optional. Map outcome names to sentiment: "
                        "'very_positive', 'positive', 'neutral', 'negative', 'very_negative'. "
                        "Only include outcomes this source is relevant to."
                    ),
                },
            },
            "required": ["source_id", "note"],
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
                "reaction": {
                    "type": "object",
                    "description": (
                        "Optional. Updated reaction map (outcome -> sentiment). "
                        "Replaces the previous reaction entirely if provided."
                    ),
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
        self._source_registry: dict[str, Source] = {}
        self._next_source_id: int = 1
        self.last_search_results: list[tuple[str, Source]] = []
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
    # Helpers
    # ------------------------------------------------------------------

    def _assign_source_id(self, source: Source) -> str:
        sid = f"S{self._next_source_id}"
        self._next_source_id += 1
        self._source_registry[sid] = source
        return sid

    def _validate_reaction(self, reaction: dict | None) -> tuple[dict[str, str], list[str]]:
        """Validate a reaction dict. Returns (cleaned, errors)."""
        if not reaction:
            return {}, []
        errors: list[str] = []
        cleaned: dict[str, str] = {}
        for key, val in reaction.items():
            if key not in self.outcomes:
                errors.append(f"Unknown outcome in reaction: '{key}'.")
            elif val not in VALID_SENTIMENTS:
                errors.append(
                    f"Invalid sentiment for '{key}': '{val}'. "
                    f"Must be one of: {', '.join(sorted(VALID_SENTIMENTS))}."
                )
            else:
                cleaned[key] = val
        return cleaned, errors

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
                text_preview = src.text[: self.config.max_source_text_chars]
                lines.append(
                    f'[{sid}] "{src.title}" ({src.url})\n'
                    f"    Snippet: {src.snippet}\n"
                    f"    Content: {text_preview}\n"
                )
            body = "\n".join(lines)

        output = body + "\n\n" + self.board.render()
        return {"output": output, "search_cost": result.cost}

    def _handle_add_source(self, args: dict) -> dict:
        source_id = args.get("source_id", "")
        if isinstance(source_id, int):
            source_id = f"S{source_id}"
        source_id = str(source_id).strip().upper()
        note = args.get("note", "").strip()
        raw_reaction = args.get("reaction")

        if not source_id:
            return {"output": "Error: 'source_id' is required (e.g. 'S3').", "error": True}
        if not note:
            return {"output": "Error: 'note' is required.", "error": True}
        if source_id not in self._source_registry:
            valid_ids = ", ".join(sorted(self._source_registry.keys(), key=lambda s: int(s[1:])))
            return {
                "output": f"Error: unknown source_id '{source_id}'. Valid IDs: {valid_ids or '(none)'}.",
                "error": True,
            }

        reaction, reaction_errors = self._validate_reaction(raw_reaction)
        if reaction_errors:
            return {
                "output": "Reaction validation errors:\n" + "\n".join(reaction_errors),
                "error": True,
            }

        source = self._source_registry[source_id]
        entry = self.board.add(source, note, reaction=reaction)
        body = f"Source {source_id} added to board as #{entry.id}."
        output = body + "\n\n" + self.board.render()
        return {"output": output}

    def _handle_edit_note(self, args: dict) -> dict:
        board_id = args.get("board_id")
        new_note = args.get("new_note", "").strip()
        raw_reaction = args.get("reaction")

        if board_id is None:
            return {"output": "Error: 'board_id' is required.", "error": True}
        if not new_note:
            return {"output": "Error: 'new_note' is required.", "error": True}

        reaction, reaction_errors = self._validate_reaction(raw_reaction)
        if reaction_errors:
            return {
                "output": "Reaction validation errors:\n" + "\n".join(reaction_errors),
                "error": True,
            }

        try:
            entry = self.board.edit_note(
                board_id, new_note, reaction=reaction if raw_reaction is not None else None
            )
        except KeyError:
            return {"output": f"Error: no board entry with id #{board_id}.", "error": True}

        body = f"Note for #{entry.id} updated."
        output = body + "\n\n" + self.board.render()
        return {"output": output}

    def _handle_submit(self, args: dict) -> dict:
        probabilities = args.get("probabilities")
        if not isinstance(probabilities, dict):
            return {
                "output": "Error: 'probabilities' must be a JSON object mapping outcomes to values.",
                "error": True,
            }

        errors: list[str] = []
        for outcome in self.outcomes:
            if outcome not in probabilities:
                errors.append(f"Missing probability for outcome: '{outcome}'.")
        for key, val in probabilities.items():
            if key not in self.outcomes:
                errors.append(f"Unknown outcome: '{key}'.")
            elif not isinstance(val, int | float) or not (0 <= val <= 1):
                errors.append(
                    f"Probability for '{key}' must be a number between 0 and 1, got {val}."
                )

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
