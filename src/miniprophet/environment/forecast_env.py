"""ForecastEnvironment: thin dispatcher that delegates to modular Tool instances."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

from miniprophet import Tool
from miniprophet.environment.source_board import Source, SourceBoard
from miniprophet.search import SearchTool

logger = logging.getLogger("miniprophet.environment")


def create_default_tools(
    search_tool: SearchTool,
    outcomes: list[str],
    board: SourceBoard,
    *,
    search_limit: int = 10,
    search_results_limit: int = 5,
    max_source_display_chars: int = 2000,
) -> list[Tool]:
    """Build the standard set of forecast tools sharing a common board and source registry."""
    from miniprophet.tools.search_tool import SearchForecastTool, SearchToolConfig
    from miniprophet.tools.source_board_tools import AddSourceTool, EditNoteTool
    from miniprophet.tools.submit import SubmitTool

    source_registry: dict[str, Source] = {}
    search_config = SearchToolConfig(
        search_results_limit=search_results_limit,
        max_source_display_chars=max_source_display_chars,
    )

    return [
        SearchForecastTool(
            search_backend=search_tool,
            source_registry=source_registry,
            search_limit=search_limit,
            config=search_config,
        ),
        AddSourceTool(source_registry=source_registry, board=board, outcomes=outcomes),
        EditNoteTool(board=board, outcomes=outcomes),
        SubmitTool(outcomes=outcomes, board=board),
    ]


class ForecastEnvConfig(BaseModel):
    search_results_limit: int = 5
    max_source_display_chars: int = 2000


class ForecastEnvironment:
    """Dispatches tool-call actions to registered Tool instances."""

    def __init__(
        self,
        tools: list[Tool],
        *,
        board: SourceBoard | None = None,
        **kwargs: Any,
    ) -> None:
        self.board = board or SourceBoard()
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    def execute(self, action: dict, **kwargs) -> dict:
        tool_name = action.get("name", "")
        try:
            raw_args = action.get("arguments", "{}")
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError as exc:
            return {"output": f"Invalid JSON in tool arguments: {exc}", "error": True}

        tool = self._tools.get(tool_name)
        if tool is None:
            return {"output": f"Unknown tool: {tool_name}", "error": True}
        # override the agent's args with runtime kwargs
        args.update(kwargs)
        return tool.execute(args)

    def get_tool_schemas(self) -> list[dict]:
        return [t.get_schema() for t in self._tools.values()]

    def get_tool(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def serialize(self) -> dict:
        return {
            "info": {
                "board": self.board.serialize(),
            },
        }
