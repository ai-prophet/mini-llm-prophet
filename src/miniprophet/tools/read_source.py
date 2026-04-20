"""ReadSourceTool: main agent tool that spawns a SourceReadingAgent.

The main agent cannot read raw sources directly — it calls this tool, which
delegates to a SourceReadingAgent.  The subagent reads the full content
and returns a focused summary.  The main agent's context only sees the
summary, not the raw source text.
"""

from __future__ import annotations

from collections.abc import Callable

READ_SOURCE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_source",
        "description": (
            "Delegate reading a source to a source-reading assistant. "
            "The assistant fetches the full content and returns a focused "
            "summary. Use this instead of trying to recall search preview text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "The source ID (e.g. 'S3') from search results.",
                },
                "focus": {
                    "type": "string",
                    "description": (
                        "Optional focus instruction for the reader (e.g. 'report "
                        "any mention of player injuries'). Leave empty for a general summary."
                    ),
                },
            },
            "required": ["source_id"],
        },
    },
}


class ReadSourceTool:
    """Spawns a SourceReadingAgent; returns its summary as the tool output."""

    def __init__(self, *, subagent_factory: Callable) -> None:
        self._factory = subagent_factory

    @property
    def name(self) -> str:
        return "read_source"

    def get_schema(self) -> dict:
        return READ_SOURCE_SCHEMA

    async def execute(self, args: dict) -> dict:
        source_id = str(args.get("source_id", "")).strip()
        focus = str(args.get("focus", "") or "").strip()

        if not source_id:
            return {"output": "Error: 'source_id' is required.", "error": True}

        subagent = self._factory()
        try:
            result = await subagent.run(source_id=source_id, focus=focus)
        except Exception as exc:
            return {"output": f"Subagent error: {exc}", "error": True}

        return {
            "output": result.summary,
            "model_cost": result.model_cost,
            "search_cost": result.search_cost,
            "n_steps": result.n_steps,
            "rendered_trace": result.rendered_trace,
            "subagent_kind": "source_reading",
        }

    def display(self, output: dict) -> None:
        # Main agent's hook renders spawner tool outputs specially; this is a
        # fallback for generic observation rendering.
        from miniprophet.cli.components.observation import print_observation

        print_observation({"output": output.get("output", "")})
