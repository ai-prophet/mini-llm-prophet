"""InvestigateSubproblemTool: main agent tool that spawns a SubproblemAgent.

This module is display-agnostic.  Callers (CLI, batch, etc.) inject a
``display_context`` callable for live progress rendering; the default is
a silent :class:`contextlib.nullcontext`.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext

INVESTIGATE_SUBPROBLEM_SCHEMA = {
    "type": "function",
    "function": {
        "name": "investigate_subproblem",
        "description": (
            "Delegate a binary yes/no sub-problem to an investigator assistant. "
            "The assistant will search, read sources, and return a probability "
            "(P(Yes) for the sub-problem) along with a brief report. "
            "Use this for sub-problems from your plan that require their own "
            "mini-investigation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": (
                        "The sub-problem as a binary yes/no question "
                        "(e.g. 'Will LeBron James be absent from the game?')."
                    ),
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Background context for the investigator: what the "
                        "sub-problem relates to, why it matters, known facts."
                    ),
                },
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of relevant source IDs from your existing "
                        "research (e.g. ['S2', 'S5']) for the investigator to start from."
                    ),
                },
            },
            "required": ["title", "context"],
        },
    },
}


class InvestigateSubproblemTool:
    """Spawns a SubproblemAgent; returns its probability + report.

    Parameters mirror :class:`ReadSourceTool`: ``subagent_factory`` builds the
    subagent, ``display_context`` optionally wraps the subagent run for live
    UI rendering.
    """

    def __init__(
        self,
        *,
        subagent_factory: Callable,
        display_context: Callable | None = None,
    ) -> None:
        self._factory = subagent_factory
        self._display_context = display_context or (lambda _status: nullcontext())

    @property
    def name(self) -> str:
        return "investigate_subproblem"

    def get_schema(self) -> dict:
        return INVESTIGATE_SUBPROBLEM_SCHEMA

    async def execute(self, args: dict) -> dict:
        title = str(args.get("title", "")).strip()
        context = str(args.get("context", "")).strip()
        source_ids = args.get("source_ids") or []
        if not isinstance(source_ids, list):
            source_ids = []
        source_ids = [str(s).strip() for s in source_ids if str(s).strip()]

        if not title:
            return {"output": "Error: 'title' is required.", "error": True}
        if not context:
            return {"output": "Error: 'context' is required.", "error": True}

        subagent = self._factory()
        try:
            with self._display_context(subagent.status):
                result = await subagent.run(title=title, context=context, source_ids=source_ids)
        except Exception as exc:
            return {"output": f"Subagent error: {exc}", "error": True}

        formatted = (
            f"<subproblem_result>\n"
            f"  <title>{title}</title>\n"
            f"  <probability>{result.probability}</probability>\n"
            f"  <report>{result.report}</report>\n"
            f"</subproblem_result>"
        )

        return {
            "output": formatted,
            "model_cost": result.model_cost,
            "search_cost": result.search_cost,
            "n_steps": result.n_steps,
            "n_tool_calls": result.n_tool_calls,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "exit_status": result.exit_status,
            "subagent_kind": "subproblem",
            "subagent_label": title[:60] + ("..." if len(title) > 60 else ""),
            "probability": result.probability,
            "report": result.report,
        }

    def display(self, output: dict) -> None:
        from miniprophet.cli.components.observation import print_observation

        print_observation({"output": output.get("output", "")})
