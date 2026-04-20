"""SourceReadingAgent: reads one source, returns a focused summary."""

from __future__ import annotations

from dataclasses import dataclass

from miniprophet.subagents.base import SubagentBase


@dataclass
class SourceReadingResult:
    summary: str
    model_cost: float
    search_cost: float
    n_steps: int
    rendered_trace: str
    exit_status: str = "summary_submitted"


class SourceReadingAgent(SubagentBase):
    """Reads one source and produces a focused summary.

    Tools: ``retrieve_source``, ``submit_summary``.  Typical flow is 2
    model calls: retrieve the full content, then submit a summary.
    """

    def _instance_prompt(self, source_id: str, focus: str = "", **_) -> str:
        base = f"Read source {source_id}"
        if focus:
            base += f", focusing on: {focus}"
        return (
            f"{base}.\n\n"
            "Workflow:\n"
            f"1. Call `retrieve_source` with source_id={source_id} to fetch the full text.\n"
            "2. Call `submit_summary` with a concise summary (3-6 sentences).\n"
            "If a focus is given, prioritize details related to that focus. "
            "Do NOT speculate beyond what the source says."
        )

    def _build_result(self, rendered_trace: str) -> SourceReadingResult:
        last_extra = self.messages[-1].get("extra", {}) if self.messages else {}
        return SourceReadingResult(
            summary=last_extra.get("summary", "")
            or "(No summary produced — subagent ended without submit_summary.)",
            model_cost=self.model_cost,
            search_cost=self.search_cost,
            n_steps=self.n_calls,
            rendered_trace=rendered_trace,
            exit_status=last_extra.get("exit_status", "unknown"),
        )
