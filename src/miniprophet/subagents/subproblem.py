"""SubproblemAgent: investigates a binary sub-problem, returns a probability."""

from __future__ import annotations

from dataclasses import dataclass

from miniprophet.subagents.base import SubagentBase, SubagentConfig


class SubproblemSubagentConfig(SubagentConfig):
    """Config for SubproblemAgent — includes its own search budget."""

    search_limit: int = 3


@dataclass
class SubproblemResult:
    probability: float
    report: str
    model_cost: float
    search_cost: float
    n_steps: int
    n_tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    exit_status: str = "subproblem_submitted"


class SubproblemAgent(SubagentBase):
    """Investigates a binary sub-problem.

    Tools: ``search``, ``list_sources``, ``retrieve_source``, ``submit_subproblem``.
    Runs a full search-and-read agent loop, typically 5-10 steps.
    """

    kind = "subproblem"

    def _status_label(
        self,
        title: str = "",
        context: str = "",
        source_ids: list[str] | None = None,
        **_,
    ) -> str:
        short = title if len(title) <= 60 else title[:57] + "..."
        return short

    def _instance_prompt(
        self,
        title: str,
        context: str,
        source_ids: list[str] | None = None,
        **_,
    ) -> str:
        sids = ", ".join(source_ids) if source_ids else "(none yet)"
        return (
            "Investigate this binary yes/no sub-problem and submit a probability "
            "P(Yes).\n\n"
            f"<sub_problem>\n"
            f"  <title>{title}</title>\n"
            f"  <context>{context}</context>\n"
            f"  <relevant_source_ids>{sids}</relevant_source_ids>\n"
            f"</sub_problem>\n\n"
            "Workflow:\n"
            "1. Use `search` for relevant information; the main agent may already "
            "have discovered sources (see relevant_source_ids above).\n"
            "2. Use `retrieve_source` to read full source content when a preview "
            "looks promising.\n"
            "3. Use `list_sources` to review all discovered sources.\n"
            "4. When ready, call `submit_subproblem` with your P(Yes) and a brief "
            "report (3-6 sentences) on key evidence and reasoning.\n\n"
            "Stay focused on THIS sub-problem; you do not need to answer the main "
            "forecasting question."
        )

    def _build_result(self) -> SubproblemResult:
        last_extra = self.messages[-1].get("extra", {}) if self.messages else {}
        return SubproblemResult(
            probability=float(last_extra.get("probability", 0.0) or 0.0),
            report=last_extra.get("report", "")
            or "(No report produced — subagent ended without submit_subproblem.)",
            model_cost=self.model_cost,
            search_cost=self.search_cost,
            n_steps=self.n_calls,
            n_tool_calls=self.n_tool_calls,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            exit_status=last_extra.get("exit_status", "unknown"),
        )
