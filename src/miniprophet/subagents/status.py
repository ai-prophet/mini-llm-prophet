"""SubagentStatus: pure data describing a subagent's live state.

This module deliberately has no Rich / CLI dependencies.  Rendering
(live display, summary lines) lives in :mod:`miniprophet.cli.components.subagent`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SubagentStatus:
    """Mutable status object shared between a subagent and its display.

    Subagent hooks write to this object as the run progresses.  A UI layer
    (if any) polls the object and renders it.  When no UI is attached, the
    status is simply ignored — no performance cost.
    """

    kind: str = "subagent"  # e.g. "source_reading", "subproblem"
    label: str = ""  # brief descriptor (e.g. "S4, focus=injury")
    state: str = "starting"  # "starting", "thinking", "calling: X", "done"
    step: int = 0
    step_limit: int = 0
    n_tool_calls: int = 0
    model_cost: float = 0.0
    search_cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_cost(self) -> float:
        return self.model_cost + self.search_cost
