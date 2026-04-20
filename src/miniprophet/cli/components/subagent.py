"""CLI display for subagent spawner tools (read_source, investigate_subproblem).

This module is the ONLY place that renders subagent status via Rich.  It
provides:

- :class:`SubagentStatusRenderable`: live-updating one-line status
- :func:`subagent_live_display`: context manager wrapping a ``rich.live.Live``
  for the duration of a subagent run; safely no-ops if another Live is active
- :func:`print_subagent_summary`: post-completion summary line + result panel

The subagent layer (``miniprophet.subagents``) and the tool layer
(``miniprophet.tools.read_source`` / ``investigate_subproblem``) are Rich-free;
they only touch :class:`SubagentStatus` (pure data) and optionally receive
:func:`subagent_live_display` as an injected ``display_context``.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager, nullcontext

from rich.live import Live
from rich.text import Text

from miniprophet.cli.components.observation import print_observation
from miniprophet.cli.utils import format_token_count, get_console
from miniprophet.subagents.status import SubagentStatus

# ---------------------------------------------------------------------------
# Live coordination: prevent concurrent Rich Lives on the same console.
# If a second subagent tries to open a Live while one is already active,
# it silently falls back to a nullcontext (its result still shows as a
# summary line after completion).
# ---------------------------------------------------------------------------

_live_lock = threading.Lock()


def _try_acquire_live() -> bool:
    return _live_lock.acquire(blocking=False)


def _release_live() -> None:
    try:
        _live_lock.release()
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# Live-updating status renderable
# ---------------------------------------------------------------------------


class SubagentStatusRenderable:
    """Rich renderable for a live-updating single-line subagent status.

    Example rendered output::

        ├─ ⟳ source_reading [S4, focus=injury] · Step 2/3 · calling submit_summary · $0.0023 · 2.4k tok
    """

    def __init__(self, status: SubagentStatus) -> None:
        self._status = status

    def __rich__(self):
        s = self._status
        text = Text("  ├─ ", style="dim cyan")
        text.append("⟳ ", style="cyan")
        text.append(s.kind, style="bold cyan")

        if s.label:
            text.append(" [", style="dim")
            text.append(s.label, style="magenta")
            text.append("]", style="dim")

        parts: list[tuple[str, str]] = []
        if s.step_limit:
            parts.append((f"Step {s.step}/{s.step_limit}", "bold"))
        else:
            parts.append((f"Step {s.step}", "bold"))

        if s.state:
            parts.append((s.state, "yellow"))

        parts.append((f"${s.total_cost:.4f}", "dim green"))

        if s.prompt_tokens:
            parts.append((format_token_count(s.prompt_tokens) + " tok", "dim"))

        for label, style in parts:
            text.append("  ·  ", style="dim")
            text.append(label, style=style)

        return text


# ---------------------------------------------------------------------------
# Context manager for a subagent's Live display
# ---------------------------------------------------------------------------


@contextmanager
def subagent_live_display(status: SubagentStatus):
    """Context manager that shows a live-updating status line during a
    subagent run.  Transient: the line disappears when the subagent finishes
    so the caller can print a clean summary in its place.

    If another Live is already active (e.g. a sibling subagent is mid-run),
    this context is a no-op — the subagent still executes, just without a
    live status; its summary will be printed normally once complete.
    """
    if not _try_acquire_live():
        # Another Live already holds the console; run silently.
        with nullcontext():
            yield
        return

    try:
        renderable = SubagentStatusRenderable(status)
        console = get_console()
        with Live(
            renderable,
            console=console,
            refresh_per_second=3,
            transient=True,
        ):
            yield
    finally:
        _release_live()


# ---------------------------------------------------------------------------
# Post-completion summary
# ---------------------------------------------------------------------------


def print_subagent_summary(action: dict, output: dict) -> None:
    """Render a post-completion summary for a subagent spawner tool.

    Layout::

        ╰─ ✓ read_source [S4, focus=injury] · 2 steps · 3 tool calls · $0.0023 · 2.4k tok

        ╭─ Observation ────────────────────────╮
        │ LeBron is healthy.                   │
        ╰──────────────────────────────────────╯
    """
    console = get_console()

    tool_name = action.get("name", "")
    summary = output.get("output", "") or ""
    n_steps = output.get("n_steps", 0) or 0
    n_tool_calls = output.get("n_tool_calls", 0) or 0
    cost = (output.get("model_cost", 0.0) or 0.0) + (output.get("search_cost", 0.0) or 0.0)
    prompt_tokens = output.get("prompt_tokens", 0) or 0
    label = output.get("subagent_label", "") or ""
    error = output.get("error", False)

    if error:
        console.print(Text(f"  ╰─ ✗ {tool_name} failed", style="red"))
        print_observation(output)
        return

    line = Text("  ╰─ ", style="dim cyan")
    line.append("✓ ", style="green")
    line.append(tool_name, style="bold cyan")
    if label:
        line.append(" [", style="dim")
        line.append(label, style="magenta")
        line.append("]", style="dim")

    stats = [f"{n_steps} steps", f"{n_tool_calls} tool calls", f"${cost:.4f}"]
    if prompt_tokens:
        stats.append(format_token_count(prompt_tokens) + " tok")
    for s in stats:
        line.append("  ·  ", style="dim")
        line.append(s, style="dim")

    console.print(line)
    print_observation({"output": summary})
