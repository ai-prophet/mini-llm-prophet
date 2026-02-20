"""Rich console trajectory display for the forecasting agent."""

from __future__ import annotations

import json
import re

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

REACTION_SYMBOLS: dict[str, str] = {
    "very_positive": "[bold green]++[/bold green]",
    "positive": "[green]+[/green]",
    "neutral": "[dim]~[/dim]",
    "negative": "[red]-[/red]",
    "very_negative": "[bold red]--[/bold red]",
}


def format_reaction(reaction: dict[str, str]) -> str:
    """Render a reaction dict as a compact Rich-markup string."""
    if not reaction:
        return ""
    parts = []
    for outcome, sentiment in reaction.items():
        symbol = REACTION_SYMBOLS.get(sentiment, "?")
        parts.append(f"{outcome} [{symbol}]")
    return "  ".join(parts)


def print_run_header(
    title: str, outcomes: str, step_limit: int, cost_limit: float, search_limit: int
) -> None:
    console.print()
    console.rule("[bold magenta]Forecasting Agent[/bold magenta]", style="magenta")
    console.print(f"  [bold]Question:[/bold] {title}")
    console.print(f"  [bold]Outcomes:[/bold] {outcomes}")
    console.print(
        f"  [bold]Limits:[/bold]   steps={step_limit}  "
        f"cost=${cost_limit:.2f}  searches={search_limit}"
    )
    console.print()


def print_run_footer(
    exit_status: str,
    n_calls: int,
    n_searches: int,
    model_cost: float,
    search_cost: float,
    total_cost: float,
) -> None:
    console.print()
    console.rule("[bold magenta]Agent Finished[/bold magenta]", style="magenta")
    console.print(
        f"  [bold]Status:[/bold]   {exit_status}\n"
        f"  [bold]Steps:[/bold]    {n_calls}   Searches: {n_searches}\n"
        f"  [bold]Cost:[/bold]     model=${model_cost:.4f}  "
        f"search=${search_cost:.4f}  total=${total_cost:.4f}"
    )
    console.print()


def print_evaluation(evaluation: dict[str, float]) -> None:
    """Render evaluation metrics as a Rich table."""
    if not evaluation:
        return
    table = Table(title="Evaluation", expand=False, show_edge=False)
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    for name, score in evaluation.items():
        table.add_row(name, f"{score:.4f}")
    console.print()
    console.print(table)


def print_step_header(step: int, model_cost: float, search_cost: float, total_cost: float) -> None:
    cost_str = f"model=${model_cost:.4f}  search=${search_cost:.4f}  total=${total_cost:.4f}"
    console.rule(f"[bold]Step {step}[/bold]  |  {cost_str}", style="cyan")


def print_model_response(message: dict) -> None:
    content = message.get("content") or ""
    actions = message.get("extra", {}).get("actions", [])

    if content:
        truncated = content[:500] + ("..." if len(content) > 500 else "")
        console.print(
            Panel(
                truncated,
                title="[bold yellow]Model Thinking[/bold yellow]",
                border_style="yellow",
                expand=False,
            )
        )

    for action in actions:
        name = action.get("name", "?")
        raw_args = action.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            args_display = json.dumps(args, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            args_display = str(raw_args)

        if len(args_display) > 300:
            args_display = args_display[:300] + "\n..."

        console.print(
            Panel(
                f"[bold]{name}[/bold]({args_display})",
                title="[bold green]Tool Call[/bold green]",
                border_style="green",
                expand=False,
            )
        )


def print_observation(output: dict) -> None:
    raw = output.get("output", "")
    is_error = output.get("error", False)

    if is_error:
        style, title = "red", "[bold red]Error[/bold red]"
        truncated = raw[:1200] + ("..." if len(raw) > 1200 else "")
        console.print(Panel(truncated or "(empty)", title=title, border_style=style, expand=False))
        return

    # Detect search results and render each source compactly
    if raw.startswith("Found ") and "source(s):" in raw[:40]:
        _print_search_observation(raw)
    else:
        truncated = raw[:1200] + ("..." if len(raw) > 1200 else "")
        console.print(
            Panel(
                truncated or "(empty)",
                title="[bold blue]Observation[/bold blue]",
                border_style="blue",
                expand=False,
            )
        )


def _print_search_observation(raw: str) -> None:
    """Parse and display search results compactly, showing all sources."""
    parts = raw.split("\n--- Source Board", 1)
    results_section = parts[0]
    board_section = ("--- Source Board" + parts[1]) if len(parts) > 1 else ""

    source_blocks = re.split(r"\n(?=\[S?\d+\] )", results_section)
    lines: list[str] = []

    for block in source_blocks:
        block = block.strip()
        if not block:
            continue
        if block.startswith("Found "):
            lines.append(f"[bold]{block.strip()}[/bold]")
            continue
        # Extract ID + title line, then show snippet preview
        block_lines = block.split("\n")
        id_line = block_lines[0] if block_lines else block
        snippet = ""
        for bl in block_lines[1:]:
            bl_stripped = bl.strip()
            if bl_stripped.startswith("Snippet:"):
                snippet = bl_stripped[8:].strip()[:150]
                break
        entry = id_line
        if snippet:
            entry += f"\n    {snippet}{'...' if len(snippet) >= 150 else ''}"
        lines.append(entry)

    console.print(
        Panel(
            "\n".join(lines) if lines else "(no sources)",
            title="[bold blue]Search Results[/bold blue]",
            border_style="blue",
            expand=False,
        )
    )

    if board_section.strip():
        console.print(
            Panel(
                board_section.strip(),
                title="[bold cyan]Source Board[/bold cyan]",
                border_style="cyan",
                expand=False,
            )
        )


def print_context_truncation(removed_count: int, window_size: int) -> None:
    console.print(
        f"  [dim]Context truncated: {removed_count} messages removed (window={window_size})[/dim]"
    )
