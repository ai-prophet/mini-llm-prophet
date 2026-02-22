"""Run header and footer display components."""

from __future__ import annotations

from rich.console import Console

console = Console()


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
