"""Forecast results bar-chart display component."""

from __future__ import annotations

from miniprophet.cli.utils import get_console

console = get_console()


def print_forecast_results(submission: dict[str, float]) -> None:
    """Render the submitted forecast as a bar chart."""
    if not submission:
        return
    console.print("\n[bold]Forecast Results:[/bold]")
    for outcome_name, prob in submission.items():
        bar = "\u2588" * int(prob * 30)
        console.print(f"  {outcome_name:30s} {prob:.4f}  {bar}")
