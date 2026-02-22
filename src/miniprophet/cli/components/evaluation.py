"""Evaluation metrics display component."""

from __future__ import annotations

from rich.table import Table

from miniprophet.cli.utils import get_console

console = get_console()


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
