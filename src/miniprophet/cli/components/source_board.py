"""Source board display component with colored reactions."""

from __future__ import annotations

from rich.panel import Panel

from miniprophet.cli.utils import get_console

console = get_console()

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
        parts.append(f"{outcome} {symbol}")
    return "  ".join(parts)


def print_board_state(board) -> None:
    """Render the current source board state from a SourceBoard instance."""
    from miniprophet.environment.source_board import SourceBoard

    if not isinstance(board, SourceBoard) or len(board) == 0:
        return
    console.print(
        Panel(
            board.render(),
            title="[bold cyan]Source Board[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
    )
