"""Source board display component with grouped source cards."""

from __future__ import annotations

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from miniprophet.cli.utils import get_console

console = get_console()

REACTION_STYLES: dict[str, tuple[str, str]] = {
    "very_positive": ("bold green", "++"),
    "positive": ("green", "+"),
    "neutral": ("dim", "~"),
    "negative": ("red", "-"),
    "very_negative": ("bold red", "--"),
}


def format_reaction(reaction: dict[str, str]) -> Text:
    """Render reaction sentiment with explicit positive/negative colors."""
    if not reaction:
        return Text("")

    line = Text("Reactions: ", style="bold white")
    for idx, (outcome, sentiment) in enumerate(reaction.items()):
        if idx > 0:
            line.append("  ")
        line.append(f"{outcome} ", style="white")
        style, symbol = REACTION_STYLES.get(sentiment, ("yellow", "?"))
        line.append(symbol, style=style)
    return line


def _render_source_board_panel(board) -> Panel:
    """Render board entries using nested group boxes, similar to search results."""
    entries = board.serialize()
    outer_border = Style(color="cyan", bold=True)
    cards = []

    for entry in entries:
        source = entry["source"]
        title = (source.get("title", "") or "").strip() or "(untitled)"
        url = (source.get("url", "") or "").strip()
        snippet = source.get("snippet", "") or ""
        note = (entry.get("note", "") or "").strip() or "(no note)"
        reaction = entry.get("reaction", {}) or {}

        first_line = Text()
        first_line.append(f"[#{entry['id']}] ", style="magenta bold")
        first_line.append(title, style="bold white")

        url_text = Text(url, style="cyan underline")
        url_text.no_wrap = False

        snippet_text = Text(snippet[:200], style="white")
        extra_chars = len(snippet) - 200
        if extra_chars > 0:
            snippet_text += Text(f" ...{extra_chars} characters omitted", style="dim italic")
        snippet_text.no_wrap = False

        note_text = Text(f"> {note[:400]}", style="italic")
        extra_note_chars = len(note) - 400
        if extra_note_chars > 0:
            note_text += Text(f"\n...{extra_note_chars} characters omitted", style="dim italic")

        quote_panel = Panel(
            note_text,
            title="Note",
            border_style="bright_black",
            box=box.SQUARE,
            padding=(0, 1),
        )

        body_items = [first_line, url_text, snippet_text]
        if reaction:
            body_items.append(format_reaction(reaction))
        body_items.append(quote_panel)

        cards.append(
            Panel(
                Group(*body_items),
                box=box.ROUNDED,
                border_style="bright_black",
                padding=(0, 1),
            )
        )

    subtitle = Text(f"{len(entries)} sources", style="dim")
    content = Group(*cards) if cards else Text("Board is empty.", style="dim")
    return Panel(
        content,
        title="Source Board",
        subtitle=subtitle,
        border_style=outer_border,
        box=box.ROUNDED,
        padding=(0, 1),
        expand=False,
    )


def print_board_state(board) -> None:
    """Render the current source board state from a SourceBoard instance."""
    from miniprophet.environment.source_board import SourceBoard

    if not isinstance(board, SourceBoard) or len(board) == 0:
        return
    console.print(_render_source_board_panel(board))
