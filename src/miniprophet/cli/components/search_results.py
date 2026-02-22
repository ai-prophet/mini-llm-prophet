"""Search results display component."""

from __future__ import annotations

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

from miniprophet.cli.utils import get_console
from miniprophet.environment.source_board import Source

console = get_console()


def _render_search_results_panel(
    search_results: list[tuple[str, Source]],
    *,
    title: str = "Search results",
    max_source_display_chars: int = 200,
) -> Panel:
    """
    results: iterable of objects/dicts with fields:
      s.id, s.url, s.title, s.snippet
    """

    # --- outer frame style: bold blue border ---
    outer_border = Style(color="blue", bold=True)

    # A grid table works nicely as "list layout" with consistent spacing.
    grid = Table.grid(padding=(0, 1))
    grid.expand = True
    grid.add_column(justify="right", width=4)  # rank / index
    grid.add_column(ratio=1, overflow="fold")  # content

    cards = []

    for sid, s in search_results:
        url = s.url
        stitle = s.title
        snippet = s.snippet

        # Title line (primary)
        title_text = Text(stitle.strip() or "(untitled)", style="bold white")

        # URL line (secondary; make it look link-like)
        url_text = Text(url.strip(), style="cyan underline")
        url_text.no_wrap = False

        # Snippet line (tertiary; dim so it doesn't dominate)
        snippet_text = Text(snippet[:max_source_display_chars], style="white")
        snippet_text.no_wrap = False

        # Small "meta" prefix (id)
        meta = Text()
        if sid is not None and str(sid).strip():
            meta.append(f"[ID: {sid}]", style="magenta bold")
            meta.append("  ")

        # Build the per-result content group
        body = Group(
            Group(meta, title_text),
            url_text,
            snippet_text,
        )

        # Put each result in its own subtle inner panel (helps scanning)
        card = Panel(
            body,
            box=box.ROUNDED,
            border_style="bright_black",
            padding=(0, 1),
        )
        cards.append(card)

    content = Group(*cards) if cards else Text("No results.", style="dim")

    # Optionally include a subtitle with count
    subtitle = Text(f"{len(search_results)} sources", style="dim")

    return Panel(
        content,
        title=title,
        subtitle=subtitle,
        border_style=outer_border,
        box=box.ROUNDED,
        padding=(0, 1),
        expand=False,
    )


def print_search_observation(
    search_results: list[tuple[str, Source]], max_source_display_chars: int = 200
) -> None:
    """Parse and display search results compactly, showing all sources."""
    panel = _render_search_results_panel(
        search_results, max_source_display_chars=max_source_display_chars
    )
    console.print(panel)
