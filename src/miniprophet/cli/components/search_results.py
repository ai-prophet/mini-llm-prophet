"""Search results display component."""

from __future__ import annotations

import re

from rich.panel import Panel

from miniprophet.cli.utils import get_console

console = get_console()


def print_search_observation(raw: str) -> None:
    """Parse and display search results compactly, showing all sources."""
    source_blocks = re.split(r"\n(?=\[S?\d+\] )", raw)
    lines: list[str] = []

    for block in source_blocks:
        block = block.strip()
        if not block:
            continue
        if block.startswith("Found "):
            lines.append(f"[bold]{block.strip()}[/bold]")
            continue
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
