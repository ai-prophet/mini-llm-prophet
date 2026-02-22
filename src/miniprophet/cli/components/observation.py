"""Generic observation display component."""

from __future__ import annotations

from rich.panel import Panel

from miniprophet.cli.utils import get_console

console = get_console()


def print_observation(output: dict) -> None:
    """Render a generic (non-search) tool observation."""
    raw = output.get("output", "")
    is_error = output.get("error", False)

    if is_error:
        truncated = raw[:1200] + ("..." if len(raw) > 1200 else "")
        console.print(
            Panel(
                truncated or "(empty)",
                title="[bold red]Error[/bold red]",
                border_style="red",
                expand=False,
            )
        )
        return

    truncated = raw[:1200] + ("..." if len(raw) > 1200 else "")
    console.print(
        Panel(
            truncated or "(empty)",
            title="[bold blue]Observation[/bold blue]",
            border_style="blue",
            expand=False,
        )
    )
