"""CLI command for managing global ``.env`` variables."""

from __future__ import annotations

import typer

from miniprophet import global_config_file
from miniprophet.cli.components.env_editor import (
    is_valid_env_key,
    prompt_and_save_env_vars,
    read_env_vars,
    save_env_var,
)
from miniprophet.cli.utils import get_console

app = typer.Typer(
    name="set",
    help="Set global .env variables used by prophet.",
    add_completion=False,
)
console = get_console()


@app.callback(invoke_without_command=True)
def main(
    key: str | None = typer.Argument(None, help="Environment variable key."),
    value: str | None = typer.Argument(None, help="Environment variable value."),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Launch an interactive .env editor.",
    ),
) -> None:
    """Save key/value pairs into the global config .env file."""
    if interactive:
        if key is not None or value is not None:
            console.print("[bold red]Error:[/bold red] Do not pass KEY/VALUE with --interactive.")
            raise typer.Exit(1)

        updates = prompt_and_save_env_vars(global_config_file)
        if updates:
            console.print(
                f"\n[green]Saved {len(updates)} variable(s)[/green] to [cyan]{global_config_file}[/cyan]."
            )
        else:
            console.print("\n[dim]No changes made.[/dim]")
        return

    if key is None or value is None:
        console.print("[bold red]Error:[/bold red] Expected KEY VALUE or use --interactive / -i.")
        raise typer.Exit(1)

    if not is_valid_env_key(key):
        console.print(
            "[bold red]Error:[/bold red] Invalid key. "
            "Use letters, numbers, and underscores; "
            "the first character must be a letter or underscore."
        )
        raise typer.Exit(1)

    existing_value = read_env_vars(global_config_file).get(key)
    save_env_var(global_config_file, key, value)

    if existing_value is None:
        console.print(f"[green]Saved[/green] {key} to [cyan]{global_config_file}[/cyan].")
    else:
        console.print(f"[green]Updated[/green] {key} in [cyan]{global_config_file}[/cyan].")
