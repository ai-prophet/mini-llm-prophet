"""Interactive editor helpers for the global ``.env`` file."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import dotenv_values, set_key
from rich.prompt import Prompt
from rich.table import Table

from miniprophet.cli.utils import get_console

console = get_console()

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_valid_env_key(key: str) -> bool:
    """Return whether a key is a valid env variable name."""
    return bool(_ENV_KEY_RE.match(key))


def read_env_vars(env_file: Path) -> dict[str, str]:
    """Load key/value pairs from an env file."""
    env_map: dict[str, str] = {}
    for key, value in dotenv_values(env_file).items():
        if key is None:
            continue
        env_map[key] = value or ""
    return env_map


def save_env_var(env_file: Path, key: str, value: str) -> None:
    """Persist a key/value pair to an env file and current process."""
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.touch(exist_ok=True)
    set_key(str(env_file), key, value, quote_mode="auto")
    os.environ[key] = value


def _print_env_table(env_file: Path) -> None:
    env_map = read_env_vars(env_file)
    console.print()

    table = Table(title="Current .env variables", show_header=True, header_style="bold cyan")
    table.add_column("Key", style="green")
    table.add_column("Value", style="white")

    if env_map:
        for key, value in env_map.items():
            table.add_row(key, value)
    else:
        table.add_row("[dim](none)[/dim]", "[dim](empty file)[/dim]")

    console.print(table)


def prompt_and_save_env_vars(env_file: Path) -> dict[str, str]:
    """Interactive flow for adding/updating env variables."""
    updates: dict[str, str] = {}

    console.rule("[bold cyan]Global Env Config[/bold cyan]", style="cyan")
    console.print(f"[bold]Config file:[/bold] [cyan]{env_file}[/cyan]")
    console.print(
        "[dim]Press Enter on the key prompt when you are done editing.[/dim]"
    )

    while True:
        _print_env_table(env_file)

        key = Prompt.ask("  [bold]Env key[/bold]", default="").strip()
        if not key:
            break

        if not is_valid_env_key(key):
            console.print(
                "  [red]Invalid key.[/red] Use letters, numbers, and underscores; "
                "the first character must be a letter or underscore."
            )
            continue

        current_value = read_env_vars(env_file).get(key, "")
        value = Prompt.ask(f"  [bold]Value for {key}[/bold]", default=current_value).strip()
        save_env_var(env_file, key, value)
        updates[key] = value
        console.print(f"  [green]Saved:[/green] {key}")

    return updates
