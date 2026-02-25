"""Root CLI application for mini-llm-prophet.

Combines the single-run CLI (``prophet run``) and batch CLI
(``prophet batch``) under a single entry point.
"""

from __future__ import annotations

import typer

from miniprophet import __version__

app = typer.Typer(
    name="prophet",
    help=f"mini-llm-prophet (v{__version__}): a minimal LLM forecasting agent.",
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=True,
)


@app.callback()
def _callback() -> None:
    pass


def _register_subcommands() -> None:
    from miniprophet.run.batch import app as batch_app
    from miniprophet.run.cli import app as cli_app
    from miniprophet.run.set import app as set_app

    app.add_typer(cli_app, name="run", help="Run a single forecast (interactive or CLI args).")
    app.add_typer(batch_app, name="batch", help="Run batch forecasting from a JSONL file.")
    app.add_typer(set_app, name="set", help="Set global .env variables.")


_register_subcommands()
