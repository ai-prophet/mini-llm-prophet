"""Root CLI application for mini-prophet.

Combines single-run, eval, datasets, and set commands under one entry point.
"""

from __future__ import annotations

import typer

from miniprophet import __version__

app = typer.Typer(
    name="prophet",
    help=f"mini-prophet (v{__version__}): a minimal LLM forecasting agent.",
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=True,
)


@app.callback()
def _callback() -> None:
    pass


def _register_subcommands() -> None:
    from miniprophet.eval.cli import app as eval_app
    from miniprophet.eval.datasets_cli import datasets_app
    from miniprophet.run.cli import app as cli_app
    from miniprophet.run.set import app as set_app

    app.add_typer(cli_app, name="run", help="Run a single forecast (interactive or CLI args).")
    app.add_typer(eval_app, name="eval", help="Run forecast evaluation from JSONL or datasets.")
    app.add_typer(datasets_app, name="datasets", help="List/download/validate datasets.")
    app.add_typer(set_app, name="set", help="Set global .env variables.")


_register_subcommands()
