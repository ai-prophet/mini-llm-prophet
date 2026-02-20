"""CLI entry point for mini-llm-prophet."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Prompt

from miniprophet import __version__
from miniprophet.utils.log import logger  # noqa: F401 — triggers root logger setup
from miniprophet.utils.serialize import UNSET, recursive_merge

app = typer.Typer(
    name="prophet",
    help="mini-llm-prophet: a minimal LLM forecasting agent.",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    title: str | None = typer.Option(None, "--title", "-t", help="The forecasting question."),
    outcomes: str | None = typer.Option(
        None, "--outcomes", "-o", help="Comma-separated list of possible outcomes."
    ),
    ground_truth_json: str | None = typer.Option(
        None,
        "--ground-truth",
        "-g",
        help='Ground truth as JSON, e.g. \'{"Yes": 1, "No": 0}\'.',
    ),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Enter interactive mode."),
    model_name: str | None = typer.Option(
        None, "--model", "-m", help="Model name (e.g. openai/gpt-4o-mini)."
    ),
    cost_limit: float | None = typer.Option(
        None, "--cost-limit", "-l", help="Total cost limit in USD."
    ),
    search_limit: int | None = typer.Option(
        None, "--search-limit", help="Max number of search queries."
    ),
    step_limit: int | None = typer.Option(None, "--step-limit", help="Max number of agent steps."),
    config_spec: list[str] | None = typer.Option(
        None, "--config", "-c", help="Config file(s) or key=value overrides."
    ),
    output: Path | None = typer.Option(None, "--output", help="Path to save the trajectory JSON."),
    model_class: str | None = typer.Option(
        None, "--model-class", help="Override model class (e.g. openrouter)."
    ),
) -> None:
    """Run the forecasting agent on a question with specified outcomes."""
    from miniprophet.agent.context import SlidingWindowContextManager
    from miniprophet.agent.default import DefaultForecastAgent
    from miniprophet.config import get_config_from_spec
    from miniprophet.environment.forecast_env import ForecastEnvironment
    from miniprophet.models import get_model
    from miniprophet.search import get_search_tool

    console.print(f"[bold green]mini-llm-prophet[/bold green] v{__version__}\n")

    # ---- Resolve title, outcomes, ground_truth ----
    ground_truth: dict[str, int] | None = None
    if ground_truth_json:
        try:
            ground_truth = json.loads(ground_truth_json)
        except json.JSONDecodeError as exc:
            console.print(f"[bold red]Error:[/bold red] Invalid --ground-truth JSON: {exc}")
            raise typer.Exit(1)

    if interactive:
        resolved_title, outcome_list, ground_truth = _interactive_flow(
            prefill_title=title or "",
            prefill_outcomes=outcomes,
            prefill_ground_truth=ground_truth,
        )
    else:
        if not title or not outcomes:
            console.print(
                "[bold red]Error:[/bold red] --title and --outcomes are required "
                "(or use --interactive / -i)."
            )
            raise typer.Exit(1)
        resolved_title = title
        outcome_list = [o.strip() for o in outcomes.split(",") if o.strip()]

    if len(outcome_list) < 2:
        console.print("[bold red]Error:[/bold red] At least 2 outcomes are required.")
        raise typer.Exit(1)

    # ---- Load and merge configs ----
    configs = [get_config_from_spec("default")]
    for spec in config_spec or []:
        configs.append(get_config_from_spec(spec))

    configs.append(
        {
            "agent": {
                "cost_limit": cost_limit or UNSET,
                "search_limit": search_limit or UNSET,
                "step_limit": step_limit or UNSET,
                "output_path": str(output) if output else UNSET,
            },
            "model": {
                "model_name": model_name or UNSET,
                "model_class": model_class or UNSET,
            },
        }
    )

    config = recursive_merge(*configs)

    # ---- Instantiate components ----
    model = get_model(config=config.get("model", {}))
    search_tool = get_search_tool(config=config.get("search", {}))

    env_kwargs = config.get("environment", {})
    agent_search_limit = config.get("agent", {}).get("search_limit", 10)
    env = ForecastEnvironment(
        search_tool=search_tool,
        outcomes=outcome_list,
        search_limit=agent_search_limit,
        **env_kwargs,
    )

    context_window = config.get("agent", {}).get("context_window", 6)
    ctx_mgr = (
        SlidingWindowContextManager(window_size=context_window) if context_window > 0 else None
    )
    agent = DefaultForecastAgent(
        model=model, env=env, context_manager=ctx_mgr, **config.get("agent", {})
    )

    # ---- Run ----
    result = agent.run(title=resolved_title, outcomes=outcome_list, ground_truth=ground_truth)

    # ---- Print results ----
    submission = result.get("submission", {})
    if submission:
        console.print("\n[bold]Forecast Results:[/bold]")
        for outcome_name, prob in submission.items():
            bar = "█" * int(prob * 30)
            console.print(f"  {outcome_name:30s} {prob:.4f}  {bar}")
    else:
        console.print(
            f"\n[bold yellow]Agent exited without submitting.[/bold yellow] "
            f"Status: {result.get('exit_status', 'unknown')}"
        )


def _interactive_flow(
    prefill_title: str,
    prefill_outcomes: str | None,
    prefill_ground_truth: dict[str, int] | None,
) -> tuple[str, list[str], dict[str, int] | None]:
    """Run the interactive TUI flow: manual input or Kalshi ticker."""
    from miniprophet.run.tui import prompt_forecast_params

    console.print()
    choice = Prompt.ask(
        "  [bold]How would you like to set up the forecast?[/bold]\n"
        "  [cyan]1[/cyan] Manual input\n"
        "  [cyan]2[/cyan] Kalshi market ticker\n"
        "  Choose",
        choices=["1", "2"],
        default="1",
    )

    prefill_outcomes_list = (
        [o.strip() for o in prefill_outcomes.split(",") if o.strip()] if prefill_outcomes else []
    )

    if choice == "2":
        prefill_title, prefill_outcomes_list, prefill_ground_truth = _fetch_kalshi()

    return prompt_forecast_params(
        prefill_title=prefill_title,
        prefill_outcomes=prefill_outcomes_list,
        prefill_ground_truth=prefill_ground_truth,
    )


def _fetch_kalshi() -> tuple[str, list[str], dict[str, int] | None]:
    """Prompt for a Kalshi ticker and fetch market data."""
    from miniprophet.run.services import get_market_service

    ticker = Prompt.ask("  [bold]Kalshi ticker[/bold]").strip()
    if not ticker:
        console.print("  [red]No ticker provided.[/red]")
        return "", [], None

    try:
        service = get_market_service("kalshi")
        data = service.fetch(ticker)
    except Exception as exc:
        console.print(f"  [bold red]Failed to fetch market:[/bold red] {exc}")
        return "", [], None

    console.print(f"  [green]Fetched:[/green] {data.title}")
    if data.metadata.get("last_price"):
        console.print(
            f"  [dim]Last price: {data.metadata['last_price']}  Volume: {data.metadata.get('volume', '?')}[/dim]"
        )

    return data.title, data.outcomes, data.ground_truth


if __name__ == "__main__":
    app()
