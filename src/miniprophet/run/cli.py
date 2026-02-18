"""CLI entry point for mini-llm-prophet."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from miniprophet import __version__
from miniprophet.utils.serialize import UNSET, recursive_merge

app = typer.Typer(
    name="prophet",
    help="mini-llm-prophet: a minimal LLM forecasting agent.",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    title: str = typer.Option(..., "--title", "-t", help="The forecasting question."),
    outcomes: str = typer.Option(
        ..., "--outcomes", "-o", help="Comma-separated list of possible outcomes."
    ),
    model_name: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model name (e.g. openai/gpt-4o-mini)."
    ),
    cost_limit: Optional[float] = typer.Option(
        None, "--cost-limit", "-l", help="Total cost limit in USD."
    ),
    search_limit: Optional[int] = typer.Option(
        None, "--search-limit", help="Max number of search queries."
    ),
    step_limit: Optional[int] = typer.Option(
        None, "--step-limit", help="Max number of agent steps."
    ),
    config_spec: Optional[list[str]] = typer.Option(
        None, "--config", "-c", help="Config file(s) or key=value overrides."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", help="Path to save the trajectory JSON."
    ),
    model_class: Optional[str] = typer.Option(
        None, "--model-class", help="Override model class (e.g. openrouter)."
    ),
) -> None:
    """Run the forecasting agent on a question with specified outcomes."""
    from miniprophet.agent.default import DefaultForecastAgent
    from miniprophet.config import get_config_from_spec
    from miniprophet.environment.forecast_env import ForecastEnvironment
    from miniprophet.models import get_model
    from miniprophet.search import get_search_tool

    console.print(
        f"[bold green]mini-llm-prophet[/bold green] v{__version__}\n"
    )

    # ---- Load and merge configs ----
    configs = [get_config_from_spec("default")]
    for spec in config_spec or []:
        configs.append(get_config_from_spec(spec))

    configs.append({
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
    })

    config = recursive_merge(*configs)

    # ---- Parse outcomes ----
    outcome_list = [o.strip() for o in outcomes.split(",") if o.strip()]
    if len(outcome_list) < 2:
        console.print("[bold red]Error:[/bold red] At least 2 outcomes are required.")
        raise typer.Exit(1)

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

    agent = DefaultForecastAgent(model=model, env=env, **config.get("agent", {}))

    # ---- Run ----
    result = agent.run(title=title, outcomes=outcome_list)

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


if __name__ == "__main__":
    app()
