"""Batch CLI entry point for mini-llm-prophet."""

from __future__ import annotations

from pathlib import Path

import typer

from miniprophet import __version__
from miniprophet.cli.utils import get_console
from miniprophet.utils.serialize import UNSET, recursive_merge

app = typer.Typer(
    name="batch",
    help="Run batch forecasting from a JSONL file.",
    add_completion=False,
)
console = get_console()


@app.callback(invoke_without_command=True)
def main(
    input_file: Path = typer.Option(
        ..., "--input", "-f", help="Path to the .jsonl input file with forecasting problems."
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Output meta-directory for all run artifacts."
    ),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel workers."),
    max_cost: float = typer.Option(
        0.0, "--max-cost", help="Total cost budget across all runs (0 = unlimited)."
    ),
    max_cost_per_run: float | None = typer.Option(
        None, "--max-cost-per-run", help="Per-run cost limit (overrides agent config)."
    ),
    model_name: str | None = typer.Option(None, "--model", "-m", help="Model name override."),
    model_class: str | None = typer.Option(
        None, "--model-class", help="Model class override (e.g. openrouter, litellm)."
    ),
    config_spec: list[str] | None = typer.Option(
        None, "--config", "-c", help="Config file(s) or key=value overrides."
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from existing output summary: skip already-seen run_ids.",
    ),
    offset: int = typer.Option(
        0, "--offset", help="Offset how many days before the end time for search upper bound."
    ),
) -> None:
    """Run the forecasting agent on a batch of problems from a JSONL file."""
    from miniprophet.config import get_config_from_spec
    from miniprophet.run.batch_runner import load_existing_summary, load_problems, run_batch

    console.print(
        f"[bold green]mini-llm-prophet[/bold green] v{__version__} [dim]batch mode[/dim]\n"
    )

    if not input_file.exists():
        console.print(f"[bold red]Error:[/bold red] Input file not found: {input_file}")
        raise typer.Exit(1)

    try:
        problems = load_problems(input_file, offset=offset)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1)

    if not problems:
        console.print("[bold yellow]No problems found in the input file.[/bold yellow]")
        raise typer.Exit(0)

    console.print(f"  Loaded [cyan]{len(problems)}[/cyan] problem(s) from {input_file}")
    console.print(f"  Workers: [cyan]{workers}[/cyan]  Output: [cyan]{output}[/cyan]\n")

    configs = [get_config_from_spec("default")]
    for spec in config_spec or []:
        configs.append(get_config_from_spec(spec))

    configs.append(
        {
            "agent": {
                "cost_limit": max_cost_per_run or UNSET,
                "output_path": UNSET,
            },
            "model": {
                "model_name": model_name or UNSET,
                "model_class": model_class or UNSET,
            },
        }
    )

    config = recursive_merge(*configs)

    resume_results = {}
    resume_total_cost = 0.0
    if resume:
        summary_path = output / "summary.json"
        if summary_path.exists():
            try:
                resume_results, resume_total_cost = load_existing_summary(summary_path)
            except ValueError as exc:
                console.print(f"[bold red]Error:[/bold red] {exc}")
                raise typer.Exit(1)

            input_run_ids = {p.run_id for p in problems}
            unexpected = sorted(set(resume_results) - input_run_ids)
            if unexpected:
                preview = ", ".join(unexpected[:10])
                suffix = " ..." if len(unexpected) > 10 else ""
                console.print(
                    "[bold red]Error:[/bold red] Resume summary contains run_ids not present "
                    f"in the input file: {preview}{suffix}"
                )
                raise typer.Exit(1)

            original_count = len(problems)
            problems = [p for p in problems if p.run_id not in resume_results]
            skipped = original_count - len(problems)
            console.print(
                f"  Resume mode: skipping [cyan]{skipped}[/cyan] existing run(s), "
                f"[cyan]{len(problems)}[/cyan] remaining."
            )
        else:
            console.print("  Resume mode: no existing summary.json found, starting fresh.")

    if not problems:
        console.print("[bold yellow]No remaining runs to process.[/bold yellow]")
        raise typer.Exit(0)

    batch_cfg = config.get("batch", {})
    timeout_seconds = max(0.0, float(batch_cfg.get("timeout", 180.0)))

    results = run_batch(
        problems,
        output,
        config,
        workers=workers,
        max_cost=max_cost,
        timeout_seconds=timeout_seconds,
        initial_results=resume_results,
        initial_total_cost=resume_total_cost,
    )

    n_submitted = sum(1 for r in results.values() if r.status == "submitted")
    n_failed = sum(1 for r in results.values() if r.status not in ("submitted", "pending"))
    console.print(
        f"\n[bold]Batch complete:[/bold] "
        f"[green]{n_submitted}[/green] submitted, "
        f"[red]{n_failed}[/red] failed, "
        f"[cyan]{len(results)}[/cyan] total"
    )
    console.print(f"  Summary: {output / 'summary.json'}")
