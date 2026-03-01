"""Dataset management CLI for prophet eval."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table
from typer import Argument, Option, Typer

from miniprophet.eval.datasets.loader import resolve_dataset_to_jsonl
from miniprophet.eval.datasets.registry import load_registry
from miniprophet.eval.datasets.validate import load_problems

datasets_app = Typer(name="datasets", no_args_is_help=True)
console = Console()


@datasets_app.command("list")
def list_datasets(
    registry_path: Path | None = Option(
        None,
        "--registry-path",
        help="Path to local registry.json (testing/dev)",
    ),
    registry_url: str | None = Option(
        None,
        "--registry-url",
        help="Registry URL override",
    ),
) -> None:
    """List datasets available in the registry."""
    try:
        datasets = load_registry(registry_path=registry_path, registry_url=registry_url)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise

    if not datasets:
        console.print("[yellow]No datasets found.[/yellow]")
        return

    table = Table(title="Available Datasets", show_lines=True)
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="magenta")
    table.add_column("Description", style="white")

    for item in sorted(datasets, key=lambda d: (d.name, d.version)):
        table.add_row(item.name, item.version, item.description or "")

    console.print(table)
    console.print(f"\n[green]Total datasets:[/green] {len(datasets)}")


@datasets_app.command("download")
def download_dataset(
    dataset: str = Argument(
        ..., help="Dataset ref: name[@version|@latest] or username/dataset[@revision]"
    ),
    registry_path: Path | None = Option(None, "--registry-path", help="Local registry path"),
    registry_url: str | None = Option(None, "--registry-url", help="Registry URL override"),
    hf_split: str = Option("train", "--hf-split", help="HF split when using username/dataset refs"),
    overwrite_cache: bool = Option(False, "--overwrite-cache", help="Refresh cached artifact"),
) -> None:
    """Download/resolve a dataset to local cache and print its path."""
    try:
        resolved = resolve_dataset_to_jsonl(
            dataset,
            registry_path=registry_path,
            registry_url=registry_url,
            hf_split=hf_split,
            overwrite_cache=overwrite_cache,
        )
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise

    console.print(f"[green]Resolved:[/green] {resolved.source_ref}")
    console.print(f"[green]Source:[/green] {resolved.kind.value}")
    console.print(f"[green]Cached path:[/green] {resolved.path}")


@datasets_app.command("validate")
def validate_dataset(
    dataset: str | None = Option(
        None,
        "--dataset",
        "-d",
        help="Dataset ref: name[@version|@latest] or username/dataset[@revision]",
    ),
    input_file: Path | None = Option(None, "--input", "-f", help="Local JSONL file path"),
    registry_path: Path | None = Option(None, "--registry-path", help="Local registry path"),
    registry_url: str | None = Option(None, "--registry-url", help="Registry URL override"),
    hf_split: str = Option("train", "--hf-split", help="HF split for username/dataset refs"),
) -> None:
    """Validate dataset rows against forecast task schema."""
    if (dataset is None) == (input_file is None):
        raise ValueError("Provide exactly one of --dataset/-d or --input/-f")

    if input_file is not None:
        path = input_file
    else:
        assert dataset is not None
        resolved = resolve_dataset_to_jsonl(
            dataset,
            registry_path=registry_path,
            registry_url=registry_url,
            hf_split=hf_split,
        )
        path = resolved.path

    problems = load_problems(path)
    console.print(f"[green]Validation successful.[/green] {len(problems)} problem(s) in {path}")
