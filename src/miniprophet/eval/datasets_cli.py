"""Dataset management CLI for prophet eval."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table
from typer import Argument, Option, Typer

from miniprophet.eval.datasets.loader import resolve_dataset_to_jsonl
from miniprophet.eval.datasets.registry import (
    load_registry,
    resolve_latest_version,
    sort_versions_desc,
)
from miniprophet.eval.datasets.validate import load_problems

datasets_app = Typer(name="datasets", no_args_is_help=True)
console = Console()


def _resolve_dataset_latest(dataset) -> str:
    return dataset.latest or resolve_latest_version([v.version for v in dataset.versions])


def _format_versions_preview(dataset, latest: str) -> str:
    ordered = sort_versions_desc([v.version for v in dataset.versions if v.version != latest])
    lines = [f"{latest} (latest)"]
    lines.extend(ordered[:2])
    if len(dataset.versions) > 3:
        lines.append(f"(...{len(dataset.versions) - 3} more versions)")
    return "\n".join(lines)


def _render_dataset_index(registry) -> None:
    table = Table(title="Available Datasets", show_lines=True)
    table.add_column("Name", style="cyan")
    table.add_column("Latest", style="magenta")
    table.add_column("Versions", style="green")
    table.add_column("Description", style="white")

    for dataset in sorted(registry.datasets, key=lambda d: d.name):
        latest = _resolve_dataset_latest(dataset)
        versions_preview = _format_versions_preview(dataset, latest)
        table.add_row(dataset.name, latest, versions_preview, dataset.description or "")

    console.print(table)
    console.print(f"\n[green]Total datasets:[/green] {len(registry.datasets)}")


def _render_dataset_versions(registry, dataset_name: str) -> None:
    dataset = next((d for d in registry.datasets if d.name == dataset_name), None)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in registry")

    latest = _resolve_dataset_latest(dataset)
    ordered = sort_versions_desc([v.version for v in dataset.versions])

    table = Table(title=f"Dataset Versions: {dataset.name}", show_lines=True)
    table.add_column("Version", style="magenta")
    table.add_column("Tag", style="green")

    for version in ordered:
        table.add_row(version, "latest" if version == latest else "")

    console.print(table)
    if dataset.description:
        console.print(f"[cyan]Description:[/cyan] {dataset.description}")
    console.print(f"[cyan]Latest:[/cyan] {latest}")
    console.print(f"[cyan]Total versions:[/cyan] {len(dataset.versions)}")


@datasets_app.command("list")
def list_datasets(
    dataset_name: str | None = Argument(
        None,
        help="Optional dataset name to display all available versions.",
    ),
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
        registry = load_registry(registry_path=registry_path, registry_url=registry_url)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise

    if not registry.datasets:
        console.print("[yellow]No datasets found.[/yellow]")
        return

    if dataset_name is None:
        _render_dataset_index(registry)
    else:
        _render_dataset_versions(registry, dataset_name=dataset_name)


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
