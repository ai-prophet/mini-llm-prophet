"""Dataset cache paths under global mini-prophet config."""

from __future__ import annotations

from pathlib import Path

from miniprophet import global_config_dir

DATASET_CACHE_DIRNAME = "datasets"


def get_dataset_cache_root() -> Path:
    root = Path(global_config_dir) / DATASET_CACHE_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_registry_cache_path(name: str, version: str) -> Path:
    path = get_dataset_cache_root() / "registry" / name / version / "dataset.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_hf_cache_path(repo: str, revision: str | None, split: str) -> Path:
    safe_repo = repo.replace("/", "__")
    safe_revision = (revision or "default").replace("/", "__")
    safe_split = split.replace("/", "__")
    path = get_dataset_cache_root() / "hf" / safe_repo / safe_revision / f"{safe_split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
