"""Hugging Face dataset loading for prophet eval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from miniprophet.eval.datasets.cache import get_hf_cache_path


@dataclass(frozen=True)
class ResolvedHFDataset:
    repo: str
    revision: str | None
    split: str
    cached_path: Path


def parse_hf_ref(dataset_ref: str) -> tuple[str, str | None]:
    if "/" not in dataset_ref:
        raise ValueError(
            "Hugging Face dataset refs must be in 'username/dataset[@revision]' format"
        )

    if "@" in dataset_ref:
        repo, revision = dataset_ref.split("@", 1)
        revision = revision or None
    else:
        repo, revision = dataset_ref, None

    if not repo or repo.startswith("/") or repo.endswith("/"):
        raise ValueError(f"Invalid Hugging Face dataset repo: '{repo}'")

    return repo, revision


def download_hf_dataset(
    dataset_ref: str,
    *,
    split: str = "train",
    overwrite_cache: bool = False,
) -> ResolvedHFDataset:
    repo, revision = parse_hf_ref(dataset_ref)
    cache_path = get_hf_cache_path(repo, revision, split)

    if cache_path.exists() and not overwrite_cache:
        return ResolvedHFDataset(repo=repo, revision=revision, split=split, cached_path=cache_path)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face dataset loading requires the 'datasets' package. "
            "Install with: pip install -e '.[hf]'"
        ) from exc

    ds = load_dataset(repo, revision=revision, split=split)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with cache_path.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")

    return ResolvedHFDataset(repo=repo, revision=revision, split=split, cached_path=cache_path)
