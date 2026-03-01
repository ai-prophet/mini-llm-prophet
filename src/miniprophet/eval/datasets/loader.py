"""Dataset source resolution for prophet eval."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from miniprophet.eval.datasets.hf_loader import download_hf_dataset
from miniprophet.eval.datasets.registry import (
    download_registry_dataset,
    load_registry,
    resolve_registry_dataset,
)


class DatasetSourceKind(StrEnum):
    LOCAL_FILE = "local_file"
    REGISTRY = "registry"
    HF = "hf"


@dataclass(frozen=True)
class ParsedDatasetRef:
    kind: DatasetSourceKind
    raw: str
    name: str | None = None
    version: str | None = None
    hf_repo: str | None = None
    hf_revision: str | None = None


@dataclass(frozen=True)
class ResolvedDataset:
    kind: DatasetSourceKind
    source_ref: str
    path: Path
    name: str
    version: str | None
    metadata: dict


def parse_dataset_ref(dataset_ref: str) -> ParsedDatasetRef:
    ref = dataset_ref.strip()
    if not ref:
        raise ValueError("dataset ref cannot be empty")

    if "/" in ref:
        if "@" in ref:
            repo, revision = ref.split("@", 1)
            revision = revision or None
        else:
            repo, revision = ref, None
        return ParsedDatasetRef(
            kind=DatasetSourceKind.HF,
            raw=ref,
            hf_repo=repo,
            hf_revision=revision,
        )

    if "@" in ref:
        name, version = ref.split("@", 1)
        version = version or None
    else:
        name, version = ref, None

    if "/" in name:
        raise ValueError("registry dataset names cannot contain '/'")

    return ParsedDatasetRef(
        kind=DatasetSourceKind.REGISTRY,
        raw=ref,
        name=name,
        version=version,
    )


def resolve_dataset_to_jsonl(
    dataset_ref: str,
    *,
    registry_path: Path | None = None,
    registry_url: str | None = None,
    hf_split: str = "train",
    overwrite_cache: bool = False,
) -> ResolvedDataset:
    parsed = parse_dataset_ref(dataset_ref)

    if parsed.kind == DatasetSourceKind.HF:
        resolved = download_hf_dataset(
            parsed.raw,
            split=hf_split,
            overwrite_cache=overwrite_cache,
        )
        return ResolvedDataset(
            kind=DatasetSourceKind.HF,
            source_ref=parsed.raw,
            path=resolved.cached_path,
            name=resolved.repo,
            version=resolved.revision,
            metadata={"split": resolved.split},
        )

    if parsed.name is None:
        raise ValueError("registry dataset ref missing name")

    registry = load_registry(registry_path=registry_path, registry_url=registry_url)
    spec = resolve_registry_dataset(registry, name=parsed.name, version=parsed.version)
    resolved = download_registry_dataset(spec, overwrite_cache=overwrite_cache)

    return ResolvedDataset(
        kind=DatasetSourceKind.REGISTRY,
        source_ref=f"{spec.name}@{spec.version}",
        path=resolved.cached_path,
        name=spec.name,
        version=spec.version,
        metadata={
            "git_url": spec.git_url,
            "git_ref": spec.git_ref,
            "registry_path": spec.path,
        },
    )
