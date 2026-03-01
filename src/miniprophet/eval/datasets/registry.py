"""Registry dataset schema, loading, and download utilities."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests
from pydantic import BaseModel, Field, field_validator

from miniprophet.eval.datasets.cache import get_registry_cache_path

DEFAULT_REGISTRY_URL = (
    "https://raw.githubusercontent.com/mini-prophet/mini-prophet-datasets/main/registry.json"
)


class RegistryDatasetSpec(BaseModel):
    name: str
    version: str
    description: str = ""
    git_url: str
    git_ref: str
    path: str
    checksum_sha256: str | None = Field(default=None)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("dataset name must be non-empty")
        if "/" in value:
            raise ValueError("registry dataset names cannot contain '/'")
        return value


@dataclass(frozen=True)
class ResolvedRegistryDataset:
    spec: RegistryDatasetSpec
    cached_path: Path


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SEMVER_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def _version_kind_and_key(version: str) -> tuple[int, tuple]:
    if version == "latest":
        return (3, (1,))
    if _DATE_RE.match(version):
        try:
            dt = datetime.strptime(version, "%Y-%m-%d")
            return (2, (dt.year, dt.month, dt.day))
        except ValueError:
            pass
    m = _SEMVER_RE.match(version)
    if m:
        return (1, tuple(int(g) for g in m.groups()))
    return (0, (version,))


def resolve_latest_version(versions: list[str]) -> str:
    if not versions:
        raise ValueError("No versions available")
    if "latest" in versions:
        return "latest"
    return max(versions, key=_version_kind_and_key)


def _parse_registry_payload(payload: str) -> list[RegistryDatasetSpec]:
    data = json.loads(payload)
    if isinstance(data, dict) and "datasets" in data:
        data = data["datasets"]
    if not isinstance(data, list):
        raise ValueError("Registry payload must be a list of dataset entries")
    return [RegistryDatasetSpec.model_validate(item) for item in data]


def load_registry(
    *,
    registry_path: Path | None = None,
    registry_url: str | None = None,
) -> list[RegistryDatasetSpec]:
    if registry_path is not None and registry_url is not None:
        raise ValueError("Cannot specify both registry_path and registry_url")

    if registry_path is not None:
        payload = registry_path.read_text()
    else:
        url = registry_url or DEFAULT_REGISTRY_URL
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        payload = response.text

    return _parse_registry_payload(payload)


def resolve_registry_dataset(
    datasets: list[RegistryDatasetSpec],
    *,
    name: str,
    version: str | None,
) -> RegistryDatasetSpec:
    candidates = [d for d in datasets if d.name == name]
    if not candidates:
        raise ValueError(f"Dataset '{name}' not found in registry")

    if version is None or version == "latest":
        resolved_version = resolve_latest_version([d.version for d in candidates])
    else:
        resolved_version = version

    for dataset in candidates:
        if dataset.version == resolved_version:
            return dataset

    raise ValueError(f"Dataset '{name}@{resolved_version}' not found in registry")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_registry_dataset(
    spec: RegistryDatasetSpec,
    *,
    overwrite_cache: bool = False,
) -> ResolvedRegistryDataset:
    cache_path = get_registry_cache_path(spec.name, spec.version)

    if cache_path.exists() and not overwrite_cache:
        return ResolvedRegistryDataset(spec=spec, cached_path=cache_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = Path(tmp_dir) / "repo"

        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", spec.git_url, str(repo_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

        checkout = subprocess.run(
            ["git", "checkout", spec.git_ref],
            cwd=repo_dir,
            check=False,
            capture_output=True,
            text=True,
        )

        if checkout.returncode != 0:
            subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", spec.git_ref],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "checkout", "FETCH_HEAD"],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True,
            )

        source_path = repo_dir / spec.path
        if not source_path.exists():
            raise FileNotFoundError(
                f"Dataset path '{spec.path}' not found in {spec.git_url}@{spec.git_ref}"
            )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, cache_path)

    if spec.checksum_sha256 is not None:
        actual = _sha256_file(cache_path)
        if actual != spec.checksum_sha256:
            raise ValueError(
                "Checksum mismatch for dataset "
                f"{spec.name}@{spec.version}: expected {spec.checksum_sha256}, got {actual}"
            )

    return ResolvedRegistryDataset(spec=spec, cached_path=cache_path)
