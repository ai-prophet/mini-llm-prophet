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
from pydantic import BaseModel, Field, field_validator, model_validator

from miniprophet.eval.datasets.cache import get_registry_cache_path

DEFAULT_REGISTRY_URL = (
    "https://raw.githubusercontent.com/ai-prophet/ai-prophet-datasets/main/registry.json"
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


class RegistryDatasetVersionSpec(BaseModel):
    version: str
    git_url: str
    git_ref: str
    path: str
    checksum_sha256: str | None = Field(default=None)
    description: str = ""


class RegistryDataset(BaseModel):
    name: str
    description: str = ""
    latest: str | None = None
    versions: list[RegistryDatasetVersionSpec]

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("dataset name must be non-empty")
        if "/" in value:
            raise ValueError("registry dataset names cannot contain '/'")
        return value

    @model_validator(mode="after")
    def _validate_versions(self) -> RegistryDataset:
        if not self.versions:
            raise ValueError(f"Dataset '{self.name}' must have at least one version entry")

        seen: set[str] = set()
        for version in self.versions:
            if version.version in seen:
                raise ValueError(
                    f"Dataset '{self.name}' has duplicate version '{version.version}' entries"
                )
            seen.add(version.version)

        if self.latest is not None and self.latest not in seen:
            raise ValueError(
                f"Dataset '{self.name}' latest='{self.latest}' is not present in versions"
            )

        return self


@dataclass(frozen=True)
class RegistryCatalog:
    datasets: list[RegistryDataset]


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


def sort_versions_desc(versions: list[str]) -> list[str]:
    """Sort version labels from most-recent/highest to oldest/lowest."""
    return sorted(versions, key=_version_kind_and_key, reverse=True)


def _parse_registry_payload(payload: str) -> RegistryCatalog:
    data = json.loads(payload)
    if isinstance(data, dict) and "datasets" in data:
        data = data["datasets"]
    if not isinstance(data, list):
        raise ValueError("Registry payload must be a list of dataset entries/groups")

    if not data:
        return RegistryCatalog(datasets=[])

    if isinstance(data[0], dict) and "versions" in data[0]:
        datasets = [RegistryDataset.model_validate(item) for item in data]
        return RegistryCatalog(datasets=datasets)

    specs = [RegistryDatasetSpec.model_validate(item) for item in data]
    grouped: dict[str, dict] = {}
    for spec in specs:
        dataset = grouped.setdefault(
            spec.name,
            {
                "name": spec.name,
                "description": spec.description or "",
                "latest": None,
                "versions": [],
            },
        )
        if not dataset["description"] and spec.description:
            dataset["description"] = spec.description

        dataset["versions"].append(
            {
                "version": spec.version,
                "git_url": spec.git_url,
                "git_ref": spec.git_ref,
                "path": spec.path,
                "checksum_sha256": spec.checksum_sha256,
                "description": spec.description,
            }
        )

        if spec.version == "latest":
            dataset["latest"] = "latest"

    return RegistryCatalog(
        datasets=[RegistryDataset.model_validate(item) for item in grouped.values()]
    )


def load_registry(
    *,
    registry_path: Path | None = None,
    registry_url: str | None = None,
) -> RegistryCatalog:
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
    registry: RegistryCatalog,
    *,
    name: str,
    version: str | None,
) -> RegistryDatasetSpec:
    dataset = next((d for d in registry.datasets if d.name == name), None)
    if dataset is None:
        raise ValueError(f"Dataset '{name}' not found in registry")

    versions_by_name = {entry.version: entry for entry in dataset.versions}

    if version is None or version == "latest":
        if dataset.latest is not None:
            resolved_version = dataset.latest
        else:
            resolved_version = resolve_latest_version([entry.version for entry in dataset.versions])
    else:
        resolved_version = version

    resolved = versions_by_name.get(resolved_version)
    if resolved is None:
        raise ValueError(f"Dataset '{name}@{resolved_version}' not found in registry")

    return RegistryDatasetSpec(
        name=dataset.name,
        version=resolved.version,
        description=resolved.description or dataset.description,
        git_url=resolved.git_url,
        git_ref=resolved.git_ref,
        path=resolved.path,
        checksum_sha256=resolved.checksum_sha256,
    )


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
