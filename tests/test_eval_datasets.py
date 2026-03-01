from __future__ import annotations

import json
from pathlib import Path

import pytest

from miniprophet.eval.datasets.loader import DatasetSourceKind, parse_dataset_ref
from miniprophet.eval.datasets.registry import RegistryDatasetSpec, resolve_latest_version
from miniprophet.eval.datasets.schema import ForecastTaskRow
from miniprophet.eval.datasets.validate import load_problems


def test_parse_dataset_ref_registry() -> None:
    parsed = parse_dataset_ref("weekly-nba@2026-03-01")
    assert parsed.kind == DatasetSourceKind.REGISTRY
    assert parsed.name == "weekly-nba"
    assert parsed.version == "2026-03-01"


def test_parse_dataset_ref_hf() -> None:
    parsed = parse_dataset_ref("alice/weekly-forecasts@main")
    assert parsed.kind == DatasetSourceKind.HF
    assert parsed.hf_repo == "alice/weekly-forecasts"
    assert parsed.hf_revision == "main"


def test_resolve_latest_version_prefers_latest() -> None:
    assert resolve_latest_version(["2026-02-20", "latest", "1.2.0"]) == "latest"


def test_resolve_latest_version_prefers_date_then_semver() -> None:
    assert resolve_latest_version(["2026-02-28", "2026-03-01", "1.9.0"]) == "2026-03-01"
    assert resolve_latest_version(["1.2.0", "v2.0.0", "alpha"]) == "v2.0.0"


def test_registry_dataset_name_disallows_slash() -> None:
    with pytest.raises(ValueError, match="cannot contain '/'"):
        RegistryDatasetSpec(
            name="alice/dataset",
            version="latest",
            description="",
            git_url="https://example.com/repo.git",
            git_ref="main",
            path="dataset.jsonl",
        )


def test_forecast_task_row_parses_task_id_and_predict_by() -> None:
    row = ForecastTaskRow.model_validate(
        {
            "task_id": "r1",
            "title": "Will A happen?",
            "outcomes": ["Yes", "No"],
            "predict_by": "2026-03-01",
        }
    )
    assert row.task_id == "r1"
    assert row.predict_by == "2026-03-01"


def test_load_problems_validates_schema_and_extra_fields(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_id": "t1",
                "title": "Will B happen?",
                "outcomes": ["Yes", "No"],
                "random_col": "kept",
            }
        )
        + "\n"
    )

    problems = load_problems(path)
    assert problems[0].task_id == "t1"
    assert problems[0].metadata["_extra_fields"]["random_col"] == "kept"


def test_load_problems_requires_outcomes(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"title": "Missing outcomes"}) + "\n")

    with pytest.raises(ValueError, match="invalid task schema"):
        load_problems(path)
