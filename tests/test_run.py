from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from miniprophet.cli.components.env_editor import is_valid_env_key
from miniprophet.cli.utils import get_console
from miniprophet.run.batch import _build_batch_config, _resolve_resume_state
from miniprophet.run.batch_runner import ForecastProblem


def test_get_console_is_singleton() -> None:
    assert get_console() is get_console()


def test_is_valid_env_key_accepts_and_rejects() -> None:
    assert is_valid_env_key("OPENAI_API_KEY") is True
    assert is_valid_env_key("1BAD") is False


def test_build_batch_config_applies_overrides() -> None:
    cfg = _build_batch_config(
        config_spec=None,
        max_cost_per_run=0.5,
        model_name="openai/gpt-4o-mini",
        model_class="litellm",
    )
    assert cfg["agent"]["cost_limit"] == 0.5
    assert cfg["model"]["model_name"] == "openai/gpt-4o-mini"
    assert cfg["model"]["model_class"] == "litellm"


def test_resolve_resume_state_disabled_returns_inputs(tmp_path: Path) -> None:
    problems = [ForecastProblem(run_id="r1", title="T", outcomes=["A", "B"])]
    filtered, results, total = _resolve_resume_state(
        enabled=False, output=tmp_path, problems=problems
    )
    assert filtered == problems
    assert results == {}
    assert total == 0.0


def test_resolve_resume_state_filters_completed_runs(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "total_cost": 1.0,
                "runs": [
                    {"run_id": "r1", "title": "t", "status": "submitted", "cost": {"total": 0.2}}
                ],
            }
        )
    )
    problems = [
        ForecastProblem(run_id="r1", title="T1", outcomes=["A", "B"]),
        ForecastProblem(run_id="r2", title="T2", outcomes=["A", "B"]),
    ]

    filtered, results, total = _resolve_resume_state(
        enabled=True, output=tmp_path, problems=problems
    )

    assert [p.run_id for p in filtered] == ["r2"]
    assert "r1" in results
    assert total == pytest.approx(1.0)


def test_resolve_resume_state_rejects_unexpected_run_ids(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "total_cost": 1.0,
                "runs": [{"run_id": "unknown", "title": "t", "status": "submitted"}],
            }
        )
    )
    problems = [ForecastProblem(run_id="r1", title="T1", outcomes=["A", "B"])]

    with pytest.raises(typer.Exit):
        _resolve_resume_state(enabled=True, output=tmp_path, problems=problems)
