from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from conftest import DummyEnvironment, DummyModel

from miniprophet.eval.agent_runtime import BatchForecastAgent, RateLimitCoordinator
from miniprophet.eval.runner import (
    ForecastProblem,
    _is_auth_error,
    _is_rate_limit_error,
    _run_agent_with_timeout,
    load_existing_summary,
    load_problems,
    to_mm_dd_yyyy,
)
from miniprophet.exceptions import BatchRunTimeoutError, SearchRateLimitError


def test_to_mm_dd_yyyy_applies_offset() -> None:
    assert to_mm_dd_yyyy("2026-01-10", offset=2) == "01/08/2026"


def test_forecast_problem_invalid_end_time_becomes_none() -> None:
    p = ForecastProblem(run_id="r1", title="t", outcomes=["a", "b"], predict_by="not-a-date")
    assert p.predict_by is None


def test_load_problems_assigns_auto_run_ids(tmp_path: Path) -> None:
    path = tmp_path / "problems.jsonl"
    path.write_text(
        '{"title": "Q1", "outcomes": ["A", "B"]}\n'
        '{"title": "Q2", "outcomes": ["C", "D"], "run_id": "custom"}\n'
    )

    probs = load_problems(path)
    assert [p.run_id for p in probs] == ["run_0", "custom"]


def test_load_problems_rejects_duplicate_run_ids(tmp_path: Path) -> None:
    path = tmp_path / "problems.jsonl"
    path.write_text(
        '{"title": "Q1", "outcomes": ["A", "B"], "run_id": "x"}\n'
        '{"title": "Q2", "outcomes": ["C", "D"], "run_id": "x"}\n'
    )

    with pytest.raises(ValueError, match="duplicate run_id"):
        load_problems(path)


def test_load_existing_summary_reads_results_and_total_cost(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "total_cost": 1.2,
                "runs": [
                    {
                        "run_id": "r1",
                        "title": "t",
                        "status": "submitted",
                        "cost": {"model": 0.2, "search": 0.3, "total": 0.5},
                    }
                ],
            }
        )
    )

    runs, total = load_existing_summary(summary)
    assert total == pytest.approx(1.2)
    assert runs["r1"].status == "submitted"


def test_load_existing_summary_rejects_bad_shape(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text('{"runs": {"not": "a-list"}}')
    with pytest.raises(ValueError, match="must be a list"):
        load_existing_summary(summary)


@pytest.mark.parametrize(
    "exc,expected",
    [
        (SearchRateLimitError("x"), True),
        (RuntimeError("HTTP 429 from provider"), True),
        (RuntimeError("other"), False),
    ],
)
def test_is_rate_limit_error(exc: Exception, expected: bool) -> None:
    assert _is_rate_limit_error(exc) is expected


@pytest.mark.parametrize(
    "exc,expected",
    [
        (RuntimeError("Authentication failed"), True),
        (RuntimeError("status code 401"), True),
        (RuntimeError("something else"), False),
    ],
)
def test_is_auth_error(exc: Exception, expected: bool) -> None:
    assert _is_auth_error(exc) is expected


class _SleepAgent:
    def __init__(self, sleep_s: float = 0.0) -> None:
        self.sleep_s = sleep_s

    def run(self, **kwargs):
        if self.sleep_s:
            time.sleep(self.sleep_s)
        return {"exit_status": "submitted"}


def test_run_agent_with_timeout_success() -> None:
    result = _run_agent_with_timeout(
        agent=_SleepAgent(),
        timeout_seconds=0.2,
        cancel_event=threading.Event(),
        run_id="r1",
        title="T",
        outcomes=["A", "B"],
        ground_truth=None,
        runtime_kwargs={},
    )
    assert result["exit_status"] == "submitted"


def test_run_agent_with_timeout_sets_cancel_event_on_timeout() -> None:
    cancel = threading.Event()
    with pytest.raises(BatchRunTimeoutError, match="timed out"):
        _run_agent_with_timeout(
            agent=_SleepAgent(sleep_s=0.1),
            timeout_seconds=0.01,
            cancel_event=cancel,
            run_id="r1",
            title="T",
            outcomes=["A", "B"],
            ground_truth=None,
            runtime_kwargs={},
        )
    assert cancel.is_set()


class _Progress:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []

    def update_run_status(self, run_id: str, message: str, cost_delta: float = 0.0) -> None:
        self.calls.append((run_id, message, cost_delta))


def test_batch_forecast_agent_step_updates_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_super_step(self):
        self.model_cost = 1.5
        return [{"role": "tool", "content": "ok"}]

    monkeypatch.setattr(
        "miniprophet.eval.agent_runtime.DefaultForecastAgent.step", _fake_super_step
    )

    progress = _Progress()
    agent = BatchForecastAgent(
        model=DummyModel(),
        env=DummyEnvironment(),
        run_id="run-x",
        progress_manager=progress,
        system_template="sys",
        instance_template="inst",
    )

    out = agent.step()

    assert out[0]["content"] == "ok"
    assert progress.calls[0][0] == "run-x"
    assert progress.calls[0][2] == pytest.approx(1.5)


def test_batch_forecast_agent_step_respects_cancel_event() -> None:
    cancel = threading.Event()
    cancel.set()

    agent = BatchForecastAgent(
        model=DummyModel(),
        env=DummyEnvironment(),
        cancel_event=cancel,
        system_template="sys",
        instance_template="inst",
    )

    with pytest.raises(BatchRunTimeoutError, match="timed out"):
        agent.step()


def test_rate_limit_coordinator_wait_cancelled() -> None:
    coordinator = RateLimitCoordinator(backoff_seconds=0.2)
    coordinator.signal_rate_limit(backoff_seconds=0.2)
    cancel = threading.Event()
    cancel.set()

    assert coordinator.wait_if_paused(cancel_event=cancel) is False
