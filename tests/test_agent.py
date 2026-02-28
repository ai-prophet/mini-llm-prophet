from __future__ import annotations

from pathlib import Path

import pytest
from conftest import DummyEnvironment, DummyModel

from miniprophet.agent.context import SlidingWindowContextManager
from miniprophet.agent.default import DefaultForecastAgent
from miniprophet.agent.trajectory import TrajectoryRecorder
from miniprophet.exceptions import Submitted


def test_sliding_window_context_no_truncation() -> None:
    mgr = SlidingWindowContextManager(window_size=4)
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]

    out = mgr.manage(messages, step=1)
    assert out == messages


def test_sliding_window_context_truncates_and_includes_query_history() -> None:
    mgr = SlidingWindowContextManager(window_size=2)
    mgr.record_query("q1")
    mgr.record_query("q2")

    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t1"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u2"},
    ]

    out = mgr.manage(messages, step=2)
    assert out[2]["extra"]["is_truncation_notice"] is True
    assert "Search Queries So Far" in out[2]["content"]
    assert "1. q1" in out[2]["content"]
    assert mgr.total_truncated == 2


def test_trajectory_recorder_deduplicates_by_identity() -> None:
    recorder = TrajectoryRecorder()
    msg = {"role": "system", "content": "a"}

    k1 = recorder.register(msg)[0]
    k2 = recorder.register(msg)[0]

    assert k1 == "S0"
    assert k1 == k2


def test_trajectory_recorder_records_steps() -> None:
    recorder = TrajectoryRecorder()
    recorder.record_step(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        {"role": "assistant", "content": "a"},
    )

    payload = recorder.serialize()
    assert payload["steps"][0]["input"] == ["S0", "U0"]
    assert payload["steps"][0]["output"] == "A0"


class _SubmitEnv(DummyEnvironment):
    def execute(self, action: dict, **kwargs) -> dict:
        if action.get("name") == "submit":
            raise Submitted(
                {
                    "role": "exit",
                    "content": "done",
                    "extra": {
                        "exit_status": "submitted",
                        "submission": {"Yes": 0.7, "No": 0.3},
                        "board": [],
                    },
                }
            )
        return super().execute(action, **kwargs)


def _agent_kwargs(tmp_path: Path) -> dict:
    return {
        "system_template": "sys: {title}",
        "instance_template": "inst: {title} {outcomes_formatted} {current_time}",
        "step_limit": 2,
        "cost_limit": 99.0,
        "search_limit": 9,
        "max_outcomes": 10,
        "output_path": tmp_path,
    }


def test_default_agent_run_submitted_and_evaluated(
    assistant_action_message,
    tmp_path: Path,
) -> None:
    model = DummyModel(
        scripted_messages=[
            assistant_action_message(
                name="submit", arguments='{"probabilities": {"Yes": 0.7, "No": 0.3}}'
            )
        ]
    )
    env = _SubmitEnv()
    agent = DefaultForecastAgent(model=model, env=env, **_agent_kwargs(tmp_path))

    result = agent.run(
        title="Will X happen?",
        outcomes=["Yes", "No"],
        ground_truth={"Yes": 1, "No": 0},
    )

    assert result["exit_status"] == "submitted"
    assert result["submission"] == {"Yes": 0.7, "No": 0.3}
    assert result["evaluation"]["brier_score"] == pytest.approx(0.09)
    assert (tmp_path / "info.json").exists()
    assert (tmp_path / "trajectory.json").exists()
    assert (tmp_path / "sources.json").exists()


class _RecordingContext:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def manage(self, messages: list[dict], *, step: int, **kwargs) -> list[dict]:
        return messages

    def record_query(self, query: str) -> None:
        self.queries.append(query)

    def display(self) -> None:
        return None


def test_default_agent_tracks_search_cost_and_query_history(
    assistant_action_message,
    tmp_path: Path,
) -> None:
    model = DummyModel(
        scripted_messages=[
            assistant_action_message(name="search", arguments='{"query": "fed rates"}', cost=0.11),
        ]
    )
    env = DummyEnvironment(outputs=[{"output": "ok", "search_cost": 0.25}])
    ctx = _RecordingContext()
    kwargs = _agent_kwargs(tmp_path)
    kwargs["step_limit"] = 1
    agent = DefaultForecastAgent(model=model, env=env, context_manager=ctx, **kwargs)

    result = agent.run(title="Q", outcomes=["A", "B"])

    assert result["exit_status"] == "LimitsExceeded"
    assert agent.model_cost == pytest.approx(0.11)
    assert agent.search_cost == pytest.approx(0.25)
    assert agent.n_searches == 1
    assert ctx.queries == ["fed rates"]


def test_default_agent_rejects_invalid_outcome_count(tmp_path: Path) -> None:
    agent = DefaultForecastAgent(
        model=DummyModel(),
        env=DummyEnvironment(),
        **{
            **_agent_kwargs(tmp_path),
            "max_outcomes": 1,
        },
    )

    with pytest.raises(ValueError, match="Too many outcomes"):
        agent.run(title="Q", outcomes=["A", "B"])
