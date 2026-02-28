from __future__ import annotations

import types

import pytest

import miniprophet.models as models_module
from miniprophet.exceptions import FormatError
from miniprophet.models import GlobalModelStats, get_model
from miniprophet.models.litellm import LitellmModel
from miniprophet.models.openrouter import OpenRouterModel


class _DummyModelClass:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


@pytest.fixture
def patch_dummy_model_class(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = types.ModuleType("_tests_dummy_model_module")
    mod.DummyModel = _DummyModelClass

    import sys

    sys.modules[mod.__name__] = mod
    monkeypatch.setitem(models_module._MODEL_CLASS_MAPPING, "dummy", f"{mod.__name__}.DummyModel")


def test_get_model_uses_mapping_and_kwargs(patch_dummy_model_class) -> None:
    model = get_model({"model_class": "dummy", "model_name": "x", "temperature": 0.2})
    assert isinstance(model, _DummyModelClass)
    assert model.kwargs["model_name"] == "x"
    assert model.kwargs["temperature"] == 0.2


def test_get_model_requires_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINIPROPHET_MODEL_NAME", raising=False)
    with pytest.raises(ValueError, match="No model name set"):
        get_model({"model_class": "dummy"})


def test_global_model_stats_add_and_limit() -> None:
    stats = GlobalModelStats()
    stats.cost_limit = 0.05
    stats.add(0.03)
    assert stats.cost == pytest.approx(0.03)
    assert stats.n_calls == 1
    with pytest.raises(RuntimeError, match="Global model cost limit exceeded"):
        stats.add(0.03)


def test_openrouter_prepare_messages_strips_extra() -> None:
    m = OpenRouterModel(model_name="openai/gpt-4o-mini", cost_tracking="ignore_errors")
    out = m._prepare_messages([{"role": "user", "content": "x", "extra": {"a": 1}}])
    assert out == [{"role": "user", "content": "x"}]


@pytest.mark.parametrize(
    "response,expected_cost",
    [
        ({"usage": {"cost": 0.12}}, 0.12),
        ({"usage": {"cost": None}}, 0.0),
    ],
)
def test_openrouter_calculate_cost_ok(response: dict, expected_cost: float) -> None:
    m = OpenRouterModel(model_name="free-model", cost_tracking="default")
    assert m._calculate_cost(response)["cost"] == pytest.approx(expected_cost)


def test_openrouter_calculate_cost_raises_when_missing_nonfree() -> None:
    m = OpenRouterModel(model_name="openai/gpt-4o-mini", cost_tracking="default")
    with pytest.raises(RuntimeError, match="No valid cost info"):
        m._calculate_cost({"usage": {"cost": 0.0}})


def test_openrouter_parse_actions_requires_single_call() -> None:
    m = OpenRouterModel(model_name="openai/gpt-4o-mini", cost_tracking="ignore_errors")
    with pytest.raises(FormatError) as exc:
        m._parse_actions({"choices": [{"message": {"tool_calls": []}}]})
    assert "No tool call found" in exc.value.messages[0]["content"]
    assert exc.value.messages[0]["extra"]["interrupt_type"] == "FormatError"

    with pytest.raises(FormatError) as exc:
        m._parse_actions(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"id": "1", "function": {"name": "a", "arguments": "{}"}},
                                {"id": "2", "function": {"name": "b", "arguments": "{}"}},
                            ]
                        }
                    }
                ]
            }
        )
    assert "Multiple tool calls found" in exc.value.messages[0]["content"]
    assert exc.value.messages[0]["extra"]["interrupt_type"] == "FormatError"


def test_openrouter_format_observation_messages_tool_and_user_roles() -> None:
    m = OpenRouterModel(model_name="openai/gpt-4o-mini", cost_tracking="ignore_errors")
    message = {
        "extra": {
            "actions": [
                {"name": "search", "arguments": "{}", "tool_call_id": "tc1"},
                {"name": "x", "arguments": "{}", "tool_call_id": None},
            ]
        }
    }
    outputs = [{"output": "ok"}, {"output": "bad", "error": True}]
    msgs = m.format_observation_messages(message, outputs)

    assert msgs[0]["role"] == "tool"
    assert msgs[0]["tool_call_id"] == "tc1"
    assert "<output>" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert "<error>" in msgs[1]["content"]


def _litellm_response(tool_calls):
    msg = types.SimpleNamespace(tool_calls=tool_calls)
    msg.model_dump = lambda: {
        "tool_calls": [
            {
                "id": getattr(tc, "id", None),
                "function": {
                    "name": getattr(getattr(tc, "function", None), "name", ""),
                    "arguments": getattr(getattr(tc, "function", None), "arguments", "{}"),
                },
            }
            for tc in (tool_calls or [])
        ]
    }
    choice = types.SimpleNamespace(message=msg)
    resp = types.SimpleNamespace(choices=[choice])
    resp.model_dump = lambda: {"choices": [{"message": msg.model_dump()}]}
    return resp


def test_litellm_parse_actions_single_call() -> None:
    model = LitellmModel(model_name="openai/gpt-4o-mini", cost_tracking="ignore_errors")
    tc = types.SimpleNamespace(
        id="x",
        function=types.SimpleNamespace(name="search", arguments='{"query":"q"}'),
    )
    actions = model._parse_actions(_litellm_response([tc]))
    assert actions == [{"name": "search", "arguments": '{"query":"q"}', "tool_call_id": "x"}]


def test_litellm_parse_actions_raises_on_empty() -> None:
    model = LitellmModel(model_name="openai/gpt-4o-mini", cost_tracking="ignore_errors")
    with pytest.raises(FormatError) as exc:
        model._parse_actions(_litellm_response([]))
    assert "No tool call found" in exc.value.messages[0]["content"]
    assert exc.value.messages[0]["extra"]["interrupt_type"] == "FormatError"


def test_litellm_calculate_cost_raises_when_tracking_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = LitellmModel(model_name="openai/gpt-4o-mini", cost_tracking="default")
    monkeypatch.setattr(
        "miniprophet.models.litellm.litellm.cost_calculator.completion_cost",
        lambda response, model: 0.0,
    )
    with pytest.raises(RuntimeError, match="Error calculating cost"):
        model._calculate_cost(object())


def test_litellm_calculate_cost_ignore_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    model = LitellmModel(model_name="openai/gpt-4o-mini", cost_tracking="ignore_errors")
    monkeypatch.setattr(
        "miniprophet.models.litellm.litellm.cost_calculator.completion_cost",
        lambda response, model: (_ for _ in ()).throw(ValueError("boom")),
    )
    assert model._calculate_cost(object())["cost"] == 0.0
