"""Shared helpers for model adapters."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from miniprophet.exceptions import FormatError

NO_TOOL_CALL_ERROR = (
    "No tool call found in the response. You MUST make exactly one tool call per step."
)
MULTIPLE_TOOL_CALLS_ERROR = (
    "Multiple tool calls found. You MUST make exactly ONE tool call per step."
)


def _format_error_message(format_error_template: str, error: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": format_error_template.format(error=error),
        "extra": {"interrupt_type": "FormatError"},
    }


def require_single_tool_call(tool_calls: Sequence[Any] | None, format_error_template: str) -> Any:
    calls = list(tool_calls or [])
    if not calls:
        raise FormatError(_format_error_message(format_error_template, NO_TOOL_CALL_ERROR))
    if len(calls) > 1:
        raise FormatError(_format_error_message(format_error_template, MULTIPLE_TOOL_CALLS_ERROR))
    return calls[0]


def parse_single_action(
    tool_calls: Sequence[Any] | None,
    format_error_template: str,
    action_builder: Callable[[Any], dict[str, Any]],
) -> list[dict[str, Any]]:
    return [action_builder(require_single_tool_call(tool_calls, format_error_template))]


def format_observation_messages(message: dict, outputs: list[dict]) -> list[dict]:
    actions = message.get("extra", {}).get("actions", [])
    not_executed = {"output": "Action was not executed.", "error": True}
    padded = outputs + [not_executed] * (len(actions) - len(outputs))

    results: list[dict] = []
    for action, output in zip(actions, padded):
        raw = output.get("output", "")
        if output.get("error"):
            content = f"<error>\n{raw}\n</error>"
        else:
            content = f"<output>\n{raw}\n</output>"
        msg: dict[str, Any] = {
            "content": content,
            "extra": {
                "error": output.get("error", False),
                "search_cost": output.get("search_cost", 0.0),
                "timestamp": time.time(),
            },
        }
        if "tool_call_id" in action and action["tool_call_id"]:
            msg["role"] = "tool"
            msg["tool_call_id"] = action["tool_call_id"]
        else:
            msg["role"] = "user"
        results.append(msg)
    return results
