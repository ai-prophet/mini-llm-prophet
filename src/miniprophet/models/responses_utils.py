"""Helpers for working with LiteLLM Responses API shapes."""

from __future__ import annotations

import json
from typing import Any


def _item_to_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return dict(item)


def _coerce_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments if arguments is not None else {})


def prepare_response_messages(messages: list[dict]) -> list[dict]:
    """Convert chat-style history into Responses API input items."""
    prepared: list[dict] = []

    for message in messages:
        msg = {k: v for k, v in message.items() if k != "extra"}
        role = msg.get("role")

        if role == "tool":
            item: dict[str, Any] = {
                "type": "function_call_output",
                "output": msg.get("content", ""),
            }
            if msg.get("tool_call_id"):
                item["call_id"] = msg["tool_call_id"]
            prepared.append(item)
            continue

        tool_calls = msg.pop("tool_calls", []) or []
        if role in {"system", "user", "assistant"}:
            prepared.append({"role": role, "content": msg.get("content", "")})
        else:
            prepared.append(msg)

        for tool_call in tool_calls:
            fn = tool_call.get("function", {})
            prepared.append(
                {
                    "type": "function_call",
                    "call_id": tool_call.get("id") or tool_call.get("call_id"),
                    "name": fn.get("name", ""),
                    "arguments": _coerce_arguments(fn.get("arguments", "{}")),
                }
            )

    return prepared


def prepare_response_tools(tools: list[dict]) -> list[dict]:
    """Convert chat-completions tool schemas into Responses API schemas."""
    prepared: list[dict] = []
    for tool in tools:
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            prepared.append(tool)
            continue

        fn = tool["function"]
        prepared.append(
            {
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        )
    return prepared


def response_output_items(response: Any) -> list[dict[str, Any]]:
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    if output is None and hasattr(response, "model_dump"):
        output = response.model_dump().get("output")
    if not isinstance(output, list):
        return []
    return [_item_to_dict(item) for item in output]


def response_function_calls(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if item.get("type") == "function_call"]


def extract_response_text(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []

    texts: list[str] = []
    for block in content:
        text = _item_to_dict(block).get("text")
        if isinstance(text, str):
            texts.append(text)
    return texts


def build_chat_message_from_response(items: list[dict[str, Any]]) -> dict[str, Any]:
    content_chunks: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for item in items:
        item_type = item.get("type")

        if item_type == "message" and item.get("role") == "assistant":
            content_chunks.extend(extract_response_text(item.get("content")))
            continue

        if item_type == "function_call":
            tool_calls.append(
                {
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": _coerce_arguments(item.get("arguments", "{}")),
                    },
                }
            )

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(chunk for chunk in content_chunks if chunk).strip(),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def action_from_response_function_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool_call.get("name", ""),
        "arguments": _coerce_arguments(tool_call.get("arguments", "{}")),
        "tool_call_id": tool_call.get("call_id") or tool_call.get("id"),
    }
