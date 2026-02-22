"""OpenRouter API model implementation for mini-llm-prophet."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Literal

import requests
from pydantic import BaseModel

from miniprophet.exceptions import FormatError
from miniprophet.models import GLOBAL_MODEL_STATS
from miniprophet.models.retry import retry

logger = logging.getLogger("miniprophet.models.openrouter")


class OpenRouterModelConfig(BaseModel):
    model_name: str
    model_kwargs: dict[str, Any] = {}
    format_error_template: str = "Error: {error}. Please make a valid tool call."
    observation_template: str = "{output}"
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv(
        "MINIPROPHET_COST_TRACKING", "default"
    )  # type: ignore


class OpenRouterAPIError(Exception):
    pass


class OpenRouterAuthenticationError(Exception):
    pass


class OpenRouterRateLimitError(Exception):
    pass


class OpenRouterModel:
    abort_exceptions: list[type[Exception]] = [OpenRouterAuthenticationError, KeyboardInterrupt]  # type: ignore

    def __init__(self, **kwargs: Any) -> None:
        self.config = OpenRouterModelConfig(**kwargs)
        self._api_url = "https://openrouter.ai/api/v1/chat/completions"
        self._api_key = os.getenv("OPENROUTER_API_KEY", "")

    # ------------------------------------------------------------------
    # Core query
    # ------------------------------------------------------------------

    def query(self, messages: list[dict], tools: list[dict]) -> dict:
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages(messages), tools)

        cost_info = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_info["cost"])

        message = dict(response["choices"][0]["message"])
        message["extra"] = {
            "actions": self._parse_actions(response),
            "response": response,
            **cost_info,
            "timestamp": time.time(),
        }
        return message

    def _query(self, messages: list[dict], tools: list[dict]) -> dict:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "tools": tools,
            "usage": {"include": True},
            **self.config.model_kwargs,
        }

        try:
            resp = requests.post(
                self._api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError:
            if resp.status_code == 401:
                raise OpenRouterAuthenticationError(
                    "Authentication failed. Check OPENROUTER_API_KEY."
                )
            if resp.status_code == 429:
                raise OpenRouterRateLimitError("Rate limit exceeded")
            raise OpenRouterAPIError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        except requests.exceptions.RequestException as exc:
            raise OpenRouterAPIError(f"Request failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_messages(self, messages: list[dict]) -> list[dict]:
        """Strip 'extra' keys before sending to the API."""
        return [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]

    def _calculate_cost(self, response: dict) -> dict[str, float]:
        usage = response.get("usage", {})
        cost = usage.get("cost", 0.0)
        if cost is None:
            cost = 0.0
        if cost <= 0.0 and self.config.cost_tracking != "ignore_errors":
            raise RuntimeError(
                f"No valid cost info from OpenRouter for {self.config.model_name}. "
                f"Usage: {usage}. Set cost_tracking='ignore_errors' to suppress."
            )
        return {"cost": cost}

    def _parse_actions(self, response: dict) -> list[dict]:
        """Parse tool calls from the API response. Raises FormatError on problems."""
        tool_calls = response["choices"][0]["message"].get("tool_calls") or []
        if not tool_calls:
            raise FormatError(
                {
                    "role": "user",
                    "content": self.config.format_error_template.format(
                        error="No tool call found in the response. You MUST make exactly one tool call per step."
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        if len(tool_calls) > 1:
            raise FormatError(
                {
                    "role": "user",
                    "content": self.config.format_error_template.format(
                        error="Multiple tool calls found. You MUST make exactly ONE tool call per step."
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                }
            )

        tc = tool_calls[0]
        fn = tc.get("function", {})
        return [
            {
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", "{}"),
                "tool_call_id": tc.get("id"),
            }
        ]

    # ------------------------------------------------------------------
    # Message formatting
    # ------------------------------------------------------------------

    def format_message(self, **kwargs: Any) -> dict:
        return dict(kwargs)

    def format_observation_messages(self, message: dict, outputs: list[dict]) -> list[dict]:
        """Format environment outputs into tool-role messages for the API."""
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
                    "raw_output": raw,
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

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }
