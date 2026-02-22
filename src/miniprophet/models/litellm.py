"""LiteLLM model implementation for mini-llm-prophet."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Literal

import litellm
from pydantic import BaseModel

from miniprophet.exceptions import FormatError
from miniprophet.models import GLOBAL_MODEL_STATS
from miniprophet.models.retry import retry

logger = logging.getLogger("miniprophet.models.litellm")


class LitellmModelConfig(BaseModel):
    model_name: str
    """Model name with provider prefix, e.g. ``openai/gpt-4o``, ``anthropic/claude-sonnet-4-5-20250929``."""
    model_kwargs: dict[str, Any] = {}
    format_error_template: str = "Error: {error}. Please make a valid tool call."
    observation_template: str = "{output}"
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv(
        "MINIPROPHET_COST_TRACKING", "default"
    )  # type: ignore


class LitellmModel:
    abort_exceptions: list[type[Exception]] = [
        litellm.exceptions.UnsupportedParamsError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.AuthenticationError,
        KeyboardInterrupt,
    ]

    def __init__(self, **kwargs: Any) -> None:
        self.config = LitellmModelConfig(**kwargs)

    # ------------------------------------------------------------------
    # Core query
    # ------------------------------------------------------------------

    def query(self, messages: list[dict], tools: list[dict]) -> dict:
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages(messages), tools)

        cost_info = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_info["cost"])

        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": self._parse_actions(response),
            "response": response.model_dump(),
            **cost_info,
            "timestamp": time.time(),
        }
        return message

    def _query(self, messages: list[dict], tools: list[dict]):
        return litellm.completion(
            model=self.config.model_name,
            messages=messages,
            tools=tools,
            **self.config.model_kwargs,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_messages(self, messages: list[dict]) -> list[dict]:
        """Strip 'extra' keys before sending to the API."""
        return [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]

    def _calculate_cost(self, response) -> dict[str, float]:
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}. "
                    "Set cost_tracking='ignore_errors' in config to suppress."
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
        return {"cost": cost}

    def _parse_actions(self, response) -> list[dict]:
        """Parse tool calls from the response. Raises FormatError on problems."""
        tool_calls = response.choices[0].message.tool_calls or []
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
        fn_name = tc.function.name or ""
        fn_args = tc.function.arguments or "{}"
        return [
            {
                "name": fn_name,
                "arguments": fn_args,
                "tool_call_id": tc.id,
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
