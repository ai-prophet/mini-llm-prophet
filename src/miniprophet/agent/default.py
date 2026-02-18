"""DefaultForecastAgent: the core agent loop for mini-llm-prophet."""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

from pydantic import BaseModel

from miniprophet import Environment, Model, __version__
from miniprophet.exceptions import InterruptAgentFlow, LimitsExceeded
from miniprophet.utils.serialize import recursive_merge


class AgentConfig(BaseModel):
    system_template: str
    instance_template: str
    step_limit: int = 30
    cost_limit: float = 3.0
    search_limit: int = 10
    max_outcomes: int = 20
    output_path: Path | None = None


class DefaultForecastAgent:
    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: type = AgentConfig,
        **kwargs,
    ) -> None:
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.logger = logging.getLogger("miniprophet.agent")
        self.model_cost = 0.0
        self.search_cost = 0.0
        self.n_searches = 0
        self.n_calls = 0

    @property
    def total_cost(self) -> float:
        return self.model_cost + self.search_cost

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------

    def _render(self, template: str, **extra_vars) -> str:
        """Render a template with str.format_map (no Jinja2)."""
        return template.format_map(extra_vars)

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug("Adding %d message(s)", len(messages))
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, exc: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(exc),
                extra={
                    "exit_status": type(exc).__name__,
                    "submission": "",
                    "exception_str": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def run(self, title: str, outcomes: list[str]) -> dict:
        if len(outcomes) > self.config.max_outcomes:
            raise ValueError(
                f"Too many outcomes ({len(outcomes)} > {self.config.max_outcomes}). "
                "Increase max_outcomes in config if intentional."
            )

        outcomes_formatted = ", ".join(outcomes)
        self.messages = []
        self.add_messages(
            self.model.format_message(
                role="system",
                content=self._render(self.config.system_template, title=title, outcomes_formatted=outcomes_formatted),
            ),
            self.model.format_message(
                role="user",
                content=self._render(self.config.instance_template, title=title, outcomes_formatted=outcomes_formatted),
            ),
        )

        self.logger.info(
            "[bold]Forecasting:[/bold] %s  |  Outcomes: %s",
            title,
            outcomes_formatted,
        )

        while True:
            try:
                self.step()
            except InterruptAgentFlow as exc:
                self.add_messages(*exc.messages)
            except Exception as exc:
                self.handle_uncaught_exception(exc)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break

        result = self.messages[-1].get("extra", {})
        self.logger.info(
            "[bold green]Done.[/bold green]  Status: %s  |  "
            "Model cost: $%.4f  Search cost: $%.4f  Total: $%.4f  |  Steps: %d  Searches: %d",
            result.get("exit_status", "unknown"),
            self.model_cost,
            self.search_cost,
            self.total_cost,
            self.n_calls,
            self.n_searches,
        )
        return result

    def step(self) -> list[dict]:
        return self.execute_actions(self.query())

    def query(self) -> dict:
        if 0 < self.config.step_limit <= self.n_calls:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "Step limit exceeded.",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config.cost_limit <= self.total_cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "Cost limit exceeded.",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )

        self.n_calls += 1
        tools = self.env.get_tool_schemas()
        message = self.model.query(self.messages, tools)
        self.model_cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)

        action_names = [a.get("name", "?") for a in message.get("extra", {}).get("actions", [])]
        self.logger.info(
            "[bold]Step %d[/bold]  tool=%s  model_cost=$%.4f  total=$%.4f",
            self.n_calls,
            ", ".join(action_names) or "none",
            self.model_cost,
            self.total_cost,
        )
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        outputs = [
            self.env.execute(action)
            for action in message.get("extra", {}).get("actions", [])
        ]
        for output in outputs:
            sc = output.get("search_cost", 0.0)
            if sc:
                self.search_cost += sc
                self.n_searches += 1
        return self.add_messages(
            *self.model.format_observation_messages(message, outputs)
        )

    # ------------------------------------------------------------------
    # Serialization / save
    # ------------------------------------------------------------------

    def serialize(self, *extra_dicts: dict) -> dict:
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "cost_stats": {
                    "model_cost": self.model_cost,
                    "search_cost": self.search_cost,
                    "total_cost": self.total_cost,
                    "n_api_calls": self.n_calls,
                    "n_searches": self.n_searches,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-llm-prophet-1.0",
        }
        return recursive_merge(
            agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts
        )

    def save(self, path: Path | None, *extra_dicts: dict) -> dict:
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data
