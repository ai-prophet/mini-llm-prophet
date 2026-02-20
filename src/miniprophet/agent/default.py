"""DefaultForecastAgent: the core agent loop for mini-llm-prophet."""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

from pydantic import BaseModel

from miniprophet import ContextManager, Environment, Model, __version__
from miniprophet.exceptions import InterruptAgentFlow, LimitsExceeded
from miniprophet.utils import display
from miniprophet.utils.metrics import evaluate_submission, validate_ground_truth
from miniprophet.utils.serialize import recursive_merge


class AgentConfig(BaseModel):
    system_template: str
    instance_template: str
    step_limit: int = 30
    cost_limit: float = 3.0
    search_limit: int = 10
    max_outcomes: int = 20
    context_window: int = 6
    output_path: Path | None = None


class DefaultForecastAgent:
    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        context_manager: ContextManager | None = None,
        config_class: type = AgentConfig,
        **kwargs,
    ) -> None:
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.context_manager = context_manager
        self.logger = logging.getLogger("miniprophet.agent")
        self.model_cost = 0.0
        self.search_cost = 0.0
        self.n_searches = 0
        self.n_calls = 0

    @property
    def total_cost(self) -> float:
        return self.model_cost + self.search_cost

    def _render(self, template: str, **extra_vars) -> str:
        return template.format_map(extra_vars)

    def add_messages(self, *messages: dict) -> list[dict]:
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

    def run(
        self,
        title: str,
        outcomes: list[str],
        ground_truth: dict[str, int] | None = None,
    ) -> dict:
        if len(outcomes) > self.config.max_outcomes:
            raise ValueError(
                f"Too many outcomes ({len(outcomes)} > {self.config.max_outcomes}). "
                "Increase max_outcomes in config if intentional."
            )
        if ground_truth is not None:
            validate_ground_truth(outcomes, ground_truth)

        outcomes_formatted = ", ".join(outcomes)
        self.messages = []
        self.add_messages(
            self.model.format_message(
                role="system",
                content=self._render(
                    self.config.system_template, title=title, outcomes_formatted=outcomes_formatted
                ),
            ),
            self.model.format_message(
                role="user",
                content=self._render(
                    self.config.instance_template,
                    title=title,
                    outcomes_formatted=outcomes_formatted,
                ),
            ),
        )

        display.print_run_header(
            title,
            outcomes_formatted,
            self.config.step_limit,
            self.config.cost_limit,
            self.config.search_limit,
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

        if ground_truth is not None and result.get("submission"):
            evaluation = evaluate_submission(result["submission"], ground_truth)
            result["evaluation"] = evaluation
            display.print_evaluation(evaluation)

        display.print_run_footer(
            result.get("exit_status", "unknown"),
            self.n_calls,
            self.n_searches,
            self.model_cost,
            self.search_cost,
            self.total_cost,
        )
        return result

    def step(self) -> list[dict]:
        if self.context_manager is not None:
            self.messages = self.context_manager.manage(
                self.messages, step=self.n_calls, board_state=self.env.board.render()
            )
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

        display.print_step_header(self.n_calls, self.model_cost, self.search_cost, self.total_cost)
        display.print_model_response(message)
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        outputs = [self.env.execute(action) for action in actions]
        for action, output in zip(actions, outputs):
            sc = output.get("search_cost", 0.0)
            if sc:
                self.search_cost += sc
                self.n_searches += 1
            if action.get("name") == "search" and self.context_manager is not None:
                raw = action.get("arguments", "{}")
                args = json.loads(raw) if isinstance(raw, str) else raw
                query = args.get("query", "")
                if query and hasattr(self.context_manager, "record_query"):
                    self.context_manager.record_query(query)
            display.print_observation(output)
        return self.add_messages(*self.model.format_observation_messages(message, outputs))

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
                "evaluation": last_extra.get("evaluation", {}),
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
