"""SubagentBase: minimal agent loop for delegated subagents.

Subagents are spawned by the main agent via spawner tools (read_source,
investigate_subproblem).  Each subagent:

- has its own Model, ForecastEnvironment (with a restricted tool set), and
  message history
- runs in a ``contextvars``-scoped buffered Rich Console so concurrent
  subagents don't interleave their output
- enforces its own per-invocation step and cost limits
- has no planning phase and no grace period (simpler than DefaultForecastAgent)

Subagents do NOT spawn further subagents: their environment has no spawner
tools.  The invariant is enforced at the environment level via the tool set.
"""

from __future__ import annotations

import asyncio
import io
import logging

from pydantic import BaseModel
from rich.console import Console

from miniprophet import Environment, Model
from miniprophet.agent.trajectory import TrajectoryRecorder
from miniprophet.cli.utils import reset_console_override, set_console_override
from miniprophet.exceptions import InterruptAgentFlow, LimitsExceeded


class SubagentConfig(BaseModel):
    """Base config shared by all subagents."""

    enabled: bool = True
    step_limit: int = 5
    cost_limit: float = 0.1
    system_template: str = ""
    model: dict | None = None  # None = inherit from parent


class SubagentBase:
    """Minimal agent loop used by SourceReadingAgent and SubproblemAgent.

    Subclasses implement :meth:`_instance_prompt` and :meth:`_build_result`.
    """

    def __init__(
        self,
        *,
        model: Model,
        env: Environment,
        config: SubagentConfig,
    ) -> None:
        self.model = model
        self.env = env
        self.config = config
        self.logger = logging.getLogger(f"miniprophet.subagents.{self.__class__.__name__}")
        self.messages: list[dict] = []
        self.n_calls = 0
        self.model_cost = 0.0
        self.search_cost = 0.0
        self.n_searches = 0
        self._trajectory = TrajectoryRecorder()

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------

    def _render(self, template: str, **vars) -> str:
        return template.format_map(vars)

    # Subclasses override these
    def _instance_prompt(self, **render_vars) -> str:
        raise NotImplementedError

    def _build_result(self, rendered_trace: str):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, **render_vars):
        """Run the subagent to completion and return a structured result.

        Redirects Rich output into an in-memory buffered Console for the
        duration of the run (via a task-local ``ContextVar`` override).  The
        captured text is attached to the result as ``rendered_trace``.
        """
        buffer = Console(
            file=io.StringIO(),
            force_terminal=True,
            width=120,
            record=True,
        )
        token = set_console_override(buffer)
        try:
            system_content = self._render(self.config.system_template, **render_vars)
            user_content = self._instance_prompt(**render_vars)

            self.messages = [
                self.model.format_message(role="system", content=system_content),
                self.model.format_message(role="user", content=user_content),
            ]
            self.on_run_start(**render_vars)

            while True:
                try:
                    await self._step()
                except LimitsExceeded as exc:
                    self.messages.extend(exc.messages)
                    break
                except InterruptAgentFlow as exc:
                    self.messages.extend(exc.messages)
                except Exception as exc:  # uncaught
                    self.logger.exception("Subagent uncaught exception: %s", exc)
                    self.messages.append(
                        self.model.format_message(
                            role="exit",
                            content=str(exc),
                            extra={"exit_status": type(exc).__name__},
                        )
                    )
                    break

                if self.messages and self.messages[-1].get("role") == "exit":
                    break
        finally:
            reset_console_override(token)

        try:
            rendered = buffer.export_text(clear=False)
        except Exception:
            rendered = buffer.file.getvalue() if hasattr(buffer.file, "getvalue") else ""

        return self._build_result(rendered)

    # ------------------------------------------------------------------
    # Loop internals
    # ------------------------------------------------------------------

    async def _step(self) -> None:
        await self._execute_actions(await self._query())

    async def _query(self) -> dict:
        """One model call.  Enforces per-subagent step and cost limits."""
        if 0 < self.config.step_limit <= self.n_calls:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "Subagent step limit exceeded.",
                    "extra": {"exit_status": "SubagentLimitsExceeded"},
                }
            )
        if 0 < self.config.cost_limit <= (self.model_cost + self.search_cost):
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "Subagent cost limit exceeded.",
                    "extra": {"exit_status": "SubagentLimitsExceeded"},
                }
            )

        self.n_calls += 1
        input_snapshot = list(self.messages)
        tools = self.env.get_tool_schemas()
        message = await self.model.query(self.messages, tools)
        extra = message.get("extra", {})
        self.model_cost += extra.get("cost", 0.0) or 0.0
        self.messages.append(message)
        self._trajectory.record_step(input_snapshot, message)

        self.on_step_start()
        self.on_model_response(message)
        return message

    async def _execute_actions(self, message: dict) -> None:
        actions = message.get("extra", {}).get("actions", [])
        if not actions:
            return

        # Parallel execution (subagent tools are short and not nested)
        outputs = await asyncio.gather(*[self.env.execute(a) for a in actions])

        for action, output in zip(actions, outputs):
            sc = output.get("search_cost", 0.0) or 0.0
            if sc:
                self.search_cost += sc
                self.n_searches += 1
            self.on_observation(action, output)

        self.messages.extend(self.model.format_observation_messages(message, outputs))

    # ------------------------------------------------------------------
    # Display hooks (default: delegate to tool.display for observations)
    # ------------------------------------------------------------------

    def on_run_start(self, **render_vars) -> None:
        pass

    def on_step_start(self) -> None:
        from miniprophet.cli.components.step_display import print_step_header

        print_step_header(
            self.n_calls,
            self.model_cost,
            self.search_cost,
            self.model_cost + self.search_cost,
        )

    def on_model_response(self, message: dict) -> None:
        from miniprophet.cli.components.step_display import print_model_response

        print_model_response(message, max_thinking_chars=300)

    def on_observation(self, action: dict, output: dict) -> None:
        tool_name = action.get("name", "")
        tool = self.env._tools.get(tool_name)
        if tool is not None and hasattr(tool, "display"):
            tool.display(output)
            return
        from miniprophet.cli.components.observation import print_observation

        print_observation(output)
