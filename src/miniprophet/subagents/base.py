"""SubagentBase: minimal agent loop for delegated subagents.

Subagents are spawned by the main agent via spawner tools (read_source,
investigate_subproblem).  Each subagent:

- has its own Model, ForecastEnvironment (with a restricted tool set), and
  message history
- enforces its own per-invocation step and cost limits
- has no planning phase and no grace period (simpler than DefaultForecastAgent)
- does NOT print panels inline — it exposes a live :class:`SubagentStatus`
  object that the spawner tool renders in a Rich ``Live`` display

Subagents do NOT spawn further subagents: their environment has no spawner
tools.  The invariant is enforced at the environment level via the tool set.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel

from miniprophet import Environment, Model
from miniprophet.agent.trajectory import TrajectoryRecorder
from miniprophet.exceptions import InterruptAgentFlow, LimitsExceeded
from miniprophet.subagents.status import SubagentStatus


class SubagentConfig(BaseModel):
    """Base config shared by all subagents."""

    enabled: bool = True
    step_limit: int = 5
    cost_limit: float = 0.1
    system_template: str = ""
    model: dict | None = None  # None = inherit from parent


class SubagentBase:
    """Minimal agent loop used by SourceReadingAgent and SubproblemAgent.

    Subclasses implement :meth:`_instance_prompt`, :meth:`_build_result`,
    and override :attr:`kind` / :meth:`_status_label`.

    Subagents do not print directly.  Their :attr:`status` is updated in hooks
    and read by the spawner tool's Rich Live display.
    """

    # Subclass override
    kind: str = "subagent"

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
        self.n_tool_calls = 0
        self.model_cost = 0.0
        self.search_cost = 0.0
        self.n_searches = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._trajectory = TrajectoryRecorder()
        self._status = SubagentStatus(
            kind=self.kind,
            step_limit=config.step_limit,
        )

    @property
    def status(self) -> SubagentStatus:
        """Live-updating status object.  Spawner tools read this via a Rich Live."""
        return self._status

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------

    def _render(self, template: str, **vars) -> str:
        return template.format_map(vars)

    # Subclasses override these
    def _instance_prompt(self, **render_vars) -> str:
        raise NotImplementedError

    def _status_label(self, **render_vars) -> str:
        """Brief label shown in the live status, e.g. 'S4, focus=injury'."""
        return ""

    def _build_result(self):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, **render_vars):
        """Run the subagent loop.  Updates :attr:`status` throughout.

        Does NOT print inline — the spawner tool is expected to wrap this
        call in a ``rich.live.Live`` using :class:`SubagentStatusRenderable`.
        """
        self._status.label = self._status_label(**render_vars)
        self._status.state = "starting"

        system_content = self._render(self.config.system_template, **render_vars)
        user_content = self._instance_prompt(**render_vars)

        self.messages = [
            self.model.format_message(role="system", content=system_content),
            self.model.format_message(role="user", content=user_content),
        ]

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

        self._status.state = "done"
        return self._build_result()

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
        self._status.state = "thinking"
        self._status.step = self.n_calls

        input_snapshot = list(self.messages)
        tools = self.env.get_tool_schemas()
        message = await self.model.query(self.messages, tools)
        extra = message.get("extra", {})

        # Cost + token tracking
        self.model_cost += extra.get("cost", 0.0) or 0.0
        call_prompt = extra.get("prompt_tokens", 0) or 0
        call_completion = extra.get("completion_tokens", 0) or 0
        self.prompt_tokens += call_prompt
        self.completion_tokens += call_completion

        self.messages.append(message)
        self._trajectory.record_step(input_snapshot, message)

        # Sync status
        self._status.model_cost = self.model_cost
        self._status.search_cost = self.search_cost
        self._status.prompt_tokens = self.prompt_tokens
        self._status.completion_tokens = self.completion_tokens

        # Announce what the model wants to do next
        actions = extra.get("actions", [])
        if actions:
            names = ", ".join(a.get("name", "?") for a in actions)
            self._status.state = f"calling: {names}"
        else:
            self._status.state = "responding (no tool call)"

        return message

    async def _execute_actions(self, message: dict) -> None:
        actions = message.get("extra", {}).get("actions", [])
        if not actions:
            return

        # Count tool calls eagerly: even a call that raises (like
        # submit_summary) is a tool call the model made.
        self.n_tool_calls += len(actions)
        self._status.n_tool_calls = self.n_tool_calls

        # Parallel within a subagent (subagent tools are short and not nested)
        outputs = await asyncio.gather(*[self.env.execute(a) for a in actions])

        for _action, output in zip(actions, outputs):
            sc = output.get("search_cost", 0.0) or 0.0
            if sc:
                self.search_cost += sc
                self.n_searches += 1

        # Sync status (search_cost may have changed)
        self._status.search_cost = self.search_cost

        self.messages.extend(self.model.format_observation_messages(message, outputs))
