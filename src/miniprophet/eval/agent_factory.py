"""Agent factory for prophet eval."""

from __future__ import annotations

import importlib
import inspect
import threading
from typing import Any

from miniprophet import ContextManager, Environment, Model
from miniprophet.agent.default import AgentConfig
from miniprophet.eval.agent_runtime import (
    BatchForecastAgent,
    EvalBatchAgentWrapper,
    RateLimitCoordinator,
)
from miniprophet.eval.progress import EvalProgressManager


class EvalAgentFactory:
    """Construct eval-capable agents from built-ins or import paths."""

    DEFAULT_AGENT_NAME = "default"

    @classmethod
    def _import_agent_class(cls, import_path: str) -> type:
        if ":" not in import_path:
            raise ValueError("Agent import path must be in format 'module.path:ClassName'")

        module_path, class_name = import_path.split(":", 1)
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ValueError(f"Failed to import module '{module_path}': {exc}") from exc

        try:
            agent_cls = getattr(module, class_name)
        except AttributeError as exc:
            raise ValueError(f"Module '{module_path}' has no class '{class_name}'") from exc

        return agent_cls

    @classmethod
    def create(
        cls,
        *,
        model: Model,
        env: Environment,
        context_manager: ContextManager | None,
        agent_name: str | None,
        agent_import_path: str | None,
        agent_kwargs: dict[str, Any],
        task_id: str,
        coordinator: RateLimitCoordinator | None,
        progress_manager: EvalProgressManager | None,
        cancel_event: threading.Event | None,
    ) -> BatchForecastAgent | EvalBatchAgentWrapper:
        if agent_import_path:
            agent_cls = cls._import_agent_class(agent_import_path)
            init_kwargs = dict(agent_kwargs)

            signature = inspect.signature(agent_cls)
            if "context_manager" in signature.parameters:
                init_kwargs["context_manager"] = context_manager

            try:
                agent = agent_cls(model=model, env=env, **init_kwargs)
            except TypeError as exc:
                raise ValueError(
                    "Custom agent constructor is incompatible. "
                    "Expected at least (model=..., env=..., **kwargs)."
                ) from exc

            return EvalBatchAgentWrapper(
                agent=agent,
                task_id=task_id,
                coordinator=coordinator,
                progress_manager=progress_manager,
                cancel_event=cancel_event,
            )

        resolved_name = (agent_name or cls.DEFAULT_AGENT_NAME).strip().lower()
        if resolved_name != cls.DEFAULT_AGENT_NAME:
            raise ValueError(
                f"Unknown built-in eval agent '{resolved_name}'. "
                f"Only '{cls.DEFAULT_AGENT_NAME}' is currently supported."
            )

        return BatchForecastAgent(
            model=model,
            env=env,
            context_manager=context_manager,
            config_class=AgentConfig,
            task_id=task_id,
            coordinator=coordinator,
            progress_manager=progress_manager,
            cancel_event=cancel_event,
            **agent_kwargs,
        )
