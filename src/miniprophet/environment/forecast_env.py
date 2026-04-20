"""ForecastEnvironment: thin dispatcher that delegates to modular Tool instances."""

from __future__ import annotations

import json
import logging
from typing import Any

from miniprophet import Tool
from miniprophet.environment.source_registry import SourceRegistry
from miniprophet.tools.search import SearchBackend

logger = logging.getLogger("miniprophet.environment")


def create_default_tools(
    search_tool: SearchBackend,
    registry: SourceRegistry,
    *,
    search_limit: int = 10,
    search_results_limit: int = 5,
    model_config: dict | None = None,
    subagents_config: dict | None = None,
) -> list[Tool]:
    """Build the main agent's execution tool set.

    Parameters
    ----------
    search_tool, registry, search_limit, search_results_limit
        Shared across main agent and subagents.
    model_config
        The main agent's model config (model_class, model_name, kwargs).
        Required to spawn subagents.  If ``None``, the ``read_source`` and
        ``investigate_subproblem`` spawner tools are omitted from the
        returned tool set.
    subagents_config
        The ``agent.subagents`` dict from the YAML config.  Controls per-
        subagent step/cost/search limits and model overrides.

    The main agent does NOT have direct ``retrieve_source`` access — it
    calls the spawner ``read_source`` which delegates to a SourceReadingAgent.
    """
    from miniprophet.tools.list_sources_tool import ListSourcesTool
    from miniprophet.tools.search_tool import SearchForecastTool, SearchToolConfig
    from miniprophet.tools.submit import SubmitTool

    main_search = SearchForecastTool(
        search_backend=search_tool,
        registry=registry,
        search_limit=search_limit,
        config=SearchToolConfig(search_results_limit=search_results_limit),
    )

    main_tools: list[Tool] = [
        main_search,
        ListSourcesTool(registry=registry),
    ]

    # Add subagent spawners when model_config is provided.
    if model_config is not None:
        spawners = _build_subagent_spawners(
            search_tool=search_tool,
            registry=registry,
            search_results_limit=search_results_limit,
            model_config=model_config,
            subagents_config=subagents_config or {},
        )
        main_tools.extend(spawners)

    main_tools.append(SubmitTool(registry=registry))
    return main_tools


def _build_subagent_spawners(
    *,
    search_tool: SearchBackend,
    registry: SourceRegistry,
    search_results_limit: int,
    model_config: dict,
    subagents_config: dict,
) -> list[Tool]:
    """Build the ReadSourceTool and InvestigateSubproblemTool spawner tools."""
    from miniprophet.models import get_model
    from miniprophet.subagents.base import SubagentConfig
    from miniprophet.subagents.source_reading import SourceReadingAgent
    from miniprophet.subagents.subproblem import (
        SubproblemAgent,
        SubproblemSubagentConfig,
    )
    from miniprophet.tools.investigate_subproblem import InvestigateSubproblemTool
    from miniprophet.tools.read_source import ReadSourceTool
    from miniprophet.tools.retrieve_source import RetrieveSourceTool
    from miniprophet.tools.search_tool import SearchForecastTool, SearchToolConfig
    from miniprophet.tools.submit_subproblem import SubmitSubproblemTool
    from miniprophet.tools.submit_summary import SubmitSummaryTool
    from miniprophet.utils.serialize import recursive_merge

    sr_config = SubagentConfig(**(subagents_config.get("source_reading") or {}))
    sp_config = SubproblemSubagentConfig(**(subagents_config.get("subproblem") or {}))

    def _merge_model(override: dict | None) -> dict:
        if not override:
            return dict(model_config)
        return recursive_merge(dict(model_config), override)

    def _wrap_list_sources():
        # Local import to avoid top-level cycle
        from miniprophet.tools.list_sources_tool import ListSourcesTool

        return ListSourcesTool(registry=registry)

    def source_reading_factory():
        model_cfg = _merge_model(sr_config.model)
        env = ForecastEnvironment(
            tools=[RetrieveSourceTool(registry=registry), SubmitSummaryTool()],
            registry=registry,
        )
        return SourceReadingAgent(
            model=get_model(config=model_cfg),
            env=env,
            config=sr_config,
        )

    def subproblem_factory():
        model_cfg = _merge_model(sp_config.model)
        sp_search = SearchForecastTool(
            search_backend=search_tool,
            registry=registry,
            search_limit=sp_config.search_limit,
            config=SearchToolConfig(search_results_limit=search_results_limit),
        )
        env = ForecastEnvironment(
            tools=[
                sp_search,
                _wrap_list_sources(),
                RetrieveSourceTool(registry=registry),
                SubmitSubproblemTool(),
            ],
            registry=registry,
        )
        return SubproblemAgent(
            model=get_model(config=model_cfg),
            env=env,
            config=sp_config,
        )

    spawners: list[Tool] = []
    if sr_config.enabled:
        spawners.append(ReadSourceTool(subagent_factory=source_reading_factory))
    if sp_config.enabled:
        spawners.append(InvestigateSubproblemTool(subagent_factory=subproblem_factory))
    return spawners


def create_planning_tools(
    *,
    ask_user_callback: Any = None,
) -> list[Tool]:
    """Build the planning-phase tool set (submit_plan + ask_user)."""
    from miniprophet.tools.ask_user import AskUserTool
    from miniprophet.tools.submit_plan import SubmitPlanTool

    return [
        SubmitPlanTool(),
        AskUserTool(callback=ask_user_callback),
    ]


class ForecastEnvironment:
    """Dispatches tool-call actions to registered Tool instances.

    Supports named tool sets (e.g. ``"execution"`` and ``"planning"``) that
    can be switched at runtime via :meth:`set_active_tools`.
    """

    def __init__(
        self,
        tools: list[Tool],
        *,
        planning_tools: list[Tool] | None = None,
        registry: SourceRegistry | None = None,
        **kwargs: Any,
    ) -> None:
        if registry is None:
            registry = SourceRegistry()
        self.registry = registry
        self._tool_sets: dict[str, dict[str, Tool]] = {
            "execution": {t.name: t for t in tools},
        }
        if planning_tools:
            self._tool_sets["planning"] = {t.name: t for t in planning_tools}
        self._active_set = "execution"
        self._tools: dict[str, Tool] = self._tool_sets[self._active_set]

    def set_active_tools(self, name: str) -> None:
        """Switch the active tool set (e.g. ``'planning'`` or ``'execution'``)."""
        self._tools = self._tool_sets[name]
        self._active_set = name

    def get_active_tool_set(self) -> str:
        return self._active_set

    async def execute(self, action: dict, **kwargs) -> dict:
        tool_name = action.get("name", "")
        try:
            raw_args = action.get("arguments", "{}")
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError as exc:
            return {"output": f"Invalid JSON in tool arguments: {exc}", "error": True}

        tool = self._tools.get(tool_name)
        if tool is None:
            return {"output": f"Unknown tool: {tool_name}", "error": True}
        args.update(kwargs)
        return await tool.execute(args)

    def get_tool_schemas(self) -> list[dict]:
        return [t.get_schema() for t in self._tools.values()]

    def get_tool(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def serialize_sources_state(self) -> dict:
        """Serialize all sources from the registry."""
        return {"sources": self.registry.serialize()}

    def serialize(self) -> dict:
        return {}
