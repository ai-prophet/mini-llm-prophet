"""Search tool interface and factory for mini-llm-prophet."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Protocol

from miniprophet.environment.source_board import Source


@dataclass
class SearchResult:
    """Return type for SearchTool.search()."""

    sources: list[Source]
    cost: float = 0.0


class SearchTool(Protocol):
    """Protocol that any search backend must satisfy."""

    def search(self, query: str, limit: int = 5, **kwargs) -> SearchResult: ...

    def serialize(self) -> dict: ...


_SEARCH_CLASS_MAPPING: dict[str, str] = {
    "brave": "miniprophet.search.brave.BraveSearchTool",
    "perplexity": "miniprophet.search.perplexity.PerplexitySearchTool",
}


def get_search_tool(config: dict) -> SearchTool:
    """Instantiate a search tool from a config dict.

    The 'search_class' key selects the implementation (default: "brave").
    Remaining keys are forwarded as keyword arguments to the constructor.
    """
    config = dict(config)
    class_key = config.pop("search_class", "brave")
    full_path = _SEARCH_CLASS_MAPPING.get(class_key, class_key)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as exc:
        raise ValueError(
            f"Unknown search class: {class_key} (resolved to {full_path}, "
            f"available: {list(_SEARCH_CLASS_MAPPING)})"
        ) from exc
    sig = inspect.signature(cls.__init__)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        accepted = config
    else:
        valid_keys = set(sig.parameters.keys()) - {"self"}
        accepted = {k: v for k, v in config.items() if k in valid_keys}
    return cls(**accepted)


__all__ = ["SearchTool", "SearchResult", "get_search_tool", "Source"]
