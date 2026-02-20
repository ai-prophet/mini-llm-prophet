"""Prediction market service layer for mini-llm-prophet."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class MarketData:
    """Standardized market data returned by any MarketService."""

    title: str
    outcomes: list[str]
    ground_truth: dict[str, int] | None = None
    metadata: dict = field(default_factory=dict)


class MarketService(Protocol):
    """Protocol for prediction market data fetchers."""

    def fetch(self, ticker: str) -> MarketData: ...


_SERVICE_MAPPING: dict[str, str] = {
    "kalshi": "miniprophet.run.services.kalshi.KalshiService",
}


def get_market_service(name: str = "kalshi", **kwargs) -> MarketService:
    full_path = _SERVICE_MAPPING.get(name, name)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as exc:
        raise ValueError(
            f"Unknown market service: {name} (resolved to {full_path}, "
            f"available: {list(_SERVICE_MAPPING)})"
        ) from exc
    return cls(**kwargs)
