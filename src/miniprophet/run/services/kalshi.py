"""Kalshi prediction market service for mini-llm-prophet."""

from __future__ import annotations

import logging

import requests

from miniprophet.run.services import MarketData

logger = logging.getLogger("miniprophet.services.kalshi")

DEFAULT_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiService:
    """Fetch market data from the Kalshi public API."""

    def __init__(self, api_base: str = DEFAULT_API_BASE) -> None:
        self._api_base = api_base.rstrip("/")

    def fetch(self, ticker: str) -> MarketData:
        url = f"{self._api_base}/markets/{ticker}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        market = data.get("market", {})

        title = market.get("title") or market.get("yes_sub_title") or ticker
        outcomes = ["Yes", "No"]

        result_str = market.get("result", "")
        ground_truth: dict[str, int] | None = None
        if result_str == "yes":
            ground_truth = {"Yes": 1, "No": 0}
        elif result_str == "no":
            ground_truth = {"Yes": 0, "No": 1}

        metadata = {
            "ticker": market.get("ticker", ticker),
            "event_ticker": market.get("event_ticker", ""),
            "status": market.get("status", ""),
            "last_price": market.get("last_price_dollars", ""),
            "volume": market.get("volume", 0),
            "rules": market.get("rules_primary", ""),
        }

        logger.info(f"Fetched Kalshi market '{ticker}': {title}")
        return MarketData(
            title=title,
            outcomes=outcomes,
            ground_truth=ground_truth,
            metadata=metadata,
        )
