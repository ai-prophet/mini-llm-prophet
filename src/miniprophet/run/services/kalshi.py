"""Kalshi prediction market service for mini-llm-prophet."""

from __future__ import annotations

import logging

import requests

from miniprophet.run.services import MarketData

logger = logging.getLogger("miniprophet.services.kalshi")

DEFAULT_API_BASE = (
    "https://api.elections.kalshi.com/trade-api/v2/events/{event_ticker}?with_nested_markets=false"
)


class KalshiService:
    """Fetch market data from the Kalshi public API."""

    def __init__(self, api_base: str = DEFAULT_API_BASE) -> None:
        self._api_base = api_base.rstrip("/")

    def _parse_single_market_dict(self, market: dict) -> tuple:
        market_title = market.get("yes_sub_title") or "no title"
        market_result = {"yes": 1, "no": 0}.get(market.get("result"), None)
        return market_title, market_result

    def fetch(self, event_ticker: str) -> MarketData:
        event_ticker = event_ticker.upper().strip()  # preprocess
        url = self._api_base.format(event_ticker=event_ticker)
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        event, markets = data.get("event", {}), data.get("markets", [])
        title = event.get("title") or event_ticker

        # parse outcomes and ground truth
        outcomes, ground_truth = [], {}
        for market in markets:
            market_title, market_result = self._parse_single_market_dict(market)
            outcomes.append(market_title)
            ground_truth[market_title] = market_result

        metadata = {
            "event_ticker": market.get("ticker", event_ticker),
            "status": market.get("status", ""),
            "last_price": market.get("last_price_dollars", ""),
            "volume": market.get("volume", 0),
            "rules": market.get("rules_primary", ""),
        }

        # if the ground_truth has any None, we simply do not include ground_truth
        if any([v is None for v in ground_truth.values()]):
            logger.warning(
                f"Kalshi event '{event_ticker}' has unresolved ground truth market, skipping ground truth"
            )
            ground_truth = None

        logger.info(f"Fetched Kalshi event '{event_ticker}': {title}")
        return MarketData(
            title=title,
            outcomes=outcomes,
            ground_truth=ground_truth,
            metadata=metadata,
        )
