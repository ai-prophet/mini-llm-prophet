"""Kalshi prediction market service for mini-prophet."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import requests

from miniprophet.run.services import MarketData

logger = logging.getLogger("miniprophet.services.kalshi")

DEFAULT_EVENT_API_BASE = (
    "https://api.elections.kalshi.com/trade-api/v2/events/{event_ticker}?with_nested_markets=false"
)
DEFAULT_MARKET_API_BASE = "https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}"


class KalshiService:
    """Fetch event or market data from the Kalshi public API."""

    def __init__(
        self,
        event_api_base: str = DEFAULT_EVENT_API_BASE,
        market_api_base: str = DEFAULT_MARKET_API_BASE,
    ) -> None:
        self._event_api_base = event_api_base.rstrip("/")
        self._market_api_base = market_api_base.rstrip("/")

    def _parse_single_market_dict(self, market: dict) -> tuple[str, int | None]:
        market_title = market.get("yes_sub_title") or "no title"
        market_result = self._parse_binary_result(market.get("result"))
        return market_title, market_result

    @staticmethod
    def _parse_binary_result(result: str | None) -> int | None:
        return {"yes": 1, "no": 0}.get((result or "").strip().lower())

    @staticmethod
    def _sum_volume(markets: Iterable[dict]) -> float:
        total = 0.0
        for market in markets:
            value = market.get("volume")
            if isinstance(value, int | float):
                total += float(value)
        return total

    @staticmethod
    def _is_not_found_error(exc: Exception) -> bool:
        if not isinstance(exc, requests.HTTPError):
            return False
        return getattr(exc.response, "status_code", None) == 404

    def fetch(self, ticker: str, ticker_type: str = "auto") -> MarketData:
        ticker_type = ticker_type.strip().lower()
        if ticker_type in {"event", "event_ticker"}:
            return self.fetch_event(ticker)
        if ticker_type in {"market", "market_ticker"}:
            return self.fetch_market(ticker)
        if ticker_type == "auto":
            try:
                return self.fetch_event(ticker)
            except Exception as event_exc:
                if not self._is_not_found_error(event_exc):
                    raise
            try:
                return self.fetch_market(ticker)
            except Exception as market_exc:
                if self._is_not_found_error(market_exc):
                    raise ValueError(
                        f"Kalshi identifier '{ticker}' was not found as an event ticker or market ticker."
                    ) from market_exc
                raise
        raise ValueError(
            f"Unknown Kalshi ticker_type '{ticker_type}'. Expected 'auto', 'event', or 'market'."
        )

    def fetch_event(self, event_ticker: str) -> MarketData:
        event_ticker = event_ticker.upper().strip()  # preprocess
        url = self._event_api_base.format(event_ticker=event_ticker)
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        event = data.get("event", {})
        markets = event.get("markets") or data.get("markets", [])
        title = event.get("title") or event_ticker

        # parse outcomes and ground truth
        outcomes, ground_truth = [], {}
        for market in markets:
            market_title, market_result = self._parse_single_market_dict(market)
            outcomes.append(market_title)
            ground_truth[market_title] = market_result

        primary_market = markets[0] if markets else {}
        metadata = {
            "entity": "event",
            "event_ticker": event.get("event_ticker", event_ticker),
            "status": event.get("status", ""),
            "market_count": len(markets),
            "market_tickers": [m.get("ticker", "") for m in markets if m.get("ticker")],
            "last_price": primary_market.get("last_price_dollars", ""),
            "volume": self._sum_volume(markets),
            "rules": primary_market.get("rules_primary", ""),
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

    def fetch_market(self, market_ticker: str) -> MarketData:
        market_ticker = market_ticker.upper().strip()
        url = self._market_api_base.format(ticker=market_ticker)
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        market = data.get("market", {})
        if not market:
            raise ValueError(f"Kalshi market '{market_ticker}' not found in response")

        title = market.get("title") or market.get("yes_sub_title") or market_ticker
        result = self._parse_binary_result(market.get("result"))
        ground_truth = None
        if result is not None:
            ground_truth = {"Yes": result, "No": 1 - result}

        metadata = {
            "entity": "market",
            "event_ticker": market.get("event_ticker", ""),
            "market_ticker": market.get("ticker", market_ticker),
            "status": market.get("status", ""),
            "last_price": market.get("last_price_dollars", ""),
            "volume": market.get("volume", 0),
            "rules": market.get("rules_primary", ""),
            "yes_sub_title": market.get("yes_sub_title", ""),
        }

        logger.info(
            f"Fetched Kalshi market '{metadata['market_ticker']}' (event {metadata['event_ticker']})"
        )
        return MarketData(
            title=title,
            outcomes=["Yes", "No"],
            ground_truth=ground_truth,
            metadata=metadata,
        )
