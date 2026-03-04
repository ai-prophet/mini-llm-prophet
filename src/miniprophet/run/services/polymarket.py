"""Polymarket prediction market service for mini-prophet."""

from __future__ import annotations

import json
import logging
from urllib.parse import quote

import requests

from miniprophet.run.services import MarketData

logger = logging.getLogger("miniprophet.services.polymarket")

DEFAULT_API_BASE = "https://gamma-api.polymarket.com"


class PolymarketService:
    """Fetch event or market data from Polymarket's Gamma API."""

    def __init__(self, api_base: str = DEFAULT_API_BASE) -> None:
        self._api_base = api_base.rstrip("/")

    @staticmethod
    def _parse_str_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return [text]
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        return []

    @staticmethod
    def _parse_float_list(value: object) -> list[float]:
        values = PolymarketService._parse_str_list(value)
        parsed: list[float] = []
        for v in values:
            try:
                parsed.append(float(v))
            except ValueError:
                return []
        return parsed

    @staticmethod
    def _pick_identifier_type(identifier: str, identifier_type: str) -> str:
        normalized = identifier_type.strip().lower()
        if normalized == "auto":
            return "id" if identifier.strip().isdigit() else "slug"
        if normalized in {"id", "slug"}:
            return normalized
        raise ValueError(
            f"Unknown Polymarket identifier_type '{identifier_type}'. Expected 'id', 'slug', or 'auto'."
        )

    def _build_url(self, resource: str, identifier: str, identifier_type: str) -> str:
        kind = self._pick_identifier_type(identifier, identifier_type)
        token = quote(identifier.strip())
        if kind == "id":
            return f"{self._api_base}/{resource}/{token}"
        return f"{self._api_base}/{resource}/slug/{token}"

    @staticmethod
    def _infer_winner_label(market: dict) -> str | None:
        outcomes = PolymarketService._parse_str_list(market.get("outcomes"))
        prices = PolymarketService._parse_float_list(market.get("outcomePrices"))
        if not outcomes or len(outcomes) != len(prices):
            return None

        uma_status = str(market.get("umaResolutionStatus", "")).lower()
        if (
            not market.get("closed")
            and "resolved" not in uma_status
            and "settled" not in uma_status
        ):
            return None

        winner_index: int | None = None
        for idx, price in enumerate(prices):
            if price >= 0.999:
                if winner_index is not None:
                    return None
                winner_index = idx
            elif price > 0.001:
                return None

        if winner_index is None:
            return None
        return outcomes[winner_index]

    @staticmethod
    def _is_not_found_error(exc: Exception) -> bool:
        if not isinstance(exc, requests.HTTPError):
            return False
        return getattr(exc.response, "status_code", None) == 404

    def fetch(
        self,
        identifier: str,
        *,
        entity: str = "auto",
        identifier_type: str = "auto",
    ) -> MarketData:
        entity = entity.strip().lower()
        if entity == "event":
            return self.fetch_event(identifier, identifier_type=identifier_type)
        if entity == "market":
            return self.fetch_market(identifier, identifier_type=identifier_type)
        if entity == "auto":
            try:
                return self.fetch_event(identifier, identifier_type=identifier_type)
            except Exception as event_exc:
                if not self._is_not_found_error(event_exc):
                    raise
            try:
                return self.fetch_market(identifier, identifier_type=identifier_type)
            except Exception as market_exc:
                if self._is_not_found_error(market_exc):
                    raise ValueError(
                        f"Polymarket identifier '{identifier}' was not found as an event or market."
                    ) from market_exc
                raise
        raise ValueError(
            f"Unknown Polymarket entity '{entity}'. Expected 'auto', 'event', or 'market'."
        )

    def fetch_event(self, identifier: str, *, identifier_type: str = "auto") -> MarketData:
        url = self._build_url("events", identifier, identifier_type)
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        event = resp.json()
        markets = event.get("markets") or []

        title = event.get("title") or str(identifier).strip()
        outcomes: list[str] = []
        ground_truth: dict[str, int] = {}
        for market in markets:
            market_title = (
                market.get("question")
                or market.get("title")
                or market.get("slug")
                or str(market.get("id", ""))
            )
            if not market_title:
                market_title = "unknown-market"
            outcomes.append(market_title)

            winner = self._infer_winner_label(market)
            if winner is None or winner.lower() not in {"yes", "no"}:
                ground_truth = {}
                break
            ground_truth[market_title] = 1 if winner.lower() == "yes" else 0

        metadata = {
            "entity": "event",
            "event_id": event.get("id"),
            "event_slug": event.get("slug"),
            "status": event.get("status", ""),
            "closed": event.get("closed"),
            "active": event.get("active"),
            "market_count": len(markets),
            "volume": event.get("volume"),
            "liquidity": event.get("liquidity"),
        }

        logger.info(f"Fetched Polymarket event '{identifier}': {title}")
        return MarketData(
            title=title,
            outcomes=outcomes,
            ground_truth=ground_truth or None,
            metadata=metadata,
        )

    def fetch_market(self, identifier: str, *, identifier_type: str = "auto") -> MarketData:
        url = self._build_url("markets", identifier, identifier_type)
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        market = resp.json()

        title = market.get("question") or market.get("title") or str(identifier).strip()
        outcomes = self._parse_str_list(market.get("outcomes"))
        if not outcomes:
            outcomes = ["Yes", "No"]

        winner = self._infer_winner_label(market)
        ground_truth = None
        if winner is not None:
            ground_truth = {outcome: int(outcome == winner) for outcome in outcomes}

        prices = self._parse_float_list(market.get("outcomePrices"))
        last_price = prices[0] if prices else ""
        metadata = {
            "entity": "market",
            "market_id": market.get("id"),
            "market_slug": market.get("slug"),
            "event_id": market.get("eventId"),
            "condition_id": market.get("conditionId"),
            "status": market.get("status", ""),
            "closed": market.get("closed"),
            "active": market.get("active"),
            "last_price": last_price,
            "volume": market.get("volumeNum", market.get("volume")),
            "liquidity": market.get("liquidityNum", market.get("liquidity")),
            "resolution_status": market.get("umaResolutionStatus", ""),
        }

        logger.info(f"Fetched Polymarket market '{identifier}': {title}")
        return MarketData(
            title=title,
            outcomes=outcomes,
            ground_truth=ground_truth,
            metadata=metadata,
        )
