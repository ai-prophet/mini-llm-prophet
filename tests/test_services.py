from __future__ import annotations

import pytest
import requests

from miniprophet.run.services import get_market_service
from miniprophet.run.services.kalshi import KalshiService
from miniprophet.run.services.polymarket import PolymarketService


class _Resp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _ErrorResp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        raise requests.HTTPError(f"status={self.status_code}", response=self)

    def json(self) -> dict:
        return {}


def test_get_market_service_returns_kalshi() -> None:
    svc = get_market_service("kalshi")
    assert isinstance(svc, KalshiService)


def test_get_market_service_returns_polymarket() -> None:
    svc = get_market_service("polymarket")
    assert isinstance(svc, PolymarketService)


def test_get_market_service_raises_for_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown market service"):
        get_market_service("missing-service")


def test_kalshi_fetch_event_parses_outcomes_and_unresolved_ground_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "event": {"title": "Election", "event_ticker": "EVT"},
        "markets": [
            {"yes_sub_title": "A", "result": "yes", "ticker": "EVT-A", "status": "active"},
            {
                "yes_sub_title": "B",
                "result": None,
                "ticker": "EVT-B",
                "status": "active",
                "last_price_dollars": 0.42,
                "volume": 123,
            },
        ],
    }
    monkeypatch.setattr(
        "miniprophet.run.services.kalshi.requests.get", lambda *a, **k: _Resp(payload)
    )

    result = KalshiService().fetch("evt")

    assert result.title == "Election"
    assert result.outcomes == ["A", "B"]
    assert result.ground_truth is None
    assert result.metadata["event_ticker"] == "EVT"
    assert result.metadata["market_count"] == 2
    assert result.metadata["volume"] == pytest.approx(123.0)


def test_kalshi_fetch_market_returns_yes_no_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "market": {
            "title": "Will it rain in NYC tomorrow?",
            "result": "no",
            "ticker": "RAIN-NYC-20260305",
            "event_ticker": "WXRAIN",
            "status": "settled",
            "last_price_dollars": 0.0,
            "volume": 150,
        }
    }
    monkeypatch.setattr(
        "miniprophet.run.services.kalshi.requests.get", lambda *a, **k: _Resp(payload)
    )

    result = KalshiService().fetch("rain-nyc-20260305", ticker_type="market")

    assert result.title == "Will it rain in NYC tomorrow?"
    assert result.outcomes == ["Yes", "No"]
    assert result.ground_truth == {"Yes": 0, "No": 1}
    assert result.metadata["market_ticker"] == "RAIN-NYC-20260305"
    assert result.metadata["event_ticker"] == "WXRAIN"


def test_kalshi_fetch_auto_falls_back_to_market(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called_urls: list[str] = []
    market_payload = {
        "market": {
            "title": "Will it rain in NYC tomorrow?",
            "result": "yes",
            "ticker": "RAIN-NYC-20260305",
            "event_ticker": "WXRAIN",
        }
    }

    def _fake_get(url: str, *args, **kwargs) -> _Resp | _ErrorResp:
        called_urls.append(url)
        if "/events/" in url:
            return _ErrorResp(404)
        return _Resp(market_payload)

    monkeypatch.setattr("miniprophet.run.services.kalshi.requests.get", _fake_get)

    result = KalshiService().fetch("rain-nyc-20260305")

    assert "/events/RAIN-NYC-20260305" in called_urls[0]
    assert "/markets/RAIN-NYC-20260305" in called_urls[1]
    assert result.outcomes == ["Yes", "No"]
    assert result.metadata["entity"] == "market"


def test_polymarket_fetch_event_by_id_parses_market_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called_urls: list[str] = []
    payload = {
        "id": "4242",
        "slug": "who-wins-2028",
        "title": "Who wins in 2028?",
        "markets": [
            {
                "id": "m1",
                "question": "Candidate A",
                "outcomes": '["Yes","No"]',
                "outcomePrices": '["1","0"]',
                "closed": True,
                "umaResolutionStatus": "resolved",
            },
            {
                "id": "m2",
                "question": "Candidate B",
                "outcomes": '["Yes","No"]',
                "outcomePrices": '["0","1"]',
                "closed": True,
                "umaResolutionStatus": "resolved",
            },
        ],
    }

    def _fake_get(url: str, *args, **kwargs) -> _Resp:
        called_urls.append(url)
        return _Resp(payload)

    monkeypatch.setattr("miniprophet.run.services.polymarket.requests.get", _fake_get)

    result = PolymarketService().fetch("4242", entity="event", identifier_type="id")

    assert called_urls == ["https://gamma-api.polymarket.com/events/4242"]
    assert result.title == "Who wins in 2028?"
    assert result.outcomes == ["Candidate A", "Candidate B"]
    assert result.ground_truth == {"Candidate A": 1, "Candidate B": 0}
    assert result.metadata["event_id"] == "4242"


def test_polymarket_fetch_market_by_slug_parses_ground_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called_urls: list[str] = []
    payload = {
        "id": "m1",
        "slug": "will-fed-cut-rates-by-june",
        "question": "Will the Fed cut rates by June?",
        "outcomes": '["Yes","No"]',
        "outcomePrices": '["1","0"]',
        "closed": True,
        "umaResolutionStatus": "resolved",
        "volumeNum": 12000.5,
        "liquidityNum": 800.1,
    }

    def _fake_get(url: str, *args, **kwargs) -> _Resp:
        called_urls.append(url)
        return _Resp(payload)

    monkeypatch.setattr("miniprophet.run.services.polymarket.requests.get", _fake_get)

    result = PolymarketService().fetch(
        "will-fed-cut-rates-by-june",
        entity="market",
        identifier_type="slug",
    )

    assert called_urls == [
        "https://gamma-api.polymarket.com/markets/slug/will-fed-cut-rates-by-june"
    ]
    assert result.title == "Will the Fed cut rates by June?"
    assert result.outcomes == ["Yes", "No"]
    assert result.ground_truth == {"Yes": 1, "No": 0}
    assert result.metadata["market_slug"] == "will-fed-cut-rates-by-june"


def test_polymarket_fetch_auto_falls_back_to_market(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called_urls: list[str] = []
    market_payload = {
        "id": "m1",
        "slug": "will-fed-cut-rates-by-june",
        "question": "Will the Fed cut rates by June?",
        "outcomes": '["Yes","No"]',
        "outcomePrices": '["0.7","0.3"]',
    }

    def _fake_get(url: str, *args, **kwargs) -> _Resp | _ErrorResp:
        called_urls.append(url)
        if "/events/slug/" in url:
            return _ErrorResp(404)
        return _Resp(market_payload)

    monkeypatch.setattr("miniprophet.run.services.polymarket.requests.get", _fake_get)

    result = PolymarketService().fetch("will-fed-cut-rates-by-june", identifier_type="slug")

    assert called_urls == [
        "https://gamma-api.polymarket.com/events/slug/will-fed-cut-rates-by-june",
        "https://gamma-api.polymarket.com/markets/slug/will-fed-cut-rates-by-june",
    ]
    assert result.outcomes == ["Yes", "No"]
    assert result.metadata["entity"] == "market"
