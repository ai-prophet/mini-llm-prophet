from __future__ import annotations

import pytest

from miniprophet.run.services import get_market_service
from miniprophet.run.services.kalshi import KalshiService


class _Resp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_get_market_service_returns_kalshi() -> None:
    svc = get_market_service("kalshi")
    assert isinstance(svc, KalshiService)


def test_get_market_service_raises_for_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown market service"):
        get_market_service("missing-service")


def test_kalshi_fetch_parses_outcomes_and_unresolved_ground_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "event": {"title": "Election"},
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
    monkeypatch.setattr("miniprophet.run.services.kalshi.requests.get", lambda *a, **k: _Resp(payload))

    result = KalshiService().fetch("evt")

    assert result.title == "Election"
    assert result.outcomes == ["A", "B"]
    assert result.ground_truth is None
    assert result.metadata["event_ticker"] == "EVT-B"
