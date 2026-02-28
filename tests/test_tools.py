from __future__ import annotations

import pytest

from miniprophet.environment.source_board import Source, SourceBoard
from miniprophet.exceptions import SearchAuthError, SearchNetworkError, Submitted
from miniprophet.tools.search_tool import SearchForecastTool
from miniprophet.tools.source_board_tools import AddSourceTool, EditNoteTool
from miniprophet.tools.submit import SubmitTool

from conftest import DummySearchTool


def test_search_tool_execute_success_assigns_ids(two_sources: list[Source]) -> None:
    registry: dict[str, Source] = {}
    backend = DummySearchTool(two_sources, cost=0.015)
    tool = SearchForecastTool(backend, registry, search_limit=5)

    output = tool.execute({"query": "nba finals"})

    assert output["search_cost"] == pytest.approx(0.015)
    assert [sid for sid, _ in output["search_results"]] == ["S1", "S2"]
    assert "<search_results count=\"2\">" in output["output"]
    assert registry["S1"].title == "A"


def test_search_tool_execute_rejects_missing_query() -> None:
    tool = SearchForecastTool(DummySearchTool(), {})
    output = tool.execute({"query": "   "})
    assert output["error"] is True
    assert "'query' is required" in output["output"]


def test_search_tool_execute_handles_backend_error() -> None:
    tool = SearchForecastTool(
        DummySearchTool(error=SearchNetworkError("timeout")),
        {},
        search_limit=2,
    )
    output = tool.execute({"query": "x"})
    assert output["error"] is True
    assert "Search failed" in output["output"]


def test_search_tool_execute_bubbles_auth_error() -> None:
    tool = SearchForecastTool(DummySearchTool(error=SearchAuthError("bad key")), {})
    with pytest.raises(SearchAuthError):
        tool.execute({"query": "x"})


def test_add_source_tool_adds_valid_source(dummy_source: Source) -> None:
    board = SourceBoard()
    registry = {"S1": dummy_source}
    tool = AddSourceTool(source_registry=registry, board=board, outcomes=["Yes", "No"])

    output = tool.execute({"source_id": "s1", "note": "good", "reaction": {"Yes": "positive"}})

    assert "added to board as #1" in output["output"]
    assert len(board.serialize()) == 1


def test_add_source_tool_rejects_unknown_source_id() -> None:
    tool = AddSourceTool(source_registry={}, board=SourceBoard(), outcomes=["Yes", "No"])
    output = tool.execute({"source_id": "S9", "note": "x"})
    assert output["error"] is True
    assert "unknown source_id" in output["output"]


def test_edit_note_tool_updates_note_and_reaction() -> None:
    board = SourceBoard()
    board.add(Source(url="u", title="t", snippet="s"), "old", source_id="S1")
    tool = EditNoteTool(board=board, outcomes=["Yes", "No"])

    output = tool.execute({"board_id": 1, "new_note": "new", "reaction": {"No": "negative"}})

    assert output["output"] == "Note for #1 updated."
    assert board.get(1).reaction == {"No": "negative"}


def test_edit_note_tool_rejects_missing_entry() -> None:
    tool = EditNoteTool(board=SourceBoard(), outcomes=["Yes", "No"])
    output = tool.execute({"board_id": 1, "new_note": "x"})
    assert output["error"] is True
    assert "no board entry" in output["output"]


def test_submit_tool_raises_submitted_on_valid_probs(dummy_source: Source) -> None:
    board = SourceBoard()
    board.add(dummy_source, "note", source_id="S1")
    tool = SubmitTool(outcomes=["Yes", "No"], board=board)

    with pytest.raises(Submitted) as exc:
        tool.execute({"probabilities": {"Yes": 0.7, "No": 0.3}})

    payload = exc.value.messages[0]
    assert payload["extra"]["exit_status"] == "submitted"
    assert payload["extra"]["submission"] == {"Yes": 0.7, "No": 0.3}


def test_submit_tool_rejects_invalid_probabilities() -> None:
    tool = SubmitTool(outcomes=["Yes", "No"], board=SourceBoard())
    output = tool.execute({"probabilities": {"Yes": 2.0}})

    assert output["error"] is True
    assert "Missing probability" in output["output"]
    assert "must be a number between 0 and 1" in output["output"]
