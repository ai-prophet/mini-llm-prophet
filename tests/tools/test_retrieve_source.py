from __future__ import annotations

import asyncio

from miniprophet.environment.source_registry import Source, SourceRegistry
from miniprophet.tools.retrieve_source import RetrieveSourceTool


def test_retrieve_valid_source() -> None:
    registry = SourceRegistry()
    src = Source(
        url="https://example.com", title="Example", snippet="full content here", date="2026-01-01"
    )
    asyncio.run(registry.add(src))

    tool = RetrieveSourceTool(registry=registry)
    assert tool.name == "retrieve_source"
    output = asyncio.run(tool.execute({"source_id": "S1"}))
    assert "full content here" in output["output"]
    assert 'id="S1"' in output["output"]
    assert not output.get("error")


def test_retrieve_unknown_source() -> None:
    tool = RetrieveSourceTool(registry=SourceRegistry())
    output = asyncio.run(tool.execute({"source_id": "S99"}))
    assert output["error"] is True
    assert "unknown source_id" in output["output"]


def test_retrieve_source_converts_int_id() -> None:
    registry = SourceRegistry()
    asyncio.run(registry.add(Source(url="u", title="t", snippet="s")))
    tool = RetrieveSourceTool(registry=registry)
    output = asyncio.run(tool.execute({"source_id": 1}))
    assert 'id="S1"' in output["output"]


def test_retrieve_source_empty_id() -> None:
    tool = RetrieveSourceTool(registry=SourceRegistry())
    output = asyncio.run(tool.execute({"source_id": ""}))
    assert output["error"] is True
    assert "'source_id' is required" in output["output"]
