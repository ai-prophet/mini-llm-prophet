"""
mini-llm-prophet: A minimal LLM forecasting agent scaffolding.

Provides:
- Version numbering
- Protocol definitions for core components (Model, Environment)
"""

__version__ = "0.1.0"

from pathlib import Path
from typing import Any, Protocol

package_dir = Path(__file__).resolve().parent


class Model(Protocol):
    """Protocol for language models."""

    config: Any

    def query(self, messages: list[dict], tools: list[dict]) -> dict: ...

    def format_message(self, **kwargs) -> dict: ...

    def format_observation_messages(self, message: dict, outputs: list[dict]) -> list[dict]: ...

    def serialize(self) -> dict: ...


class Environment(Protocol):
    """Protocol for forecast environments."""

    config: Any

    def execute(self, action: dict) -> dict: ...

    def get_tool_schemas(self) -> list[dict]: ...

    def serialize(self) -> dict: ...


__all__ = [
    "Model",
    "Environment",
    "package_dir",
    "__version__",
]
