"""
mini-llm-prophet: A minimal LLM forecasting agent scaffolding.

Provides:
- Version numbering
- Protocol definitions for core components (Model, Environment, Tool)
"""

__version__ = "0.1.4"

from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

package_dir = Path(__file__).resolve().parent
global_dir = package_dir.parent.parent
load_dotenv(dotenv_path=global_dir / ".env")


class Model(Protocol):
    """Protocol for language models."""

    config: Any

    def query(self, messages: list[dict], tools: list[dict]) -> dict: ...

    def format_message(self, **kwargs) -> dict: ...

    def format_observation_messages(self, message: dict, outputs: list[dict]) -> list[dict]: ...

    def serialize(self) -> dict: ...


class Tool(Protocol):
    """Protocol for modular forecast tools."""

    @property
    def name(self) -> str: ...

    def get_schema(self) -> dict: ...

    def execute(self, args: dict) -> dict: ...

    def display(self, output: dict) -> None: ...


class Environment(Protocol):
    """Protocol for forecast environments."""

    config: Any
    _tools: dict[str, Tool]

    def execute(self, action: dict) -> dict: ...

    def get_tool_schemas(self) -> list[dict]: ...

    def serialize(self) -> dict: ...


class ContextManager(Protocol):
    """Protocol for managing the message context between steps."""

    def manage(self, messages: list[dict], *, step: int, **kwargs) -> list[dict]: ...

    def display(self) -> None: ...


__all__ = [
    "Model",
    "Tool",
    "Environment",
    "ContextManager",
    "package_dir",
    "__version__",
]
