"""Subagents: lightweight agents spawned by the main agent for delegated work."""

from miniprophet.subagents.base import SubagentBase
from miniprophet.subagents.source_reading import SourceReadingAgent, SourceReadingResult
from miniprophet.subagents.subproblem import SubproblemAgent, SubproblemResult

__all__ = [
    "SubagentBase",
    "SourceReadingAgent",
    "SourceReadingResult",
    "SubproblemAgent",
    "SubproblemResult",
]
