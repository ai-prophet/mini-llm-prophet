"""Default context management strategy for the forecasting agent."""

from __future__ import annotations

from miniprophet.utils import display


class SlidingWindowContextManager:
    """Stateful sliding-window context manager.

    Always preserves:
      - messages[0]: system prompt
      - messages[1]: instance/user prompt
    When the message body exceeds `window_size`, older messages are discarded
    and replaced by a single synthetic summary that includes:
      - The cumulative count of all messages ever truncated
      - The current source board state (the "invariant")
      - A log of all search queries issued so far (to avoid repeats)
    """

    def __init__(self, window_size: int = 6) -> None:
        self.window_size = window_size
        self._total_truncated: int = 0
        self._past_queries: list[str] = []

    def record_query(self, query: str) -> None:
        """Called by the environment after each search to track query history."""
        self._past_queries.append(query)

    def manage(
        self, messages: list[dict], *, step: int, board_state: str = "", **kwargs
    ) -> list[dict]:
        if self.window_size <= 0:
            return messages

        preamble = messages[:2]
        body = messages[2:]

        # Strip any previous truncation notice before counting
        body = [m for m in body if not m.get("extra", {}).get("is_truncation_notice")]

        if len(body) <= self.window_size:
            return preamble + body

        newly_removed = len(body) - self.window_size
        self._total_truncated += newly_removed
        kept = body[-self.window_size :]

        lines = [
            f"[Context truncated: {self._total_truncated} earlier messages have been "
            f"omitted across this conversation. The source board and query history "
            f"below reflect all accumulated state.]",
            "",
            board_state,
        ]

        if self._past_queries:
            lines.append("")
            lines.append("--- Search Queries So Far ---")
            for i, q in enumerate(self._past_queries, 1):
                lines.append(f"  {i}. {q}")
            lines.append("(Do not repeat these queries.)")

        truncation_msg = {
            "role": "user",
            "content": "\n".join(lines),
            "extra": {"is_truncation_notice": True},
        }

        display.print_context_truncation(self._total_truncated, self.window_size)
        return preamble + [truncation_msg] + kept
