"""Trajectory recording for per-step observability and replay.

The TrajectoryRecorder maintains a global pool of all messages ever created
during an agent run, along with a step log that records which messages
formed the input/output of each query() call. This allows full reconstruction
of what the LLM saw at every step, even when a context manager truncates
or replaces messages between steps.
"""

from __future__ import annotations

from collections import Counter

# We use different prefixes for different message roles to make the trajectory easier to read.
TRAJ_ROLE_MAPPING = {
    "system": "S",
    "user": "U",
    "assistant": "A",
    "tool": "T",
    "other": "O",
}


class TrajectoryRecorder:
    """Records per-step input/output message references into a global pool.

    Messages are deduplicated by object identity (``id(msg)``), so the same
    dict object appearing across multiple steps is stored only once.
    """

    def __init__(self) -> None:
        self._pool: list[dict] = []
        self._role_counts: Counter[str] = Counter()
        self._identity_to_key: dict[int, str] = {}
        self._steps: list[dict] = []

    @property
    def n_steps(self) -> int:
        return len(self._steps)

    def _derive_and_increment_key(self, role: str) -> str:
        if role not in TRAJ_ROLE_MAPPING:
            role = "other"
        key = f"{TRAJ_ROLE_MAPPING[role]}{self._role_counts[role]}"
        self._role_counts[role] += 1
        return key

    def register(self, *messages: dict) -> list[str]:
        """Add messages to the pool if not already present. Returns their keys."""
        keys: list[str] = []
        for msg in messages:
            obj_id = id(msg)
            if obj_id not in self._identity_to_key:
                key = self._derive_and_increment_key(msg["role"])
                self._identity_to_key[obj_id] = key
                self._pool.append(msg)
            keys.append(self._identity_to_key[obj_id])
        return keys

    def record_step(self, input_messages: list[dict], output_message: dict) -> None:
        """Record a single query step: the full input list and the model's output."""
        input_keys = self.register(*input_messages)
        output_keys = self.register(output_message)
        self._steps.append(
            {
                "input": input_keys,
                "output": output_keys[0],
            }
        )

    def serialize(self) -> dict:
        """Return the trajectory in a format suitable for JSON serialization.

        Returns a dict with:
        - ``messages``: list of ``{"key": "S0", "message": {...}}`` entries
        - ``steps``: list of ``{"input": ["S0", ...], "output": "A1"}`` entries
        """
        messages = []
        for msg in self._pool:
            messages.append(
                {
                    "key": self._identity_to_key[id(msg)],
                    "message": msg,
                }
            )
        return {
            "messages": messages,
            "steps": self._steps,
        }
