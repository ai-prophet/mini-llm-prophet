"""Trajectory recording for per-step observability and replay.

The TrajectoryRecorder maintains a global pool of all messages ever created
during an agent run, along with a step log that records which messages
formed the input/output of each query() call. This allows full reconstruction
of what the LLM saw at every step, even when a context manager truncates
or replaces messages between steps.
"""

from __future__ import annotations


class TrajectoryRecorder:
    """Records per-step input/output message references into a global pool.

    Messages are deduplicated by object identity (``id(msg)``), so the same
    dict object appearing across multiple steps is stored only once.
    """

    def __init__(self) -> None:
        self._pool: list[dict] = []
        self._identity_to_key: dict[int, str] = {}
        self._steps: list[dict] = []

    @property
    def n_steps(self) -> int:
        return len(self._steps)

    def register(self, *messages: dict) -> list[str]:
        """Add messages to the pool if not already present. Returns their keys."""
        keys: list[str] = []
        for msg in messages:
            obj_id = id(msg)
            if obj_id not in self._identity_to_key:
                key = f"m{len(self._pool)}"
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
        - ``messages``: list of ``{"key": "m0", "message": {...}}`` entries
        - ``steps``: list of ``{"input": ["m0", ...], "output": "m3"}`` entries
        """
        messages = []
        for i, msg in enumerate(self._pool):
            messages.append(
                {
                    "key": f"m{i}",
                    "message": msg,
                }
            )
        return {
            "messages": messages,
            "steps": self._steps,
        }
