"""Task validation and JSONL -> ForecastProblem loading."""

from __future__ import annotations

import json
from pathlib import Path

from miniprophet.eval.datasets.schema import ForecastTaskRow, row_to_problem
from miniprophet.eval.types import ForecastProblem


def load_problems(path: Path, offset: int = 0) -> list[ForecastProblem]:
    """Load and validate forecast problems from a JSONL file."""
    problems: list[ForecastProblem] = []
    seen_ids: set[str] = set()
    auto_idx = 0

    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

        try:
            row = ForecastTaskRow.model_validate(payload)
        except Exception as exc:
            raise ValueError(f"Line {line_no}: invalid task schema: {exc}") from exc

        task_id = row.task_id or f"task_{auto_idx}"
        if row.task_id is None:
            auto_idx += 1

        if task_id in seen_ids:
            raise ValueError(f"Line {line_no}: duplicate task_id '{task_id}'.")
        seen_ids.add(task_id)

        problems.append(row_to_problem(row, task_id=task_id, offset=offset))

    return problems
