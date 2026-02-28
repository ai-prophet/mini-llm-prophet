from __future__ import annotations

import pytest

from miniprophet.utils.metrics import evaluate_submission, validate_ground_truth
from miniprophet.utils.serialize import UNSET, recursive_merge


def test_recursive_merge_merges_nested_and_skips_unset() -> None:
    merged = recursive_merge(
        {"a": 1, "nested": {"x": 1, "y": UNSET}},
        {"nested": {"y": 2}, "b": 3},
    )
    assert merged == {"a": 1, "b": 3, "nested": {"x": 1, "y": 2}}


def test_validate_ground_truth_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="Missing outcomes"):
        validate_ground_truth(["Yes", "No"], {"Yes": 1})


def test_evaluate_submission_returns_brier_score() -> None:
    result = evaluate_submission({"Yes": 0.8, "No": 0.2}, {"Yes": 1, "No": 0})
    assert "brier_score" in result
    assert result["brier_score"] == pytest.approx(0.04)
