"""Evaluation metrics for mini-llm-prophet forecasts."""

from __future__ import annotations

from typing import Protocol


class Metric(Protocol):
    """Protocol for forecast evaluation metrics."""

    name: str

    def compute(self, probabilities: dict[str, float], ground_truth: dict[str, int]) -> float: ...


class BrierScore:
    """Mean squared error between predicted probabilities and binary ground truth."""

    name = "brier_score"

    def compute(self, probabilities: dict[str, float], ground_truth: dict[str, int]) -> float:
        total = 0.0
        n = 0
        for outcome, truth in ground_truth.items():
            pred = probabilities.get(outcome, 0.0)
            total += (pred - truth) ** 2
            n += 1
        return total / n if n else 0.0


_METRIC_REGISTRY: dict[str, Metric] = {}


def register_metric(metric: Metric) -> None:
    _METRIC_REGISTRY[metric.name] = metric


def get_metrics() -> dict[str, Metric]:
    return dict(_METRIC_REGISTRY)


register_metric(BrierScore())


def validate_ground_truth(outcomes: list[str], ground_truth: dict[str, int]) -> None:
    """Raise ValueError if ground_truth doesn't match outcomes."""
    gt_keys = set(ground_truth.keys())
    outcome_set = set(outcomes)
    missing = outcome_set - gt_keys
    extra = gt_keys - outcome_set
    errors: list[str] = []
    if missing:
        errors.append(f"Missing outcomes in ground_truth: {missing}")
    if extra:
        errors.append(f"Unknown outcomes in ground_truth: {extra}")
    for key, val in ground_truth.items():
        if val not in (0, 1):
            errors.append(f"ground_truth['{key}'] must be 0 or 1, got {val}")
    if errors:
        raise ValueError("Invalid ground_truth: " + "; ".join(errors))


def evaluate_submission(
    probabilities: dict[str, float], ground_truth: dict[str, int]
) -> dict[str, float]:
    """Run all registered metrics and return {metric_name: score}."""
    return {
        name: metric.compute(probabilities, ground_truth)
        for name, metric in _METRIC_REGISTRY.items()
    }
