"""Model factory and global cost tracking for mini-llm-prophet."""

from __future__ import annotations

import copy
import importlib
import os
import threading

from miniprophet import Model


class GlobalModelStats:
    """Thread-safe global model cost/call tracker with optional limits."""

    def __init__(self) -> None:
        self._cost = 0.0
        self._n_calls = 0
        self._lock = threading.Lock()
        self.cost_limit = float(os.getenv("MINIPROPHET_GLOBAL_COST_LIMIT", "0"))

    def add(self, cost: float) -> None:
        with self._lock:
            self._cost += cost
            self._n_calls += 1
        if 0 < self.cost_limit < self._cost:
            raise RuntimeError(f"Global model cost limit exceeded: ${self._cost:.4f}")

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def n_calls(self) -> int:
        return self._n_calls


GLOBAL_MODEL_STATS = GlobalModelStats()


_MODEL_CLASS_MAPPING: dict[str, str] = {
    "openrouter": "miniprophet.models.openrouter.OpenRouterModel",
    "litellm": "miniprophet.models.litellm.LitellmModel",
}


def get_model(config: dict | None = None) -> Model:
    """Instantiate a model from a config dict.

    Resolves model_name from config or MINIPROPHET_MODEL_NAME env var.
    Selects model class via 'model_class' key or defaults to OpenRouterModel.
    """
    if config is None:
        config = {}
    config = copy.deepcopy(config)

    if not config.get("model_name"):
        config["model_name"] = os.getenv("MINIPROPHET_MODEL_NAME", "")
    if not config["model_name"]:
        raise ValueError("No model name set. Pass --model or set MINIPROPHET_MODEL_NAME.")

    model_class_key = config.pop("model_class", "openrouter")
    full_path = _MODEL_CLASS_MAPPING.get(model_class_key, model_class_key)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as exc:
        raise ValueError(
            f"Unknown model class: {model_class_key} (resolved to {full_path}, "
            f"available: {list(_MODEL_CLASS_MAPPING)})"
        ) from exc

    return cls(**config)


__all__ = ["get_model", "GLOBAL_MODEL_STATS", "GlobalModelStats"]
