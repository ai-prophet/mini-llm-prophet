"""BatchForecastAgent: lightweight agent subclass for batch runs.

Adds rate-limit coordination (global pause/resume) and progress
reporting hooks on top of DefaultForecastAgent. No CLI/Rich display.
"""

from __future__ import annotations

import logging
import threading

from miniprophet import ContextManager, Environment, Model
from miniprophet.agent.default import AgentConfig, DefaultForecastAgent
from miniprophet.run.batch_progress import BatchProgressManager

logger = logging.getLogger("miniprophet.agent.batch")


class RateLimitCoordinator:
    """Shared pause/resume gate for all batch workers.

    When any worker signals a rate limit, all workers block on
    ``wait_if_paused()`` until the backoff period expires.
    """

    def __init__(self, backoff_seconds: float = 60) -> None:
        self._resume = threading.Event()
        self._resume.set()
        self._lock = threading.Lock()
        self._backoff = backoff_seconds

    def wait_if_paused(self) -> None:
        self._resume.wait()

    def signal_rate_limit(self, backoff_seconds: float | None = None) -> None:
        secs = backoff_seconds if backoff_seconds is not None else self._backoff
        with self._lock:
            if self._resume.is_set():
                logger.warning("Rate limit detected -- pausing all workers for %.0fs", secs)
                self._resume.clear()
                threading.Timer(secs, self._resume.set).start()


class BatchForecastAgent(DefaultForecastAgent):
    """Agent variant for batch execution with progress + rate-limit support."""

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        context_manager: ContextManager | None = None,
        config_class: type = AgentConfig,
        run_id: str = "",
        coordinator: RateLimitCoordinator | None = None,
        progress_manager: BatchProgressManager | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            env=env,
            context_manager=context_manager,
            config_class=config_class,
            **kwargs,
        )
        self.run_id = run_id
        self._coordinator = coordinator
        self._progress = progress_manager
        # storing the previous cost
        self._prev_cost = 0.0

    def step(self) -> list[dict]:
        if self._coordinator is not None:
            self._coordinator.wait_if_paused()

        res = super().step()

        cost_delta = self.total_cost - self._prev_cost
        self._prev_cost = self.total_cost

        if self._progress is not None:
            self._progress.update_run_status(
                self.run_id,
                f"Step {self.n_calls + 1} (${self.total_cost:.2f}/{self.config.cost_limit:.2f})",
                cost_delta=cost_delta,
            )
        return res
