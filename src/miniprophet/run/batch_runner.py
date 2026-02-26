"""Core batch orchestration for running multiple forecasting problems."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from miniprophet.agent.batch_agent import BatchForecastAgent, RateLimitCoordinator
from miniprophet.exceptions import (
    BatchFatalError,
    BatchRunTimeoutError,
    SearchAuthError,
    SearchRateLimitError,
)
from miniprophet.run.batch_progress import BatchProgressManager

logger = logging.getLogger("miniprophet.batch")

MAX_RETRIES = 3

RETRYABLE_EXCEPTIONS = (SearchRateLimitError,)


@dataclass
class ForecastProblem:
    """A single forecasting problem from the input JSONL."""

    run_id: str
    title: str
    outcomes: list[str]
    ground_truth: dict[str, int] | None = None
    end_time: str | None = None
    retries: int = 0
    offset: int = 0

    def __post_init__(self):
        if self.end_time:
            try:
                self.end_time = to_mm_dd_yyyy(self.end_time, self.offset)
            except ValueError:
                self.end_time = None


@dataclass
class RunResult:
    """Result summary for one forecasting run."""

    run_id: str
    title: str
    status: str = "pending"
    cost: dict[str, float] = field(
        default_factory=lambda: {"model": 0.0, "search": 0.0, "total": 0.0}
    )
    submission: dict[str, float] | None = None
    evaluation: dict[str, float] | None = None
    error: str | None = None
    output_dir: str = ""

    @classmethod
    def from_dict(cls, payload: dict) -> RunResult:
        cost = payload.get("cost", {}) if isinstance(payload.get("cost", {}), dict) else {}
        return cls(
            run_id=str(payload.get("run_id", "")),
            title=str(payload.get("title", "")),
            status=str(payload.get("status", "pending")),
            cost={
                "model": float(cost.get("model", 0.0) or 0.0),
                "search": float(cost.get("search", 0.0) or 0.0),
                "total": float(cost.get("total", 0.0) or 0.0),
            },
            submission=payload.get("submission"),
            evaluation=payload.get("evaluation"),
            error=payload.get("error"),
            output_dir=str(payload.get("output_dir", "")),
        )

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "title": self.title,
            "status": self.status,
            "cost": self.cost,
            "submission": self.submission,
            "evaluation": self.evaluation,
            "error": self.error,
            "output_dir": self.output_dir,
        }


@dataclass
class BatchRunArgs:
    """Runtime arguments for a batch execution."""

    output_dir: Path
    config: dict
    workers: int = 1
    max_cost: float = 0.0
    timeout_seconds: float = 180.0
    initial_results: dict[str, RunResult] | None = None
    initial_total_cost: float = 0.0
    search_date_before: str | None = None
    search_date_after: str | None = None


@dataclass
class BatchRunState:
    """Shared mutable state for workers in a single batch execution."""

    coordinator: RateLimitCoordinator
    progress: BatchProgressManager
    summary_lock: threading.Lock
    summary_path: Path
    results: dict[str, RunResult]
    total_cost_ref: list[float]
    fatal_lock: threading.Lock
    fatal_event: threading.Event
    fatal_error: str | None = None


def to_mm_dd_yyyy(dt_str: str, offset: int = 0) -> str:
    from datetime import timedelta

    from dateutil import parser

    dt = parser.parse(dt_str)
    return (dt - timedelta(days=offset)).strftime("%m/%d/%Y")


def load_problems(path: Path, offset: int = 0) -> list[ForecastProblem]:
    """Load and validate forecasting problems from a JSONL file."""
    problems: list[ForecastProblem] = []
    seen_ids: set[str] = set()
    auto_idx = 0

    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

        if "title" not in data or "outcomes" not in data:
            raise ValueError(f"Line {line_no}: 'title' and 'outcomes' are required fields.")

        run_id = data.get("run_id")
        if run_id is None:
            run_id = f"run_{auto_idx}"
            auto_idx += 1
        else:
            run_id = str(run_id)

        if run_id in seen_ids:
            raise ValueError(f"Line {line_no}: duplicate run_id '{run_id}'.")
        seen_ids.add(run_id)

        problems.append(
            ForecastProblem(
                run_id=run_id,
                title=data["title"],
                outcomes=data["outcomes"],
                ground_truth=data.get("ground_truth"),
                end_time=data.get("end_time"),
                offset=offset,
            )
        )

    return problems


def load_existing_summary(path: Path) -> tuple[dict[str, RunResult], float]:
    """Load existing summary.json into RunResult mapping and total_cost.

    Returns empty state when summary file does not exist.
    """
    if not path.exists():
        return {}, 0.0

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid summary JSON at {path}: {exc}") from exc

    runs = data.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError(f"Invalid summary format at {path}: 'runs' must be a list")

    results: dict[str, RunResult] = {}
    for item in runs:
        if not isinstance(item, dict):
            continue
        run = RunResult.from_dict(item)
        if run.run_id:
            results[run.run_id] = run

    total_cost = float(data.get("total_cost", 0.0) or 0.0)
    return results, total_cost


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception indicates a rate limit (retryable)."""
    if isinstance(exc, RETRYABLE_EXCEPTIONS):
        return True
    exc_str = str(exc).lower()
    return "429" in exc_str or "rate limit" in exc_str


def _is_auth_error(exc: Exception) -> bool:
    """Check if an exception indicates auth/API credential failure."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()

    if "auth" in name:
        return True
    if "permissiondenied" in name:
        return True

    auth_tokens = (
        "authentication",
        "unauthorized",
        "permission denied",
        "invalid api key",
        "api key",
        "http 401",
        "status code 401",
    )
    return any(token in msg for token in auth_tokens)


def _write_summary(args: BatchRunArgs, state: BatchRunState) -> None:
    summary = {
        "batch_config": {k: v for k, v in args.config.items() if k != "agent"},
        "total_cost": state.total_cost_ref[0],
        "runs": [r.to_dict() for r in state.results.values()],
    }
    with state.summary_lock:
        state.summary_path.parent.mkdir(parents=True, exist_ok=True)
        state.summary_path.write_text(json.dumps(summary, indent=2))


def _run_agent_with_timeout(
    *,
    agent: BatchForecastAgent,
    timeout_seconds: float,
    cancel_event: threading.Event,
    run_id: str,
    title: str,
    outcomes: list[str],
    ground_truth: dict[str, int] | None,
    runtime_kwargs: dict[str, Any],
) -> Any:
    if timeout_seconds <= 0:
        return agent.run(
            title=title,
            outcomes=outcomes,
            ground_truth=ground_truth,
            **runtime_kwargs,
        )

    done = threading.Event()
    result_holder: list[Any] = []
    error_holder: list[Exception] = []

    def _target() -> None:
        try:
            result_holder.append(
                agent.run(
                    title=title,
                    outcomes=outcomes,
                    ground_truth=ground_truth,
                    **runtime_kwargs,
                )
            )
        except Exception as exc:
            error_holder.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=_target, name=f"batch-run-{run_id}", daemon=True)
    thread.start()

    if not done.wait(timeout=timeout_seconds):
        cancel_event.set()
        raise BatchRunTimeoutError(f"Batch run timed out after {timeout_seconds:.1f}s")

    if error_holder:
        raise error_holder[0]

    if not result_holder:
        raise RuntimeError(f"Run {run_id} finished without a result payload")

    return result_holder[0]


def process_problem(
    problem: ForecastProblem,
    args: BatchRunArgs,
    state: BatchRunState,
) -> bool:
    """Process a single forecasting problem. Returns True if successful or permanently failed."""
    from miniprophet.agent.context import SlidingWindowContextManager
    from miniprophet.environment.forecast_env import ForecastEnvironment, create_default_tools
    from miniprophet.environment.source_board import SourceBoard
    from miniprophet.models import get_model
    from miniprophet.search import get_search_tool

    run_id = problem.run_id
    run_dir = args.output_dir / "runs" / run_id
    result = state.results.setdefault(run_id, RunResult(run_id=run_id, title=problem.title))
    result.output_dir = f"runs/{run_id}"

    state.progress.on_run_start(run_id)

    agent = None
    timed_out = False
    cancel_event = threading.Event()
    try:
        if args.max_cost > 0 and state.total_cost_ref[0] >= args.max_cost:
            result.status = "skipped_cost_limit"
            result.error = f"Total batch cost limit (${args.max_cost:.2f}) reached."
            state.progress.on_run_end(run_id, result.status)
            return True

        model = get_model(config=args.config.get("model", {}))
        search_cfg = args.config.get("search", {})
        search_backend = get_search_tool(search_cfg=search_cfg)

        agent_cfg = args.config.get("agent", {})
        agent_search_limit = agent_cfg.get("search_limit", 10)
        board = SourceBoard()
        tools = create_default_tools(
            search_tool=search_backend,
            outcomes=problem.outcomes,
            board=board,
            search_limit=agent_search_limit,
            search_results_limit=search_cfg.get("search_results_limit", 5),
            max_source_display_chars=search_cfg.get("max_source_display_chars", 2000),
        )
        env = ForecastEnvironment(tools, board=board)

        context_window = agent_cfg.get("context_window", 6)
        ctx_mgr = (
            SlidingWindowContextManager(window_size=context_window) if context_window > 0 else None
        )
        agent = BatchForecastAgent(
            model=model,
            env=env,
            context_manager=ctx_mgr,
            run_id=run_id,
            coordinator=state.coordinator,
            progress_manager=state.progress,
            cancel_event=cancel_event,
            **agent_cfg,
        )

        runtime_kwargs = {
            "search_date_before": args.search_date_before or problem.end_time,
            "search_date_after": args.search_date_after,
        }

        forecast = _run_agent_with_timeout(
            agent=agent,
            timeout_seconds=args.timeout_seconds,
            cancel_event=cancel_event,
            run_id=run_id,
            title=problem.title,
            outcomes=problem.outcomes,
            ground_truth=problem.ground_truth,
            runtime_kwargs=runtime_kwargs,
        )

        result.status = forecast.get("exit_status", "unknown")
        result.submission = forecast.get("submission") or None
        result.evaluation = forecast.get("evaluation") or None
        result.cost = {
            "model": agent.model_cost,
            "search": agent.search_cost,
            "total": agent.total_cost,
        }

        with state.summary_lock:
            state.total_cost_ref[0] += agent.total_cost

        state.progress.on_run_end(run_id, result.status)
        return True

    except SearchAuthError as exc:
        result.status = "auth_error"
        result.error = str(exc)
        state.progress.on_run_end(run_id, result.status)
        raise BatchFatalError(f"Run {run_id} failed with auth error: {exc}") from exc

    except BatchRunTimeoutError as exc:
        timed_out = True
        result.status = type(exc).__name__
        result.error = str(exc)
        state.progress.on_run_end(run_id, result.status)
        return True

    except Exception as exc:
        if _is_rate_limit_error(exc):
            state.coordinator.signal_rate_limit()
            if problem.retries < MAX_RETRIES:
                problem.retries += 1
                logger.warning(
                    "Rate limit for %s (attempt %d/%d) -- will retry.",
                    run_id,
                    problem.retries,
                    MAX_RETRIES,
                )
                state.progress.on_run_end(run_id, f"rate_limited (retry {problem.retries})")
                return False

        if _is_auth_error(exc):
            result.status = "auth_error"
            result.error = str(exc)
            state.progress.on_run_end(run_id, result.status)
            raise BatchFatalError(f"Run {run_id} failed with auth error: {exc}") from exc

        result.status = type(exc).__name__
        result.error = str(exc)
        logger.error("Run %s failed: %s", run_id, exc, exc_info=True)
        state.progress.on_run_end(run_id, result.status)
        return True

    finally:
        if agent is not None and not timed_out:
            try:
                agent.save(run_dir)
            except Exception:
                logger.error("Failed to save trajectory for %s", run_id, exc_info=True)
        _write_summary(args, state)


def run_batch(
    problems: list[ForecastProblem],
    args: BatchRunArgs,
) -> dict[str, RunResult]:
    """Execute a batch of forecasting problems with parallel workers."""
    import concurrent.futures

    from rich.live import Live

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = BatchRunState(
        coordinator=RateLimitCoordinator(),
        progress=BatchProgressManager(len(problems)),
        summary_lock=threading.Lock(),
        summary_path=args.output_dir / "summary.json",
        results=dict(args.initial_results or {}),
        total_cost_ref=[float(args.initial_total_cost)],
        fatal_lock=threading.Lock(),
        fatal_event=threading.Event(),
    )

    queue: Queue[ForecastProblem] = Queue()
    for p in problems:
        queue.put(p)

    def _drain_pending_queue() -> int:
        drained = 0
        while True:
            try:
                queue.get_nowait()
            except Empty:
                break
            else:
                queue.task_done()
                drained += 1
        return drained

    def worker_loop() -> None:
        while True:
            if state.fatal_event.is_set():
                return

            try:
                problem = queue.get(timeout=0.5)
            except Empty:
                if queue.empty() or state.fatal_event.is_set():
                    return
                continue

            try:
                if state.fatal_event.is_set():
                    return

                done = process_problem(problem, args, state)
                if not done and not state.fatal_event.is_set():
                    queue.put(problem)
            except BatchFatalError as exc:
                with state.fatal_lock:
                    if not state.fatal_event.is_set():
                        state.fatal_error = str(exc)
                        state.fatal_event.set()
                _drain_pending_queue()
                return
            except Exception as exc:
                result = state.results.setdefault(
                    problem.run_id,
                    RunResult(run_id=problem.run_id, title=problem.title),
                )
                result.status = type(exc).__name__
                result.error = str(exc)
                logger.error("Unhandled worker exception for %s", problem.run_id, exc_info=True)
                state.progress.on_run_end(problem.run_id, result.status)
                _write_summary(args, state)
            finally:
                queue.task_done()

    with Live(state.progress.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(worker_loop) for _ in range(args.workers)]
            try:
                queue.join()
            except KeyboardInterrupt:
                logger.info("Batch interrupted by user. Waiting for active runs to finish...")
            for f in futures:
                try:
                    f.result(timeout=1)
                except Exception:
                    pass

    _write_summary(args, state)

    if state.fatal_event.is_set():
        raise BatchFatalError(state.fatal_error or "Batch terminated due to a fatal error.")

    return state.results
