"""Core batch orchestration for running multiple forecasting problems."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

from miniprophet.agent.batch_agent import BatchForecastAgent, RateLimitCoordinator
from miniprophet.exceptions import BatchRunTimeoutError, SearchAuthError, SearchRateLimitError
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

    def __post_init__(self):
        if self.end_time:
            try:
                self.end_time = to_mm_dd_yyyy(self.end_time)
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


def to_mm_dd_yyyy(dt_str: str) -> str:
    from dateutil import parser

    dt = parser.parse(dt_str)
    return dt.strftime("%m/%d/%Y")


def load_problems(path: Path) -> list[ForecastProblem]:
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


def process_problem(
    problem: ForecastProblem,
    output_dir: Path,
    config: dict,
    coordinator: RateLimitCoordinator,
    progress: BatchProgressManager,
    summary_lock: threading.Lock,
    summary_path: Path,
    results: dict[str, RunResult],
    total_cost_ref: list[float],
    max_cost: float,
    timeout_seconds: float,
) -> bool:
    """Process a single forecasting problem. Returns True if successful or permanently failed."""
    from miniprophet.agent.context import SlidingWindowContextManager
    from miniprophet.environment.forecast_env import ForecastEnvironment, create_default_tools
    from miniprophet.environment.source_board import SourceBoard
    from miniprophet.models import get_model
    from miniprophet.search import get_search_tool

    run_id = problem.run_id
    run_dir = output_dir / "runs" / run_id
    result = results.setdefault(run_id, RunResult(run_id=run_id, title=problem.title))
    result.output_dir = f"runs/{run_id}"

    progress.on_run_start(run_id)

    agent = None
    timed_out = False
    try:
        if max_cost > 0 and total_cost_ref[0] >= max_cost:
            result.status = "skipped_cost_limit"
            result.error = f"Total batch cost limit (${max_cost:.2f}) reached."
            progress.on_run_end(run_id, result.status)
            return True

        model = get_model(config=config.get("model", {}))
        search_cfg = config.get("search", {})
        search_backend = get_search_tool(search_cfg=search_cfg)

        agent_cfg = config.get("agent", {})
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
            coordinator=coordinator,
            progress_manager=progress,
            **agent_cfg,
        )

        runtime_kwargs = (
            {
                "search_date_before": problem.end_time,
            }
            if problem.end_time
            else {}
        )

        if timeout_seconds > 0:
            import concurrent.futures

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                agent.run,
                title=problem.title,
                outcomes=problem.outcomes,
                ground_truth=problem.ground_truth,
                **runtime_kwargs,
            )
            try:
                forecast = future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError as exc:
                future.cancel()
                # Best-effort release of the batch worker without waiting for this run.
                executor.shutdown(wait=False, cancel_futures=True)
                raise BatchRunTimeoutError(
                    f"Batch run timed out after {timeout_seconds:.1f}s"
                ) from exc
            else:
                executor.shutdown(wait=True)
        else:
            forecast = agent.run(
                title=problem.title,
                outcomes=problem.outcomes,
                ground_truth=problem.ground_truth,
                **runtime_kwargs,
            )

        result.status = forecast.get("exit_status", "unknown")
        result.submission = forecast.get("submission") or None
        result.evaluation = forecast.get("evaluation") or None
        result.cost = {
            "model": agent.model_cost,
            "search": agent.search_cost,
            "total": agent.total_cost,
        }

        with summary_lock:
            total_cost_ref[0] += agent.total_cost

        progress.on_run_end(run_id, result.status)
        return True

    except SearchAuthError as exc:
        result.status = "auth_error"
        result.error = str(exc)
        progress.on_run_end(run_id, result.status)
        return True

    except BatchRunTimeoutError as exc:
        timed_out = True
        result.status = type(exc).__name__
        result.error = str(exc)
        progress.on_run_end(run_id, result.status)
        return True

    except Exception as exc:
        if _is_rate_limit_error(exc):
            coordinator.signal_rate_limit()
            if problem.retries < MAX_RETRIES:
                problem.retries += 1
                logger.warning(
                    "Rate limit for %s (attempt %d/%d) -- will retry.",
                    run_id,
                    problem.retries,
                    MAX_RETRIES,
                )
                progress.on_run_end(run_id, f"rate_limited (retry {problem.retries})")
                return False

        result.status = type(exc).__name__
        result.error = str(exc)
        logger.error("Run %s failed: %s", run_id, exc, exc_info=True)
        progress.on_run_end(run_id, result.status)
        return True

    finally:
        if agent is not None and not timed_out:
            try:
                agent.save(run_dir)
            except Exception:
                logger.error("Failed to save trajectory for %s", run_id, exc_info=True)
        _write_summary(summary_lock, summary_path, config, results, total_cost_ref[0])


def _write_summary(
    lock: threading.Lock,
    path: Path,
    config: dict,
    results: dict[str, RunResult],
    total_cost: float,
) -> None:
    summary = {
        "batch_config": {k: v for k, v in config.items() if k != "agent"},
        "total_cost": total_cost,
        "runs": [r.to_dict() for r in results.values()],
    }
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))


def run_batch(
    problems: list[ForecastProblem],
    output_dir: Path,
    config: dict,
    *,
    workers: int = 1,
    max_cost: float = 0.0,
    timeout_seconds: float = 180.0,
    initial_results: dict[str, RunResult] | None = None,
    initial_total_cost: float = 0.0,
) -> dict[str, RunResult]:
    """Execute a batch of forecasting problems with parallel workers."""
    import concurrent.futures

    from rich.live import Live

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_lock = threading.Lock()
    total_cost_ref: list[float] = [float(initial_total_cost)]

    coordinator = RateLimitCoordinator()
    progress = BatchProgressManager(len(problems))
    results: dict[str, RunResult] = dict(initial_results or {})

    queue: Queue[ForecastProblem] = Queue()
    for p in problems:
        queue.put(p)

    def worker_loop() -> None:
        while True:
            try:
                problem = queue.get(timeout=0.5)
            except Exception:
                if queue.empty():
                    return
                continue
            done = process_problem(
                problem,
                output_dir,
                config,
                coordinator,
                progress,
                summary_lock,
                summary_path,
                results,
                total_cost_ref,
                max_cost,
                timeout_seconds,
            )
            if not done:
                queue.put(problem)
            queue.task_done()

    with Live(progress.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker_loop) for _ in range(workers)]
            try:
                queue.join()
            except KeyboardInterrupt:
                logger.info("Batch interrupted by user. Waiting for active runs to finish...")
            for f in futures:
                try:
                    f.result(timeout=1)
                except Exception:
                    pass

    _write_summary(summary_lock, summary_path, config, results, total_cost_ref[0])
    return results
