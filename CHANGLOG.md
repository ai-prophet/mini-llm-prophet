# Changelog

## v0.1.4

### CLI: Source board redesign

The source board panel is now rendered as grouped source cards for better scanability during runs:
- Each entry shows `[#id] title`, URL, snippet preview, and note content in a dedicated nested panel
- Reactions are rendered with explicit sentiment colors/symbols (`++`, `+`, `~`, `-`, `--`)
- Long snippets/notes now show truncation hints (for example, `...N characters omitted`)
- Panel subtitle includes source count and empty-state text is cleaner

### Improved: Search results readability

Search result rendering has been refined to reduce visual noise:
- Result metadata and title now render on one line
- Snippet truncation now explicitly reports omitted character count

### Improved: Batch run progress and cost accounting

Batch progress behavior is now more informative and simpler:
- `BatchForecastAgent` now tracks per-step cost deltas and forwards them to the progress manager
- Per-run status now displays both consumed cost and run limit (`$used/$limit`)
- Batch progress UI removed ETA estimation (elapsed/progress/cost remain)
- Added `examples/example_batch_job.jsonl` as a ready-to-run batch input sample

### Important: Trajectory key format update

Trajectory message keys are now role-prefixed instead of generic `mN` IDs:
- Previous format example: `m0`, `m1`, `m2`
- New format example: `S0`, `U0`, `A0`, `T0` (with `O*` for non-standard roles)

This improves trace readability but changes `trajectory.json` key naming for downstream consumers.

### Config and docs updates

- Default config now explicitly sets `model.model_class: "litellm"`
- README quickstart examples now include `--model-class litellm` with Gemini model usage
- Minor context-summary wording cleanup in truncated context text

## v0.1.3

### Major: Agent trajectory observability

Agent runs now record full per-step input/output trajectories via a `TrajectoryRecorder`. Each message is stored once in a global pool and referenced by key, allowing exact reconstruction of what the LLM saw at every step -- even when the context manager truncates or replaces messages between steps.

Serialization output has changed from a single JSON file to a **directory** containing:
- `info.json` -- config, cost stats, version, exit status, submission, evaluation
- `trajectory.json` -- message pool + per-step input/output indices

The `--output` CLI option now points to a directory instead of a file.

### Improved: XML-tagged prompt format

Structured data in prompts and tool outputs now uses XML tags for clearer parsing by LLMs:
- Source board: `<source_board>` / `<source>` tags
- Tool output: `<output>` / `<error>` tags
- Search results: `<search_results>` / `<result>` tags
- Instance template: `<forecast_problem>` tag for the problem definition

Instruction sections (Strategy, Guidelines) remain as markdown lists.

### Major: Batch agent running mode

New `prophet batch` CLI command for running multiple forecasting problems in parallel:
- Input: `.jsonl` file with `title`, `outcomes`, optional `ground_truth` and `run_id` per line
- Output: meta-directory with per-run artifacts and a `summary.json` with status, costs, submissions, and evaluations
- Parallel execution via `--workers N` with a queue-based scheduler
- Global `RateLimitCoordinator`: when any worker hits a rate limit, all workers pause for a backoff period
- Automatic retry (up to 3 attempts) for rate-limited runs; permanent failures are recorded immediately
- Total cost budget (`--max-cost`) and per-run cost limit (`--max-cost-per-run`)
- Rich live progress display with overall progress bar, per-worker status, and exit status summary table

### CLI restructured

The CLI is now organized as sub-commands under a root `prophet` app:
- `prophet run` -- single forecast run (previously the default command)
- `prophet batch` -- batch forecasting from JSONL

## v0.1.2

### New: LiteLLM model support

Added `LitellmModel` as a second model class alongside `OpenRouterModel`. Use `--model-class litellm` to access any provider supported by LiteLLM (OpenAI, Anthropic, Google, etc.) without needing an OpenRouter key.

### Major: Agent and CLI separation

The agent loop no longer contains any display logic. `DefaultForecastAgent` is now a pure backend class with hook methods (`on_run_start`, `on_step_start`, `on_model_response`, `on_observation`, `on_run_end`) that subclasses can override.

A new `CliForecastAgent` inherits from it and provides all Rich-based CLI display, including a live spinner while the model is thinking. The CLI entry point (`run/cli.py`) now uses `CliForecastAgent`.

All display components have been moved into a dedicated `cli/components/` package, replacing the old monolithic `utils/display.py` and `run/tui.py`.

### Major: Modular tool system

Tools are no longer defined inline inside `ForecastEnvironment`. A new `Tool` protocol (`name`, `get_schema`, `execute`, `display`) allows each tool to be a self-contained module under the `tools/` package:

- `tools/search_tool.py` -- web search with source ID tracking
- `tools/source_board_tools.py` -- `add_source` and `edit_note`
- `tools/submit.py` -- final forecast submission

`ForecastEnvironment` is now a thin dispatcher that receives a list of `Tool` instances and delegates execution by name.

### Improved: Source board as invariant context

The source board state is now injected as a persistent message (position 2, after system and user prompts) that refreshes every step, rather than being appended to each tool's output. This means the model always sees the current board at a fixed location, and tool responses only contain their own results.

### CLI improvements

- A live spinner shows the model name while waiting for a response (e.g. "openai/gpt-4o is forecasting...").
- In interactive mode, the agent loops back to the forecast setup screen after each run instead of exiting.
- Empty fields in the forecast setup TUI now show "(Empty: awaiting input...)" instead of blank parentheses.
- Forecast results, evaluation metrics, and the run footer are displayed in that order after the agent finishes (previously metrics appeared before the forecast).
- `run()` now returns a `ForecastResult` TypedDict with structured `exit_status`, `submission`, `evaluation`, and `board` fields.
