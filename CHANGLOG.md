# Changelog

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
