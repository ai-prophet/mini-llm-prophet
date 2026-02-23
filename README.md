<p align="left">
  <img src="./assets/icon.svg" alt="mini-llm-prophet icon" style="height:10em"/>
</p>

# mini-llm-prophet

A minimal LLM forecasting agent scaffolding. Inspired by [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent).

<p align="left">
  <img src="./assets/mini-llm-prophet-loop.svg" alt="mini-llm-prophet loop" style="height:100%"/>
</p>

## Quick Start

### 1. Install

```bash
cd mini-llm-prophet
pip install -e .
```

### 2. Set API Keys

```bash
export OPENROUTER_API_KEY="your-openrouter-key"    # for OpenRouter model class
export PERPLEXITY_API_KEY="your-perplexity-key"    # for Perplexity search backend (default searcher)
# or
export BRAVE_API_KEY="your-brave-search-key"       # for Brave search backend
```

Two model classes are supported: **OpenRouter** (default) and **LiteLLM**. Use `--model-class litellm` to access any provider supported by [LiteLLM](https://docs.litellm.ai/) (OpenAI, Anthropic, etc.).

### 3. Run

```bash
prophet run --title "Which team will win the NBA championship in 2026?" \
  --outcomes "Bucks,Warriors,Nets,Suns,Celtics" \
  --model-class litellm \
  --model gemini/gemini-3-flash-preview
```

**Optionally**, if you want to perform evaluation, you can pass the ground truth as a JSON string:

```bash
prophet run \
  --title "Which team will win the NBA championship in 2025?" \
  --outcomes "Thunder,Warriors,Nets,Suns,Celtics" \
  --ground-truth '{"Thunder": 1, "Warriors": 0, "Nets": 0, "Suns": 0, "Celtics": 0}'
```

Or use **interactive mode** to set up via a TUI (with optional Kalshi market import):

```bash
prophet run -i \
  --model-class litellm \
  --model gemini/gemini-3-flash-preview
```

### 4. Batch Mode

Run multiple forecasting problems from a `.jsonl` file:

```bash
prophet batch -f problems.jsonl -o ./batch_output -w 4 --model openai/gpt-4o
```

Each line in the JSONL file is a forecasting problem:

```json
{"title": "Will X happen?", "outcomes": ["Yes", "No"], "ground_truth": {"Yes": 1, "No": 0}, "run_id": "my_problem"}
```

`ground_truth` and `run_id` are optional. Output goes to a meta-directory with per-run artifacts and a `summary.json`.

## CLI Options

### `prophet run` (single forecast)

| Flag             | Short | Description                                                     |
| ---------------- | ----- | --------------------------------------------------------------- |
| `--title`        | `-t`  | The forecasting question                                        |
| `--outcomes`     | `-o`  | Comma-separated possible outcomes                               |
| `--model`        | `-m`  | Model name (e.g. `openai/gpt-4o`)                               |
| `--interactive`  | `-i`  | Enter interactive TUI mode                                      |
| `--ground-truth` | `-g`  | Ground truth JSON for evaluation (e.g. `'{"Yes": 1, "No": 0}'`) |
| `--cost-limit`   | `-l`  | Total cost limit in USD (default: 3.0)                          |
| `--search-limit` |       | Max search queries (default: 10)                                |
| `--step-limit`   |       | Max agent steps (default: 30)                                   |
| `--config`       | `-c`  | Config files or `key=value` overrides                           |
| `--output`       |       | Output directory for run artifacts (info.json + trajectory.json) |
| `--model-class`  |       | Override model class (`openrouter` or `litellm`)                |

### `prophet batch` (batch forecast)

| Flag               | Short | Description                                       |
| ------------------ | ----- | ------------------------------------------------- |
| `--input`          | `-f`  | Path to `.jsonl` input file (required)             |
| `--output`         | `-o`  | Output meta-directory (required)                   |
| `--workers`        | `-w`  | Number of parallel workers (default: 1)            |
| `--max-cost`       |       | Total cost budget across all runs (0 = unlimited)  |
| `--max-cost-per-run` |     | Per-run cost limit (overrides agent config)        |
| `--model`          | `-m`  | Model name override                                |
| `--model-class`    |       | Model class override                               |
| `--config`         | `-c`  | Config files or `key=value` overrides              |

## Configuration

Default config is in `src/miniprophet/config/default.yaml`. Override with `-c`:

```bash
prophet run -t "..." -o "..." -c model.model_kwargs.temperature=0.5
prophet run -t "..." -o "..." -c search.search_class=perplexity
```

Multiple `-c` flags are merged in order (later values win).

## Architecture

The agent runs a tool-call loop where each step allows exactly one tool call. Tools are modular -- each lives in the `tools/` package and implements the `Tool` protocol (schema, execute, display):

1. **Search** -- query the web for relevant information (sources get global IDs: S1, S2, ...)
2. **Add Source** -- save a source to the board with an analytical note and optional per-outcome reaction
3. **Edit Note** -- update notes or reactions on previously saved sources
4. **Submit** -- provide a probabilistic forecast for all outcomes

The agent loop is cleanly separated from the CLI display layer. `DefaultForecastAgent` contains no UI code; `CliForecastAgent` extends it with Rich-based display hooks and is used by the CLI entry point.

Before each step, if the agent has a `ContextManager` enabled, it will manage the context window (e.g. by truncating the message history).
You can see the default context management strategy in `src/miniprophet/agent/context.py`.

### Key Components

- **`DefaultForecastAgent`** -- the core agent loop (no display dependencies), with built-in `TrajectoryRecorder` for per-step observability
- **`CliForecastAgent`** -- CLI-aware agent that adds Rich display via hook overrides
- **`BatchForecastAgent`** -- batch-mode agent with rate-limit coordination and progress hooks
- **`ForecastEnvironment`** -- thin dispatcher that delegates to registered `Tool` instances
- **`tools/`** -- modular tool package (`SearchForecastTool`, `AddSourceTool`, `EditNoteTool`, `SubmitTool`)
- **`SlidingWindowContextManager`** -- stateful context truncation with query history tracking
- **`OpenRouterModel`** / **`LitellmModel`** -- LLM interfaces (OpenRouter API and LiteLLM)
- **`BraveSearchTool`** / **`PerplexitySearchTool`** -- pluggable search backends

### Cost Tracking

The agent tracks three cost categories:

- **Model cost** -- LLM API usage
- **Search cost** -- search tool usage (0 for Brave, nonzero for agentic search backends)
- **Total cost** -- sum of both; the `cost_limit` applies here

### Evaluation Mode

Pass `--ground-truth / -g` to evaluate the agent's forecast against known outcomes:

```bash
prophet run -t "Will it rain tomorrow?" -o "Yes,No" -g '{"Yes": 1, "No": 0}'
```

Computes Brier Score (and any registered custom metrics) and includes results in both the CLI output and the trajectory file.

### Source Reactions

When adding sources, the model can annotate per-outcome sentiment (`very_positive`, `positive`, `neutral`, `negative`, `very_negative`), displayed compactly in the CLI with colored symbols.

## Extending

### Custom Search Backend

Implement the `SearchTool` protocol:

```python
from miniprophet.search import SearchResult
from miniprophet.environment.source_board import Source

class MySearchTool:
    def search(self, query: str, limit: int = 5) -> SearchResult:
        sources = [...]
        return SearchResult(sources=sources, cost=0.05)

    def serialize(self) -> dict:
        return {"info": {"config": {"search": {"search_class": "my_search"}}}}
```

### Custom Metrics

```python
from miniprophet.utils.metrics import register_metric

class LogScore:
    name = "log_score"
    def compute(self, probabilities, ground_truth):
        ...

register_metric(LogScore())
```

### Custom Tool

Implement the `Tool` protocol and register it with the environment. Each tool defines its schema, execution logic, and display rendering:

```python
class MyTool:
    @property
    def name(self) -> str:
        return "my_tool"

    def get_schema(self) -> dict:
        return {"type": "function", "function": {"name": "my_tool", ...}}

    def execute(self, args: dict) -> dict:
        return {"output": "result"}

    def display(self, output: dict) -> None:
        ...  # Rich rendering (optional, no-op for headless usage)
```

See the built-in tools in `src/miniprophet/tools/` for full examples.

### Custom Model

Implement the `Model` protocol defined in `miniprophet/__init__.py`. The easiest path for most providers is to use the built-in `LitellmModel` class with `--model-class litellm`, which supports any provider via [LiteLLM](https://docs.litellm.ai/).

### Market Services

Implement the `MarketService` protocol to add new prediction market integrations (Kalshi is built-in).

## Development

```bash
pip install -e ".[dev]"
pre-commit install
```
