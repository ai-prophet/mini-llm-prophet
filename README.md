# mini-llm-prophet

A minimal LLM forecasting agent scaffolding. Inspired by [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent).

## Quick Start

### 1. Install

```bash
cd mini-llm-prophet
pip install -e .
```

### 2. Set API Keys

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
export BRAVE_API_KEY="your-brave-search-key"       # for Brave search backend
# or
export PERPLEXITY_API_KEY="your-perplexity-key"    # for Perplexity search backend
```

### 3. Run

```bash
prophet \
  --title "Which team will win the NBA championship in 2026?" \
  --outcomes "Bucks,Warriors,Nets,Suns,Celtics" \
  --model openai/gpt-4o-mini
```

Or use **interactive mode** to set up via a TUI (with optional Kalshi market import):

```bash
prophet -i
```

## CLI Options

| Flag | Short | Description |
|------|-------|-------------|
| `--title` | `-t` | The forecasting question |
| `--outcomes` | `-o` | Comma-separated possible outcomes |
| `--model` | `-m` | Model name (OpenRouter format) |
| `--interactive` | `-i` | Enter interactive TUI mode |
| `--ground-truth` | `-g` | Ground truth JSON for evaluation (e.g. `'{"Yes": 1, "No": 0}'`) |
| `--cost-limit` | `-l` | Total cost limit in USD (default: 3.0) |
| `--search-limit` | | Max search queries (default: 10) |
| `--step-limit` | | Max agent steps (default: 30) |
| `--config` | `-c` | Config files or `key=value` overrides |
| `--output` | | Save trajectory JSON to this path |
| `--model-class` | | Override model class |

## Configuration

Default config is in `src/miniprophet/config/default.yaml`. Override with `-c`:

```bash
prophet -t "..." -o "..." -c model.model_kwargs.temperature=0.5
prophet -t "..." -o "..." -c search.search_class=perplexity
```

Multiple `-c` flags are merged in order (later values win).

## Architecture

The agent runs a tool-call loop where each step allows exactly one tool call:

1. **Search** -- query the web for relevant information (sources get global IDs: S1, S2, ...)
2. **Add Source** -- save a source to the board with an analytical note and optional per-outcome reaction
3. **Edit Note** -- update notes or reactions on previously saved sources
4. **Submit** -- provide a probabilistic forecast for all outcomes

### Key Components

- **`DefaultForecastAgent`** -- the core agent loop
- **`ForecastEnvironment`** -- dispatches tool calls, manages source board state
- **`SlidingWindowContextManager`** -- stateful context truncation with query history tracking
- **`OpenRouterModel`** -- LLM interface via OpenRouter API
- **`BraveSearchTool` / `PerplexitySearchTool`** -- pluggable search backends

### Cost Tracking

The agent tracks three cost categories:
- **Model cost** -- LLM API usage
- **Search cost** -- search tool usage (0 for Brave, nonzero for agentic search backends)
- **Total cost** -- sum of both; the `cost_limit` applies here

### Evaluation Mode

Pass `--ground-truth` to evaluate the agent's forecast against known outcomes:

```bash
prophet -t "Will it rain tomorrow?" -o "Yes,No" -g '{"Yes": 1, "No": 0}'
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

### Custom Model

Implement the `Model` protocol defined in `miniprophet/__init__.py`.

### Market Services

Implement the `MarketService` protocol to add new prediction market integrations (Kalshi is built-in).

## Development

```bash
pip install -e ".[dev]"
pre-commit install
```
