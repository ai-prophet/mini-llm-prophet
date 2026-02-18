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
export BRAVE_API_KEY="your-brave-search-key"
```

### 3. Run

```bash
prophet \
  --title "Which team will win the NBA championship in 2026?" \
  --outcomes "Bucks,Warriors,Nets,Suns,Celtics" \
  --model openai/gpt-4o-mini
```

Or with `python -m`:

```bash
python -m miniprophet \
  -t "Will it rain in SF tomorrow?" \
  -o "Yes,No" \
  -m openai/gpt-4o-mini
```

## CLI Options

| Flag | Short | Description |
|------|-------|-------------|
| `--title` | `-t` | The forecasting question |
| `--outcomes` | `-o` | Comma-separated possible outcomes |
| `--model` | `-m` | Model name (OpenRouter format) |
| `--cost-limit` | `-l` | Total cost limit in USD (default: 3.0) |
| `--search-limit` | | Max search queries (default: 10) |
| `--step-limit` | | Max agent steps (default: 30) |
| `--config` | `-c` | Config files or `key=value` overrides |
| `--output` | | Save trajectory JSON to this path |
| `--model-class` | | Override model class |

## Configuration

Default config is in `src/miniprophet/config/default.yaml`. Override with `-c`:

```bash
# Use a custom config file
prophet -t "..." -o "..." -c my_config.yaml

# Override individual values
prophet -t "..." -o "..." -c model.model_kwargs.temperature=0.5
```

Multiple `-c` flags are merged in order (later values win).

## Architecture

The agent runs a tool-call loop:

1. **Search** -- query the web for relevant information
2. **Add Source** -- save a search result to the source board with analytical notes
3. **Edit Note** -- update notes on previously saved sources
4. **Submit** -- provide a probabilistic forecast for all outcomes

Everything is a tool call. Each step allows exactly one tool call.

### Key Components

- **`DefaultForecastAgent`** -- the core loop (~150 lines)
- **`ForecastEnvironment`** -- dispatches tool calls, manages source board state
- **`OpenRouterModel`** -- LLM interface via OpenRouter API
- **`BraveSearchTool`** -- web search via Brave Search API + trafilatura extraction

### Cost Tracking

The agent tracks three cost categories:
- **Model cost** -- LLM API usage
- **Search cost** -- search tool usage (0 for Brave, nonzero for agentic search backends)
- **Total cost** -- sum of both; the `cost_limit` applies here

## Extending

### Custom Search Backend

Implement the `SearchTool` protocol:

```python
from miniprophet.search import SearchResult, SearchTool
from miniprophet.environment.source_board import Source

class MySearchTool:
    def search(self, query: str, limit: int = 5) -> SearchResult:
        sources = [...]  # your search logic
        return SearchResult(sources=sources, cost=0.05)

    def serialize(self) -> dict:
        return {"info": {"config": {"search": {"search_class": "my_search"}}}}
```

### Custom Model

Implement the `Model` protocol defined in `miniprophet/__init__.py`.
