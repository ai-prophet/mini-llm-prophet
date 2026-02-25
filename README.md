<p align="center">
  <img src="./assets/icon.svg" alt="mini-llm-prophet icon" style="height:10em"/>
</p>

# mini-llm-prophet

A minimal LLM forecasting agent scaffolding. Inspired by [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent).

<p align="left">
  <img src="./assets/mini-llm-prophet-loop.svg" alt="mini-llm-prophet loop" style="height:100%"/>
</p>

## Install

```bash
cd mini-llm-prophet
pip install -e ".[perplexity]"
```

`perplexity` is the default search backend, so this is the recommended install.
If you only use Brave search, `pip install -e .` is enough.

## Set API keys

Use the built-in CLI to persist keys into prophet's global `.env` file:

```bash
# direct set
prophet set PERPLEXITY_API_KEY "your-perplexity-key"
prophet set BRAVE_API_KEY "your-brave-key"
prophet set OPENROUTER_API_KEY "your-openrouter-key"

# interactive editor
prophet set -i
```

By default, prophet stores and loads values from:

- `~/.config/mini-llm-prophet/.env` (in linux; or your platform's equivalent config directory)

To use a different global config directory, set:

```bash
export MINIPROPHET_GLOBAL_CONFIG_DIR="/path/to/custom/config-dir"
```

You can still set environment variables directly in your shell if you prefer.

## Try this

Single run:

```bash
prophet run \
  --title "Which team will win the NBA championship in 2026?" \
  --outcomes "Bucks,Warriors,Celtics,Nuggets,Other" \
  --model-class litellm \
  --model gemini/gemini-3-flash-preview
```

Interactive mode:

```bash
prophet run -i --model-class litellm --model gemini/gemini-3-flash-preview
```

Batch mode with the sample file:

```bash
prophet batch \
  -f examples/example_batch_job.jsonl \
  -o outputs/batch-demo \
  -w 4 \
  --model-class litellm \
  --model gemini/gemini-3-flash-preview
```

Resume an interrupted batch and skip completed run IDs:

```bash
prophet batch -f examples/example_batch_job.jsonl -o outputs/batch-demo --resume
```

Run artifacts now include `sources.json` in addition to `info.json` and `trajectory.json`.

## Docs

Detailed docs are split by topic:

- [Architecture](docs/architecture.md)
- [CLI (single run)](docs/cli.md)
- [Batch processing](docs/batch.md)
- [Extending the framework](docs/extension.md)
- [Output and trajectory format](docs/output.md)

Release notes: [CHANGLOG.md](CHANGLOG.md)
