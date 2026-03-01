# Eval Processing

This page covers `prophet eval` for running many forecasting tasks from either:

- a local JSONL file (`-f/--input`)
- a dataset reference (`-d/--dataset`)

## CLI entrypoint

```bash
prophet eval (-f <input.jsonl> | -d <dataset_ref>) -o <output_dir> [options]
```

## Dataset references

`--dataset/-d` supports two patterns:

- Registry dataset: `name`, `name@version`, `name@latest`
- Hugging Face dataset: `username/dataset`, `username/dataset@revision`

If a ref contains `/`, it is treated as Hugging Face.

## Core options

| Flag | Short | Description |
| --- | --- | --- |
| `--input` | `-f` | Input `.jsonl` file (mutually exclusive with `--dataset`) |
| `--dataset` | `-d` | Dataset ref (`name@version` or `username/dataset@revision`) |
| `--output` | `-o` | Output meta-directory |
| `--workers` | `-w` | Parallel worker count (default `1`) |
| `--max-cost` |  | Total eval budget in USD (`0` = unlimited) |
| `--max-cost-per-run` |  | Per-run cost cap |
| `--model` | `-m` | Model override |
| `--model-class` |  | Model class override |
| `--config` | `-c` | Config file(s) or `key=value` overrides |
| `--resume` |  | Resume from existing `summary.json` |
| `--search-date-before` |  | Global upper-bound date (`MM/DD/YYYY`) |
| `--search-date-after` |  | Global lower-bound date (`MM/DD/YYYY`) |
| `--offset` |  | Days subtracted from each row's `predict_by` |
| `--agent` |  | Built-in eval agent alias (`default`) |
| `--agent-import-path` |  | Custom agent class (`module.path:ClassName`) |
| `--agent-kwarg` |  | Agent kwarg (`key=value`, repeatable) |
| `--registry-path` |  | Local registry file override |
| `--registry-url` |  | Remote registry URL override |
| `--hf-split` |  | HF split to load (default `train`) |
| `--overwrite-cache` |  | Refresh cached dataset artifact |

## Examples

JSONL input:

```bash
prophet eval \
  -f examples/example_batch_job.jsonl \
  -o outputs/eval-demo \
  -w 4 \
  --model-class litellm \
  --model gemini/gemini-3-flash-preview
```

Registry dataset:

```bash
prophet eval -d weekly-nba@latest -o outputs/weekly-nba
```

Hugging Face dataset:

```bash
prophet eval -d your-org/forecast-bench@main --hf-split train -o outputs/hf-bench
```

Custom agent:

```bash
prophet eval \
  -f examples/example_batch_job.jsonl \
  --agent-import-path mypkg.my_agent:MyForecastAgent \
  --agent-kwarg temperature=0.2
```

## Input row schema

Each task row is one JSON object. Required field:

- `title` (string)
- `outcomes` (array of strings, length >= 2)

Optional fields:

- `task_id` (string): used as run ID
- `context` (string)
- `ground_truth` (object)
- `predict_by` (string date/datetime)
- `source` (string)
- `criteria` (string)
- `metadata` (object)

Legacy aliases accepted during ingestion:

- `run_id` -> `task_id`
- `end_time` -> `predict_by`

## Resume mode (`--resume`)

When enabled, eval startup checks `<output_dir>/summary.json`:

- existing run IDs in summary are skipped
- if summary includes run IDs not in current input, eval exits with an error

## Timeout config

Default config:

```yaml
eval:
  timeout: 180
```

## Output layout

Given `-o outputs/eval-demo`:

- `outputs/eval-demo/summary.json`
- `outputs/eval-demo/runs/<run_id>/info.json`
- `outputs/eval-demo/runs/<run_id>/trajectory.json`
- `outputs/eval-demo/runs/<run_id>/sources.json`
