# Batch Processing

This page covers `prophet batch` for running many forecasting jobs from a JSONL file.

## CLI entrypoint

```bash
prophet batch -f <input.jsonl> -o <output_dir> [options]
```

### Options

| Flag | Short | Description |
| --- | --- | --- |
| `--input` | `-f` | Input `.jsonl` file (required) |
| `--output` | `-o` | Output meta-directory (required) |
| `--workers` | `-w` | Parallel worker count (default `1`) |
| `--max-cost` |  | Total batch budget, USD (`0` = unlimited) |
| `--max-cost-per-run` |  | Per-run cost cap (overrides agent config) |
| `--model` | `-m` | Model override |
| `--model-class` |  | Model class override |
| `--config` | `-c` | Config file(s) or `key=value` overrides |
| `--resume` |  | Resume from existing `summary.json` and skip already-seen run IDs |

## Try it with the example file

```bash
prophet batch \
  -f examples/example_batch_job.jsonl \
  -o outputs/batch-demo \
  -w 4 \
  --model-class litellm \
  --model gemini/gemini-3-flash-preview
```

Resume a partial run:

```bash
prophet batch \
  -f examples/example_batch_job.jsonl \
  -o outputs/batch-demo \
  --resume
```

Set per-run timeout to 5 minutes:

```bash
prophet batch \
  -f examples/example_batch_job.jsonl \
  -o outputs/batch-demo \
  -c batch.timeout=300
```

## Input format (one JSON object per line)

Required fields:

- `title` (string)
- `outcomes` (array of strings)

Optional fields:

- `run_id` (string): unique id for this run (auto-generated as `run_<n>` if omitted)
- `ground_truth` (object): used for evaluation metrics
- `end_time` (string): parsed and forwarded as search upper-bound date (`search_date_before`)

Example line:

```json
{"title": "Will X happen?", "outcomes": ["Yes", "No"], "run_id": "x-forecast", "end_time": "2026-03-01T00:00:00Z", "ground_truth": {"Yes": 1, "No": 0}}
```

Validation behavior:

- invalid JSON raises an error with line number
- missing `title`/`outcomes` fails validation
- duplicate `run_id` fails validation

`end_time` behavior:

- accepts common datetime/date strings
- converted internally to `MM/DD/YYYY`
- forwarded to search backends as `search_date_before`

## Resume mode (`--resume`)

When `--resume` is enabled, batch startup checks `<output_dir>/summary.json`:

- if the file exists, all run IDs already present in summary are skipped
- if output dir or summary does not exist, resume behaves like a fresh run
- if summary contains any run ID not present in the current input file, batch exits with an error

This strict check prevents mixing results from different input sets.

## Execution model

```mermaid
flowchart LR
    A[Load JSONL problems] --> B[Queue]
    B --> C[Worker 1]
    B --> D[Worker 2]
    B --> E[Worker N]
    C --> F[Per-run output dir]
    D --> F
    E --> F
    F --> G[summary.json]
```

Operational details:

- workers consume jobs from a shared queue
- rate-limit errors can be retried (up to max retries)
- global max cost can stop later runs (`skipped_cost_limit`)
- per-run timeout is enforced (`batch.timeout`, default `180s`)
- timed-out runs are reported as `BatchRunTimeoutError`
- each run still writes `info.json` + `trajectory.json`

## Timeout config

Default config:

```yaml
batch:
  timeout: 180
```

Notes:

- value is in seconds
- `0` disables timeout
- applies per run (not whole batch)

## Output layout

Given `-o outputs/batch-demo`:

- `outputs/batch-demo/summary.json`
- `outputs/batch-demo/runs/<run_id>/info.json`
- `outputs/batch-demo/runs/<run_id>/trajectory.json`

`summary.json` includes status, costs, submission/evaluation, and output path per run.

In resume mode, `summary.json` keeps previous completed runs and appends newly processed runs.
