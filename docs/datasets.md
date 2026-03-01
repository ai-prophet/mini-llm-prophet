# Dataset Management

`prophet datasets` manages standardized eval datasets.

## Commands

- `prophet datasets list`
- `prophet datasets download`
- `prophet datasets validate`

## `list`

List datasets from the registry.

```bash
prophet datasets list
prophet datasets list --registry-path ./registry.json
prophet datasets list --registry-url https://.../registry.json
```

## `download`

Resolve a registry/HF dataset and cache a JSONL locally.

```bash
# Registry
prophet datasets download weekly-nba@latest

# Hugging Face
prophet datasets download your-org/forecast-bench@main --hf-split train
```

## `validate`

Validate dataset rows against the forecast task schema.

```bash
# Validate local file
prophet datasets validate -f examples/example_batch_job.jsonl

# Validate registry/HF dataset
prophet datasets validate -d weekly-nba@latest
prophet datasets validate -d your-org/forecast-bench@main --hf-split train
```

## Cache location

Dataset artifacts are cached under your global mini-prophet config directory:

- `${MINIPROPHET_GLOBAL_CONFIG_DIR}/datasets/...` if set
- otherwise the platform default for `mini-prophet`
