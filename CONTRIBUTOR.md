# Contributor Guide

This document collects contributor-facing instructions for testing and CI.

## Local Developer Setup

Install editable package with dev dependencies:

```bash
pip install -e ".[dev]"
```

`dev` includes optional search backends (`perplexityai`, `exa-py`) so test collection does not fail.

## Running Tests Locally

Run the default suite (fast, no real paid API calls):

```bash
python -m pytest -q
```

Run only live API tests (explicit opt-in):

```bash
export OPENROUTER_API_KEY="..."
export MINIPROPHET_TEST_OPENROUTER_MODEL="openai/gpt-4o-mini"
python -m pytest -q -m live_api --run-live-api
```

Notes:
- `live_api` tests are skipped unless `--run-live-api` is provided.
- `-m live_api` by itself is not enough.

## GitHub Actions CI Model

The repo uses two lanes:

1. `unit-tests` (required)
- Workflow: `.github/workflows/ci.yml`
- Trigger: push to `main`, and all pull requests.
- Command: `python -m pytest -q -m "not live_api"`

2. `live-api-gate` (required check with label-controlled strictness)
- Workflow: `.github/workflows/live-api-gate.yml`
- Trigger: `pull_request_target` events.
- Behavior:
  - If PR does **not** have label `live-api-required`, check passes immediately.
  - If label is present, it runs `python -m pytest -q -m live_api --run-live-api`.

This allows live tests to be opt-in while still enforcing them once requested.

## Maintainer PR Commands

Workflow: `.github/workflows/live-api-command.yml`

On a PR conversation:

- Enable live test gate:

```text
/test-live
```

- Disable live test gate:

```text
/test-live-off
```

Authorization:
- Only `OWNER`, `MEMBER`, and `COLLABORATOR` associations can run these commands.

## Required Repository Secrets

For live API tests in GitHub Actions:

- `OPENROUTER_API_KEY`
- `MINIPROPHET_TEST_OPENROUTER_MODEL`

## Branch Protection Recommendation

Set required status checks on the main branch to:

- `unit-tests`
- `live-api-gate`

This preserves strictness after `/test-live` while keeping default PR flow fast.
