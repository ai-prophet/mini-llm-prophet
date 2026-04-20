# Subagents

Subagents are lightweight agents spawned by the **main (orchestrator) agent**
to handle delegated work — reading sources, investigating sub-problems — so
the main agent's context stays focused on planning and synthesis.

## Why subagents?

Before subagents, the main agent did everything itself: searches, full-source
reads, multi-step reasoning about sub-problems. Each deep read flooded its
context with raw source text; aggressive truncation then threw that
information away.

With subagents the main agent is an **orchestrator**: it plans, delegates,
and synthesizes. It never sees raw source content — only focused summaries
and sub-problem probabilities returned by subagents.

```
Main agent (execution)
  tools: [search, list_sources, read_source, investigate_subproblem, submit]
    │
    │  read_source(id, focus="...") ─────▶  SourceReadingAgent  ─▶  summary
    │
    │  investigate_subproblem(title,  ───▶  SubproblemAgent    ─▶  probability + report
    │      context, source_ids)
```

All agents share a single async-safe `SourceRegistry`: sources found by a
subagent are visible to everyone via `list_sources`.

## Types & capabilities

### SourceReadingAgent

Reads **one** source and returns a concise, focused summary.

| Aspect | Value |
|--------|-------|
| Tools available | `retrieve_source`, `submit_summary` |
| Search access | **No** — cannot search the web |
| Typical run | 2 model calls (retrieve → summarize) |
| Output | Summary text (3–6 sentences) |
| Spawned via | Main agent's `read_source(source_id, focus=None)` |

The main agent can pass an optional `focus` instruction (e.g. "report any
mention of player injuries") to steer the summary toward what matters for
the current line of investigation.

### SubproblemAgent

Investigates **one** binary yes/no sub-problem end-to-end and returns a
probability.

| Aspect | Value |
|--------|-------|
| Tools available | `search`, `list_sources`, `retrieve_source`, `submit_subproblem` |
| Search access | **Yes** — has its own small search budget |
| Typical run | 5–10 model calls |
| Output | Probability in [0, 1] + brief report (3–6 sentences) |
| Spawned via | Main agent's `investigate_subproblem(title, context, source_ids=[])` |

Sub-problem titles must be phrased as binary yes/no questions. The main
agent typically populates these from the planning phase (`<sub_problem>`
entries in the XML plan) and passes along any relevant source IDs it has
already discovered.

## Access control

Tool access is enforced at the **environment level**, not in prompts:

- The main agent's environment contains the spawner tools (`read_source`,
  `investigate_subproblem`) but **not** the raw `retrieve_source` tool.
  Calling `retrieve_source` on the main agent returns an `Unknown tool`
  error from the harness.
- Subagents **cannot spawn further subagents**: their environments do not
  contain the spawner tools. No recursive nesting.

This invariant is structural — you cannot bypass it by prompting the model
to try harder.

## Config

Subagents are configured under `agent.subagents` in `default.yaml` (or
your config override). Both types support these fields:

| Field | Default | Purpose |
|-------|---------|---------|
| `enabled` | `true` | If `false`, the corresponding spawner tool is omitted from the main agent's tool set |
| `step_limit` | 3 / 10 | Per-invocation hard cap on model calls |
| `cost_limit` | 0.05 / 0.5 | Per-invocation hard cap on combined model + search cost ($USD) |
| `system_template` | (see YAML) | System prompt for the subagent |
| `model` | `null` | Model override (`null` = inherit from `agent.model`) |

`SubproblemAgent` has one additional field:

| Field | Default | Purpose |
|-------|---------|---------|
| `search_limit` | 3 | Subagent's own search budget (separate from main agent's) |

### Model inheritance and override

By default, subagents reuse the **main agent's** model configuration
(`agent.model.model_class`, `model_name`, `model_kwargs`). To run a cheaper
or faster model for subagents, provide an override:

```yaml
agent:
  subagents:
    source_reading:
      model:
        model_class: litellm
        model_name: "gemini/gemini-flash-lite"
```

Only the keys you specify are overridden; everything else is inherited.

### Example config snippet

```yaml
agent:
  subagents:
    source_reading:
      enabled: true
      step_limit: 3
      cost_limit: 0.05
      # model: null  (inherit)
      system_template: |
        You are a source-reading assistant for a forecasting system...

    subproblem:
      enabled: true
      step_limit: 10
      cost_limit: 0.5
      search_limit: 3
      system_template: |
        You are a sub-problem investigator for a forecasting system...
```

## Cost and limit model

Subagent costs enforce **both** a per-invocation cap and aggregation into
the main agent's totals:

- **Per-invocation cap**: when a subagent hits its own `cost_limit` or
  `step_limit`, it stops and returns a partial result with
  `exit_status: SubagentLimitsExceeded`.
- **Aggregate**: a subagent's `model_cost` and `search_cost` are returned
  in its tool result and added to the main agent's running totals. They
  count toward `agent.cost_limit` — so the main agent's budget bounds the
  total spend across every delegated call.

## Display

### CLI mode

While a subagent is running, the CLI shows a compact **one-line live status**
under the main agent's current step:

```
  ├─ ⟳ source_reading [S4, focus=injury]  ·  Step 2/3  ·  calling submit_summary  ·  $0.0023  ·  2.4k tok
```

When the subagent finishes, the live line disappears and a summary is
printed in its place, followed by the main agent's observation for that
tool call:

```
  ╰─ ✓ read_source [S4, focus=injury]  ·  2 steps  ·  2 tool calls  ·  $0.0023  ·  3.9k tok

  ╭─ Observation ────────────────────────────────────────╮
  │ LeBron played 82 games this season, no reported     │
  │ injuries.                                            │
  ╰──────────────────────────────────────────────────────╯
```

Toggle via `agent.show_subagent_trace` in the config (default: `true`).

### Batch / eval mode

In batch mode (`prophet eval`, `ai-prophet` integrations, any
non-interactive caller) no Rich display is attached to subagents — they
run silently. Summaries, costs, and trajectories are still captured in
the saved artifacts for later inspection.

## Parallel subagents

When the main agent emits multiple tool calls in one step (e.g. three
`read_source` calls at once), subagent tools run concurrently via
`asyncio.gather`. Each subagent has its own model instance and its own
environment; they only share the `SourceRegistry` (which is async-safe).

In CLI mode only one live status displays at a time (to avoid competing
cursor redraws); concurrent subagents run silently until they finish, at
which point each prints its summary in the order their `await` completed.

## Under the hood (minimal)

- `src/miniprophet/subagents/` — agent classes (`SubagentBase`,
  `SourceReadingAgent`, `SubproblemAgent`) and the `SubagentStatus`
  dataclass. Pure logic, no Rich / CLI dependencies.
- `src/miniprophet/tools/read_source.py`,
  `src/miniprophet/tools/investigate_subproblem.py` — spawner tools. Also
  display-agnostic; they accept an injected `display_context` from the
  caller (CLI passes the live-display context; batch passes nothing).
- `src/miniprophet/cli/components/subagent.py` — all Rich rendering for
  subagent status and summaries lives here.
