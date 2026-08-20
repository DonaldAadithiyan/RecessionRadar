# Task 19, Item 2 — Real Signal Validation: NOT RUN (no qualifying data)

**Status: blocked on data availability, not on a negative result.**

Item 1's correlated generator succeeded, so Item 2 was triggered. It cannot be
completed under its own guardrail.

## The constraint

Item 2 requires "any existing LLM eval set already on hand with per-example
correctness and a confidence or score signal (**no new labeling or new model
calls needed — this should use data that already exists**)."

## What the project actually contains

An exhaustive sweep of the repository (all `.csv`, `.json`, `.jsonl`, `.parquet`
outside `node_modules`/`venv`, plus every `.py` for LLM tooling imports) found:

- `data/raw/` — 13 FRED macroeconomic series (yield curve, UNRATE, CPI, INDPRO, ...)
- `data/domains/uci_diabetes_130.csv` — UCI 30-day readmission, tabular
- `data/domains/noaa_panel.csv`, `data/domains/noaa/` — NOAA storm intensity, tabular
- `data/combined`, `data/feature-engineered`, `data/processed`, `supplementary/` — derivatives of the above

There is **no LLM eval set**: no per-example correctness records, no confidence
or logprob signal, no prompt/response pairs. There is also no LLM tooling
anywhere in the project — zero matches for `openai`, `anthropic`, `huggingface`,
`transformers`, `tokenizer`, or `prompt` across all project `.py` files.

The three domains in this project are all tabular regression/classification. None
of them is an LLM setting, and none can be recast as one without fabricating the
per-example correctness-and-confidence structure the item requires.

## Why this was not worked around

Three available workarounds were each rejected as violating an explicit guardrail:

1. **Generate an eval set by querying a model** — forbidden: "no new model calls needed."
2. **Label existing data to manufacture correctness records** — forbidden: "no new labeling."
3. **Relabel a tabular domain's residuals as an "LLM-style" score** — this would
   report a tabular result under an LLM heading. It is the same category of error
   as reporting an in-sample number as out-of-fold, and it would make Items 3-4
   meaningless since they would inherit the mislabeling.

Per the task's guardrail that "a negative or null result at any item is a valid,
complete answer," this is reported as a gap rather than filled.

## What would unblock this

Any eval set with one row per example carrying (a) a correctness outcome and
(b) a confidence/probability or scalar score. A few hundred rows suffices — the
diagnostic runs at N=254 with 200 draws. Candidates that need no new model calls
include cached eval outputs from prior work, a public eval dump with model
confidences, or any logged production traffic with correctness labels already
attached. Given such a file, the Item 1 diagnostic code runs on it directly: the
three-part diagnostic in `task19_1_synthetic.py` is written against a
`(scores, is_hard)` pair and needs only a loader swapped in.
