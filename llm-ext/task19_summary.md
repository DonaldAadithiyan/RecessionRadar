# Task 19 — Does the Diversity Mechanism Transfer to LLM Calibration?

## Verdict

**Worth pursuing, but only after acquiring one specific thing: an LLM eval set
with per-example correctness and a confidence signal. The cheap synthetic gate
(Item 1) passed cleanly and told us the mechanism's structural precondition is
coherent for LLM-style scores. Items 2-4 did not run, and not because of a
negative result — this project contains no LLM data of any kind, and every way
to proceed without it would have violated an explicit guardrail. The honest
status is: the gate is green, and the extension is blocked on one input.**

Concretely: Item 1 cost one script and reproduced the paper's own falsification
signature in an LLM-styled setting. That is the result this task was scoped to
produce most cheaply, and it came out positive. But a passing synthetic gate is
evidence that the mechanism *could* transfer, not that it does — the synthetic
generator has correlated severity because it was built with it. Whether real LLM
scores do is untested and is the entire question.

## Item-by-item

| Item | Status | Finding |
|---|---|---|
| 1 — Synthetic gate | **Run, passed** | Correlated-severity generator reproduces "support width beats hard-count" in 60/60 cells (median gap +0.408). Independent-draw generator does not — wrong-signed in 13/60, median gap **-0.059** in the high-severity regime where real domains sit. |
| 2 — Real signal | **Not run** | No LLM eval set exists in this project. Blocked by the item's own "no new model calls / no new labeling" guardrail. |
| 3 — Headroom | **Not run** | Defined on Item 2's data; no data, no baseline to measure. |
| 4 — Score sensitivity | **Not run** | Gated on Items 1-3 showing a real effect, as specified. |

## What Item 1 actually established

The two generators separate exactly as the paper's Section 3.5 predicts. The
independent-draw construction — the paper's original, later-falsified one —
inverts at high severity coupling, the same wrong-sign failure Task 16C
diagnosed, now reproduced with LLM-styled score semantics. The correlated
construction is positive everywhere and strengthens as coupling rises.

Two details raise confidence that this is a real reproduction rather than a
coincidence of my re-implementation:

- The correlated generator's output matches `task17_4_correlated_synthetic.csv`
  **exactly, max |difference| = 0.000 across all 60 shared cells.** The
  functional form and seeding are faithful, not re-derived.
- The fixed-size ablation explains *why* count looks informative under the
  independent generator: forcing hard examples in mechanically drags support
  width up (1.87 -> 2.22), yoking the two. Under the correlated generator width
  is already near-saturated (2.52 -> 2.61) and coverage barely moves. That is
  the mechanism working as described, not just a correlation flipping sign.

One caveat I want on the record: the independent generator's *headline* number
is 78% of cells positive, which in isolation looks like partial agreement. It
isn't. That average is dominated by the low-severity corner; in the Delta_S >= 2.0
region where all four real domains were measured, it drops to 39% positive with a
negative median. Reading the headline alone would have produced a "both
generators roughly agree" conclusion that the plane structure contradicts.

## Why Items 2-4 were left undone rather than worked around

An exhaustive sweep found only FRED macro series, UCI diabetes readmission, and
NOAA storm data — all tabular — and zero LLM tooling (no `openai`, `anthropic`,
`huggingface`, `transformers` imports anywhere). Three workarounds were available
and all were rejected: querying a model (forbidden), labeling data (forbidden),
and relabeling a tabular domain's residuals as "LLM-style" scores. The third is
worth naming explicitly because it would have produced complete-looking
deliverables for all four items — and it is the same category of error as
reporting an in-sample number as out-of-fold. Items 3 and 4 would have inherited
the mislabeling and looked like corroboration.

## Recommended next step

Acquire one eval file with per-example correctness plus a confidence/probability
or scalar score; a few hundred rows is enough, since the diagnostic runs at
N=254. No new model calls are needed if cached eval outputs, a public eval dump
with confidences, or logged traffic with correctness labels already exist. The
Item 1 diagnostic is written against a `(scores, is_hard)` pair, so Item 2 needs
only a loader swapped in — it is a small amount of work once the data is there.

Expect a *smaller* effect than the recession domain when it runs. Well-calibrated
LLM confidences on standard benchmarks often already sit near 90% coverage at
alpha=0.10, which is the paper's low-headroom healthcare regime — where the
effect was weakest and least clean. That is a reason to size expectations
correctly in advance, not a reason to go looking for a lower-coverage eval set.

## Files

- `task19_1_synthetic.py` — generators + three-part diagnostic (Item 1)
- `task19_1_synthetic.csv` — 120 cells, both generators
- `task19_1_ablation.csv` — fixed-size ablation
- `task19_1_verdict.csv` — per-generator summary
- `task19_1_synthetic.md` — Item 1 report
- `task19_2_real_signal.md` — Item 2 gap report
- `task19_3_headroom.md` — Item 3 gap report
- `task19_4_score_sensitivity.md` — Item 4 gap report
