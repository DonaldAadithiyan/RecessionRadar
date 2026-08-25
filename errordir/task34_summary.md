# Task 34 — Fixing the Null: Isolating Directional Specificity

> **⚠ SUPERSEDED IN PART — see `task36_3_correction.md` (Task 36).**
> The four **ridge** rows in this report are an instrument artifact. Task 36
> measured δᵢ directly on those fits and found **one unique value across 400 test
> points** (CV ≈ 2×10⁻¹⁵): the ridge prediction path is three composed affine
> maps, so δᵢ has no xᵢ dependence and Spearman(δ, e) is tie-degenerate. The
> conclusion drawn from them — *"Check 2 certifies magnitude alignment, not
> difficulty relevance"* — is **withdrawn**.
> The **tree** results (0/16 under DS), the synthetic verification, and the n=40
> power finding are **unaffected and stand**.


## **No. The corrected test recovers zero tree-model validations — 0/16 at both n=40 and n=400. The null-discrimination hypothesis is confirmed as a real defect in Check 2, but it is not the explanation for the tree failures.**

## What the task established, in order

**Item 1 — the statistic.** Directional Specificity (DS) = Spearman correlation
between a direction's per-point induced change and realized per-point error,
against a **magnitude-matched** null: every random direction is rescaled (and
recomputed, not linearly scaled) so its mean induced change equals the
candidate's. Magnitude drops out; only the cross-point *pattern* survives. Stated
in full before any application.

**Item 2 — verification, both directions, all three cases correct.**

| case | requirement | DS percentile | verdict |
|---|---|---|---|
| known signal | DETECT | **1.000** | correct |
| known null | REJECT | 0.639 | correct |
| known null, **hyper-responsive** | REJECT | 0.463 | correct |

Case C used a deeper, larger tree with **7× the responsiveness** of case A — the
exact condition Task 33 found breaks Check 2 — and DS still correctly rejected.

**And it produced the sharpest evidence in this line of work that Check 2 was
broken:** Check 2 scores **0.000 on the known-signal case**, ranking the literal
generating direction of the heteroscedasticity *below every random direction*.
Task 33's diagnosis was right, demonstrated now on ground truth rather than
inferred from repeated failures.

**Item 3 — the retest, and a power flaw I had to fix mid-task.**

The first run inherited Check 2's n=40 test points. For a Spearman-based
statistic that gives SE = 0.164 and a pass bar of **|DS| > 0.270** — larger than
the biggest value observed on any real fit. The test could not pass anything. I
re-ran the full table at **n=400** (bar 0.083).

The rerun is what makes the result trustworthy:

| | median &#124;DS&#124; | pass bar |
|---|---|---|
| n=40 | 0.135 | 0.270 |
| n=400 | **0.044** | 0.083 |

**DS shrank 67% as n grew 10×.** Real-but-underpowered signal would have
persisted and cleared the lower bar. This is noise averaging out. Tree DS at
n=400: median −0.044, range −0.124 to +0.106 — indistinguishable from zero.

## The guardrail's red flag, resolved

Four ridge fits passed Check 2; **none passes DS**. The guardrail required
revisiting the correction rather than accepting this.

Investigation shows it is not a broken correction. Check 2 and DS measure
different things, and a direction can dominate on induced *magnitude* (β on
ridge: Check 2 percentile 0.995) while carrying no cross-point *difficulty
information* (DS −0.037). Correctly powered, DS finds no directional specificity
on the ridge fits either.

That reframes what a Check 2 pass ever certified: **magnitude alignment, not
difficulty relevance.** The four ridge passes across Tasks 24–33 were weaker
evidence than they appeared.

## Where this leaves the tree question

| hypothesis | status |
|---|---|
| Trees are unresponsive to perturbation | **Falsified** (Task 33: zero-rate 0.00003, 2.7× ridge) |
| Check 2's magnitude null can't discriminate on trees | **Confirmed as a defect** (Item 2: 0.000 on known signal) |
| ...and that defect explains the 0/20 tree record | **Falsified** (Item 3: 0/16 under a magnitude-blind test) |
| Tree fits carry a difficulty direction at all | **No evidence** — DS ≈ 0 at n=400 |

The remaining possibilities have narrowed considerably. Two mechanisms have now
been ruled out as *explanations* while being confirmed as *real phenomena*, which
is the useful kind of narrowing. What stands is the plainest reading: on these
fits, no direction in feature space carries information about where tree-model
errors are large — not because the test can't see it, but because it isn't there.

**Task 32's "linear-only" scope stands, now on much firmer ground.** It survived
a test built specifically to overturn it, verified on ground truth in both
directions.

## For any future task using this instrument

Per the guardrail: use **this task's verified DS test at n ≥ 400**, not the
original Check 2, and not DS at n=40. The n=40 configuration is documented here
as a power failure so it is not repeated.

## Files

- `task34_1_test_design.md` — statistic and matching procedure, stated in advance
- `task34_2_synthetic_verification.py` / `.md`, `task34_2_synthetic.csv`
- `task34_3_retest.py` / `.md`, `task34_3_retest.csv` (n=40),
  `task34_3_retest_n400.csv` (n=400)
