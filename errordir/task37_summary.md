# Task 37 — A Real Instrument for Linear Models

## **Item 1: Check 2's ridge pass is NOT construction-trivial — fitted-but-meaningless directions score 0.18–0.48 and pass 0/20, while real β scores 0.995. Item 4: the new HOS instrument VALIDATES both fits behind the positive claim — climate at percentile 1.000, energy at 0.990. The mechanistic claim, open since Task 36, is now supported by two independent non-degenerate instruments.**

## Item 1 — the triviality hypothesis, falsified

The concern was that β, being regression-fit, is geometrically favored on a
linear model, so Check 2 would pass for any fitted direction regardless of
meaning. Testing that with the unmodified Check 2:

| candidate | Check 2 mean (insurance / energy) | **pass rate** |
|---|---|---|
| **β (real)** | **0.995 / 0.995** | **1.00** |
| noise_target (fitted) | 0.395 / 0.182 | **0.00** |
| perm_target (fitted) | 0.409 / 0.482 | **0.00** |
| random_unit (untrained) | 0.525 / 0.533 | 0.10 |

The permutation control is decisive: `perm_target` uses the *exact same error
values* as β, merely shuffled, with an identical fitting procedure. It averages
0.41/0.48 and never passes. **Check 2 on ridge measures difficulty-relevance,
not "was this direction fitted."**

Consequence: **Item 5's banner-adding is not warranted.** Tasks 21, 24–27 and 30
cited ridge Check 2 passes as evidence, and that citation stands.

## Items 2–3 — HOS, specified and verified both ways

**HOS(v) = |Spearman(X_holdout · v, |error|_holdout)|**, tested against a null of
**200 permutation-fitted directions** — same estimator, same alphas, same
features, same error values, shuffled. This never computes a model-output
derivative (so it cannot inherit DS's degeneracy) and holds "was this fitted"
constant between candidate and null (so it cannot inherit the Item 1 confound).

| synthetic case | requirement | HOS | null mean | percentile | correct |
|---|---|---|---|---|---|
| known signal | DETECT | 0.5938 | 0.1577 | **1.000** | ✓ |
| known null | REJECT | 0.0432 | 0.0251 | 0.825 | ✓ |

## Item 4 — the decisive result

| fit | n_fit | n_holdout | **HOS(β)** | null mean | **percentile** | validated |
|---|---|---|---|---|---|---|
| **climate/ridge/β** | 597 | 344 | 0.3583 | 0.1175 | **1.000** | **Yes** |
| **energy/ridge/β** | 1734 | 1000 | 0.4023 | 0.1499 | **0.990** | **Yes** |

Climate's β beats all 200 permutation directions; energy's beats 198 of 200.
HOS sits at **2.7–3.0× the null mean** in both.

Two independent, non-degenerate instruments now support the mechanism by
different routes: **Check 2** says perturbing along β moves the prediction more
than fitted-but-irrelevant directions do; **HOS** says β *orders points by
difficulty* better than they do. DS, the one that failed, was structurally
incapable of measuring either (Task 36).

## Honest qualifications

1. **HOS is correlational.** It does not restore the causal-adjacency framing
   that degenerated on linear models — it tests specificity, which is the
   appropriate question there.
2. **The known-null synthetic landed at 0.825**, not near 0.5, so the instrument
   has less headroom than ideal. Both real results sit far clear of that, but a
   future result near 0.95 should be treated cautiously.
3. **n=1 per domain for β** — a valid permutation p-value, but no within-domain
   replication.
4. **Nothing here touches tree models.** The 0/20 tree record stands.

## Item 5 — record updated

No banners added (Item 1's condition unmet). Task 36's "untested, not exonerated"
status is superseded by Item 4's answer, and a pointer has been appended to
`task36_summary.md`. Task 36's substantive findings — DS's degeneracy, the Task
34 withdrawal, the Task 34 banners — are all unaffected.

## The mechanism-vs-results boundary, restated

As in Task 36, and it applies identically here: **nothing in this task changes
any measured Winkler, coverage, or width number.** Climate's 11.220 vs RLCP's
11.292, energy's 28.775, Task 30's 8×/10× degradation-rate advantage, the
n_cal=50 data-efficiency result — all are empirical outcomes observed directly,
independent of *why* the method works.

What this task settles is the **explanation**. Task 36 left it open; Task 37
closes it in the method's favour on both ridge fits. That strengthens the paper's
framing — it does not alter a single interval it reported.

## Files

- `task37_1_check2_triviality.py` / `.md`, `task37_1_check2_triviality.csv`
- `task37_2_held_out_specificity.py` / `.md`
- `task37_3_synthetic_verification.md`, `task37_3_synthetic.csv`
- `task37_4_climate_energy.md`, `task37_4_real.csv`
- `task37_5_record_update.md`
