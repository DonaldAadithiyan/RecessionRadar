# Task 36 — Resolving DS on Linear Models

## **Item 1 outcome: DEGENERACY CONFIRMED. All four ridge fits produce exactly one unique δ value across 400 test points (CV ≈ 2×10⁻¹⁵). DS is structurally void on linear models, Item 2 did not run, and Task 34's ridge-based conclusion is withdrawn.**

## Item 1 — direct measurement, with a working control

| class | mechanism | mean δ | **CV** | **unique δ / 400** |
|---|---|---|---|---|
| **ridge** (insurance, energy × β, kNN) | all 4 | 0.099–2.597 | **~2e−15** | **1** |
| lightgbm control (insurance, energy × β, kNN) | all 4 | 0.224–1.475 | 0.49–0.82 | 381–400 |

The tree control is the check that the measurement itself works — it shows
381–400 distinct values, exactly as it must. The ridge rows are not "low
variance"; they are **one number repeated 400 times**.

**The pipeline trace explains it exactly.** Task 34's ridge path is
`Ridge.predict(sc0.transform(scb.inverse_transform(·)))` — three composed affine
maps, with no clipping, flooring, or bound anywhere (verified by grep). So
δᵢ = |2h·(a·v)|, which has **no xᵢ dependence**. The constancy is forced by the
code, not an empirical property of these datasets. The 10⁻¹⁵ is floating-point
noise around an exact identity.

With δ constant, Spearman(δ, e) is tie-degenerate and the reported percentile
reflects tie-breaking, not the fit. Both the statistic and its magnitude-matched
null are void.

## What this resolves

**The urgent branch dissolves.** Energy/ridge/β's DS failure (percentile 0.204)
is an instrument artifact, not evidence against the mechanistic claim. The
foundation of the positive result is not undermined by Task 34.

**But it is not vindicated either.** Climate/ridge/β remains **untested, not
exonerated**. No instrument capable of measuring directional specificity on a
linear model has been applied to either fit. What exists is held-out correlation
(energy β: r = 0.417; climate β: r = 0.333) — real, positive, and correlational,
which is exactly the gap Check 2 was introduced to close and cannot close here.

## Item 3 — the record is corrected

Task 34's claim that *"Check 2 certifies magnitude alignment, not difficulty
relevance"* is **withdrawn**, and both `task34_3_retest.md` and
`task34_summary.md` now carry a superseded banner at the top pointing to
`task36_3_correction.md`. The original framing is marked superseded rather than
left standing beside the correction.

**What still stands in Task 34:** the synthetic verification (DS detects known
signal at 1.000, rejects known null at 0.639/0.463, including a hyper-responsive
case); Check 2 scoring **0.000 on the known-signal case**; the tree retest 0/16
at both sample sizes; and the n=40 power finding. All of those live in the tree
regime where DS is valid. Only the linear-model subsidiary claim is withdrawn.

**How the error arose,** recorded as a reusable lesson: DS was designed for the
tree problem, verified on tree synthetics, then applied uniformly to a 24-cell
table including linear fits. The verification was thorough *within its intended
regime* and silent outside it. One line — `len(np.unique(δ))` — would have caught
it. **An instrument verified on one model class is not thereby verified on
another**, the same discipline this project has applied to mechanisms via ridge
control arms since Task 25.

## Item 4 — what this task does and does not affect

Stated explicitly, because it is easy to lose in a correction:

**Nothing in this task changes any measured Winkler, coverage, or width number
from climate or energy.** Those are empirical outcomes — errordir produced
intervals of a given width at a given coverage, observed directly. Climate's
Winkler 11.220 vs RLCP's 11.292, energy's 28.775, the 8×/10× degradation-rate
advantage in Task 30, the data-efficiency result at n_cal=50 — all unaffected,
because none of them depends on *why* the method works.

What is at stake is the **explanation**: whether "β identifies a genuine
difficulty direction" is the right account of those outcomes. That question is
now known to be open rather than settled in either direction — DS cannot address
it on linear models, and no valid instrument has yet been applied.

A DS failure on a ridge fit must never be read as calling the interval results
themselves into question. It was never capable of that, and after this task it is
not capable of saying anything about ridge at all.

## Files

- `task36_1_degeneracy.py` / `.md`, `task36_1_degeneracy.csv`
- `task36_2_climate_ds.md` — not run, with reason and what remains open
- `task36_3_correction.md` — the withdrawal, plus banners applied to Task 34

---

## Update from Task 37

The question this task left open — *does β genuinely identify a difficulty
direction on the two fits behind the positive claim?* — has since been answered.
Task 37 built and verified **HOS** (held-out specificity against a
permutation-fitted null), an instrument that never computes a model-output
derivative and so does not inherit DS's degeneracy. Both fits **validate**:
climate/ridge/β at percentile **1.000**, energy/ridge/β at **0.990**, with HOS
2.7–3.0× the null mean.

Task 37 Item 1 additionally showed Check 2's ridge pass is **not**
construction-trivial (fitted-but-meaningless directions score 0.18–0.48 and never
pass), so the ridge Check 2 passes cited across Tasks 21–30 stand.

Everything else in this report — DS's degeneracy, the Task 34 withdrawal, the
banners — is unaffected. See `task37_4_climate_energy.md` and
`task37_5_record_update.md`.
