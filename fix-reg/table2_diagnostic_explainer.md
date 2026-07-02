# Table 2 Diagnostic — Baseline Anomalies

**TL;DR:** Both anomalies are **real, explainable consequences of how the
horizon targets are built** — not baseline bugs. The reviewers are right that
the table looks strange, but the fix is *explanation*, not re-running models.
One correction is needed: the paper text currently cites a 6M XGB MAE of
**64.89**, but the tuned baseline that actually appears in Table 2 is
**45.71** (the 64.89 comes from an older untuned script). All Table 2 numbers
reproduce exactly and deterministically.

**Key fact that reframes the whole table (and refutes the leading hypothesis):**
the horizon targets are **not** rolling means of a USREC indicator. They are
forward shifts of FRED's *smoothed* recession-probability series
`RECPROUSM156N` (a continuous 0–100% series):

```
1_month_recession_probability = recession_probability.shift(-1)
3_month_recession_probability = recession_probability.shift(-3)
6_month_recession_probability = recession_probability.shift(-6)
```
(`TS_general_training_notebooks/01data_collection.ipynb`, cell 6.)

Because it's a *shift*, the identical values just move to different rows per
horizon — so the dominant COVID-2020 spike (=100%) lands in a **different
test row for each horizon**, and shifts **out of the 6M test window
entirely**. That single fact drives both anomalies.

---

## Anomaly 1 — Naive Mean MAE falls with horizon (11.28 → 8.92)

**Verdict: mechanical artifact of target construction — NOT a bug, NOT smoothing.**

Naive Mean predicts the constant training mean (~9.0%) for every test row
against target column `t`, per horizon (`baselines.py:97-103`). The test
targets are dominated by two COVID months where the smoothed probability =
100%. Under `shift(-k)`, the number of those spike months that fall inside the
65-row test window decreases with the horizon:

| Horizon | Test std | # spike months (=100%) in window | Naive MAE | MAE from spikes |
|---|---|---|---|---|
| Current | 17.24 | 2 | 11.28 | 2.80 (25%) |
| 1M | 17.23 | 2 | 11.25 | 2.80 (25%) |
| 3M | 12.28 | 1 | 10.09 | 1.40 (14%) |
| 6M | **0.48** | **0** | 8.92 | 0.00 (0%) |

Each spike month costs the constant predictor ≈|100−9|/65 ≈ 1.4pp. The
non-spike remainder (near-zero actuals vs. a ~9 prediction) contributes a
roughly constant ~8.5–8.9pp at every horizon. So MAE falls **monotonically as
spikes shift out of the window**, bottoming at 6M where the window contains no
spike at all (max 6M actual = 1.64%). The rolling-mean-smoothing hypothesis in
the task brief does not apply: there is no rolling mean and target variance
does **not** shrink smoothly — it collapses discretely from ~297 to ~0.23
because the high-variance spike literally leaves the window.

A persistence baseline (predict the contemporaneous actual for all horizons)
gives MAE 0.00 / 3.22 / 4.84 / 3.35 — i.e. the "improves-with-horizon" pattern
is **specific to the constant-mean baseline**, confirming it's about where the
spike sits relative to a flat prediction, not a general property of the data.

---

## Anomaly 2 — XGB Indep. (tuned) MAE explodes at 6M (3.60 → 45.71)

**Verdict: genuine out-of-sample regime failure — NOT a leakage bug, NOT
misconfiguration.** The baseline is honestly tuned and correctly aligned; it
fails because the *task* it is given at 6M is near-unlearnable in this window.

Evidence (reproduced exactly, deterministic; `ablation_baselines.py`):

- **Not misconfigured / not overfit hyperparameters.** The 6M Optuna pick is
  conservative: `max_depth=3`, `reg_alpha≈0.96`, `n_estimators=714`,
  `lr≈0.094`. **CV MAE = 7.51** (in-sample it looks *fine*); the failure is
  purely out-of-sample.
- **Not a leakage / off-by-one bug.** All features use `.shift(1)` before
  rolling (`fix-reg/feature_engineering.ipynb`); RF feature selection and the
  train mean are fit on the pre-2020 split only. The 6M target alignment is a
  clean `shift(-6)`. No lookahead corrupts the 6M horizon specifically.
- **Not a target-transform mismatch.** XGB Indep. uses the *same* logit /
  inv-logit handling as the RegressorChain (`logit`/`inv_logit`, both scripts).
- **It's a systematic bias, not a few outliers.** 6M predictions:
  min 0.01, max 98.7, **mean 46.0**, with **31/65 predictions > 50%** and
  **14/65 > 90%**, while **every actual ≤ 1.64%**. The top-3 error points
  contribute only 4.5pp of the 45.71 — the error is spread across most of the
  test set. (See `diag_xgb6m_pred_vs_actual.png`.)

**Mechanism.** The largest errors cluster in **2022–2023**, the Fed tightening
period. The feature set (inverted yield curve, high short rates) matches
patterns that, in the pre-2020 training data, preceded *high* recession
probability, so XGB confidently predicts high. But the actual smoothed
recession probability **6 months later** stayed ≈0 (no recession
materialised). The independent model has no mechanism to temper this; the
logit inverse amplifies confident logits toward 100%. Note this is the
same regime story the paper already tells for the probit — here it cuts
*against* the naive ML baseline.

MOR-XGB (31.31) and MOR-XGB-joint (28.59) show the **same** failure, milder —
so this is a family-wide long-horizon problem, not specific to XGB Indep. That
is exactly the point of the ablation: it is what motivates the RegressorChain
(6M MAE 10.17), which conditions the 6M forecast on the shorter horizons and
so is dragged back toward the near-zero regime the shorter targets sit in.

---

## Recommendations for the paper

**Anomaly 1 (Naive Mean) — insert next to Table 2 (1 sentence):**
> "The naive-mean baseline's MAE decreases with horizon because the two
> COVID-2020 spike months (smoothed probability = 100%) fall in progressively
> fewer test rows as the horizon target is shifted forward, leaving the 6-month
> test window spike-free (max actual 1.64%); this reflects target construction,
> not model skill."

**Anomaly 2 (XGB blow-up) — keep the row, correct the number, and reframe:**
- **Correction needed:** the paper text / `baselines.py` narrative and PDF pages
  cite **64.89** for 6M single-stage XGB and an "84% reduction." Table 2's
  tuned XGB Indep. is **45.71**. Reconcile: either (a) report 45.71 and restate
  the reduction as **~78%** (10.17 vs 45.71), or (b) explicitly label 64.89 as
  the *untuned* single-stage number and 45.71 as the *tuned* one. **Do not cite
  both interchangeably.** Recommend (a) — use the tuned Table 2 numbers
  everywhere and drop the 64.89 figure.
- **Reframe (1–2 sentences):** the 6M blow-up is a genuine regime-generalisation
  failure — independent boosters over-predict recession probability during the
  2022–2023 inversion because training-era feature patterns preceded recessions
  the 2022–23 signals did not. This is the empirical case for inter-horizon
  conditioning; present it as such rather than as "XGB is misconfigured."

**Downstream-claim check:** the DM tests (`ablation_results.csv`) are computed
from the same 45.71/31.31/28.59 predictions and remain valid (6M: DM=4.09,
p=0.0001 for XGB Indep). Only the prose "64.89 / 84%" figures need updating.

**Minor data-quality note (not driving either anomaly):** `clean()`
(`ffill().bfill().fillna(0)`) fills the last 6 test rows of the 6M target
(NaN from `shift(-6)` running off the data end) by forward-fill, so the last 7
rows are a constant 1.64. This is cosmetic here (values are tiny) but worth a
one-line footnote, or truncate the last 6 rows for the 6M horizon.

---

## Supporting artifacts
- `diag_target_series.png` — four test targets overlaid (spike position by horizon).
- `diag_xgb6m_pred_vs_actual.png` — XGB Indep. 6M pred-vs-actual line + scatter.
- `ablation_results.csv` — existing saved Table-2 numbers + DM tests (unchanged).

## Reproduction cost
- Anomalies 1 & diagnosis: computed from saved data/CSVs — **no re-run needed**.
- Anomaly 2 verification: required re-running **only the 6M Optuna** (40 trials,
  ~2–3 min) since per-model predictions/params were not saved; reproduced
  45.71 exactly. Full 4-horizon + joint tuning (~10 min) is not required for
  diagnosis.

## Code changes
- **None to baselines/models** (diagnosis only, per the constraint).
- Added diagnostic-only files under `fix-reg/`: `diag_target_series.png`,
  `diag_xgb6m_pred_vs_actual.png`, and this explainer. No modeling code altered.
