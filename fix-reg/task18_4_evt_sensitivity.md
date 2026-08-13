# Task 18 Item 4 — EVT Threshold Sensitivity

**Verdict: the GPD fit is unstable. The tail index ξ swings from +1.61 to −1.93
across thresholds and flips sign in every horizon. This means EVT-tail's numbers
in the paper are threshold-dependent artefacts of a badly-specified fit, and the
method should be reported as such rather than as a fair representative of
extreme-value approaches.**

Script: `fix-reg/task18_13457_recession_items.py`.
Data: `task18_4_evt_sensitivity.csv`.

---

## Tail-index stability — the standard GPD diagnostic

A well-specified GPD has ξ roughly **flat** across thresholds above the point
where the asymptotic approximation kicks in. Here it is anything but:

| Horizon | q=0.80 | q=0.85 | q=0.90 | q=0.95 |
|---|---|---|---|---|
| Current | +0.509 | +0.082 | −0.127 | −1.120 |
| 1M | **+1.611** | +0.286 | −0.613 | −1.373 |
| 3M | +0.395 | −0.006 | −0.228 | −1.722 |
| 6M | +0.136 | −1.122 | −1.434 | **−1.928** |

**ξ flips sign in all four horizons** and spans a range of 2–3 units. Sign
matters here: ξ > 0 is an unbounded heavy tail, ξ < 0 is a bounded one with a
finite endpoint. The fit cannot decide which regime the data is in.

The cause is exceedance count. At q = 0.95 only ~13 of 254 calibration scores
exceed the threshold — far too few for a stable three-parameter fit — and the
strongly negative ξ values are the classic small-sample artefact.

## Coverage across thresholds

| Horizon | q=0.80 | q=0.85 | q=0.90 | q=0.95 |
|---|---|---|---|---|
| Current | 93.8 | 93.8 | 93.8 | 93.8 |
| 1M | 87.5 | 89.1 | 90.6 | 89.1 |
| 3M | 90.3 | 90.3 | 90.3 | 90.3 |
| 6M | **83.0** | **91.5** | 88.1 | 84.8 |

Coverage is stable at Current and 3M but moves **8.5 points** at 6M
(83.0 → 91.5) purely from the threshold choice. The paper's published EVT-tail
6M number uses q = 0.80 and reports 66.1% (Phase 3b, in-sample) / 83.05%
(Task 7, out-of-fold) — and a different, equally defensible threshold would have
given 91.5%.

## What this changes

1. **EVT-tail's reported failure is partly a fitting artefact.** The paper
   concludes EVT-tail "hurts" (Phase 3b) and is worst on both axes (Task 14,
   110–199× compute). That conclusion holds *for the implementation as
   configured*, but the threshold sensitivity shows the method was never given a
   stable fit to work with.
2. **Report the sweep, not just the point.** One sentence and the ξ table make
   this honest: *EVT-tail's performance is threshold-sensitive, and its tail
   index is not stable across thresholds on these score pools, so its result
   should be read as specific to the q = 0.80 configuration rather than as a
   general verdict on extreme-value calibration.*
3. **It does not rescue EVT-tail.** Even at its best threshold (q = 0.85, 91.5%
   at 6M) it does not reach the selector's 96.61%, and it still refits a GPD at
   every step (Task 14: 110–199× baseline compute).

## Honest caveats

- **Only four thresholds** (0.80, 0.85, 0.90, 0.95). A finer sweep would map the
  instability better but would not change the sign-flipping conclusion.
- **The exceedance counts are not reported per cell in this table** but drive the
  instability; at N = 254 they range from ~51 (q = 0.80) to ~13 (q = 0.95).
- **A Hill estimator or a threshold-selection procedure** (mean residual life
  plot, automated stability selection) would be the principled fix. Neither was
  attempted — the spec asked for sensitivity, not for a better EVT method.
- **Recession testbed only**, per the spec.
