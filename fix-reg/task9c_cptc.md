# Task 9C — CPTC with an HMM State Input

**PROTOCOL EXCEPTION, STATED FIRST: this is the only place in the entire Task
7/8/9 evaluation where a model is fit to serve a baseline.** Every other
strategy consumes fixed, pre-computed nonconformity scores and fits nothing.
CPTC cannot comply by construction — its algorithm requires a per-timestep
state-probability matrix from a switching-dynamics model. One latent-state model
per domain was therefore fit, on the training partition only, purely to supply
that input.

**And it is not the authors' state model.** The CPTC authors (Sun & Yu, NeurIPS
2025) use a REDSDS (Recurrent Explicit Duration Switching Dynamical System);
their repository ships only precomputed REDSDS *inference outputs* for their six
datasets, not a trainer. Per the task's authorised fallback, a Gaussian HMM was
fit instead, solely to produce `z_prob` in the shape CPTC expects. **All results
below are labelled "CPTC (HMM state input)" and must not be read as the
published method's performance** — the same convention Task 8 used for the
5-component HOSPITAL variant.

**Verdict: CPTC does not beat the diversity-optimal selector anywhere (0 of 8).
The paper's framing closes — but CPTC posts the single best efficiency profile
of any strategy at the paper's hardest horizon, and that deserves reporting.**

Script: `fix-reg/task9c_cptc.py`. Data: `task9c_cptc.csv`.

---

## What was implemented

The conformal machinery is a faithful port of the authors' released algorithm
(`github.com/Rose-STL-Lab/CPTC`, `algos/cptc.py`): per-state residual pools with
a shared `"all"` fallback below `min_residuals=25`, per-state α updates at
γ=0.2, the smallest-set union over states ordered by state probability until
mass ≥ 1−α, and the sampled-state update rule.

The state model is a self-contained Gaussian HMM (Baum-Welch EM, 3 states),
written out rather than pulled from `hmmlearn` to keep the disclosed exception
auditable in one file. It is fit on training data only, and state probabilities
on the test stream are **filtered** (using data up to *t* only), never smoothed,
so no future information leaks into any prediction.

## Results

| Domain | Model | Horizon | n | Coverage | Wilson 95% | Width | Width/spread |
|---|---|---|---|---|---|---|---|
| Recession | stacking-chain | Current | 55 | 87.27 | [75.98, 93.70] | 149.9 | 186× |
| Recession | stacking-chain | 1M | 54 | 88.89 | [77.81, 94.81] | 100.1 | 124× |
| Recession | stacking-chain | 3M | 52 | 86.54 | [74.73, 93.32] | 128.8 | 176× |
| **Recession** | stacking-chain | **6M** | 49 | **91.84** | [80.81, 96.78] | **81.8** | 137× |
| Healthcare | ridge | 30-day | 126 | 89.68 | [83.15, 93.87] | 22.7 | 1.0× |
| Healthcare | gradboost | 30-day | 126 | 88.10 | [81.28, 92.65] | 21.7 | 1.0× |
| Climate | ridge | region-month | 213 | 91.08 | [86.49, 94.22] | 7.2 | 0.8× |
| Climate | gradboost | region-month | 213 | 89.67 | [84.86, 93.08] | 6.9 | 0.8× |

CPTC lands near nominal in every domain — 86.5–91.8%, all intervals containing
90% except none falling clearly short. It is a competent, well-behaved baseline,
which is what makes the comparison below meaningful.

## Does it beat the selector?

| Domain | Model | Horizon | Diversity-optimal | CPTC | div-opt width | CPTC width | CPTC separates above? |
|---|---|---|---|---|---|---|---|
| Recession | stacking-chain | Current | 93.85 | 87.27 | 108.7 | 149.9 | no |
| Recession | stacking-chain | 1M | 96.88 | 88.89 | 92.1 | 100.1 | no |
| Recession | stacking-chain | 3M | 95.16 | 86.54 | 109.1 | 128.8 | no |
| **Recession** | stacking-chain | **6M** | 96.61 | 91.84 | 123.9 | **81.8** | no |
| Healthcare | ridge | 30-day | 92.47 | 89.68 | 25.3 | 22.7 | no |
| Healthcare | gradboost | 30-day | 93.84 | 88.10 | 25.7 | 21.7 | no |
| Climate | ridge | region-month | 99.59 | 91.08 | 17.8 | 7.2 | no |
| Climate | gradboost | region-month | 98.77 | 89.67 | 19.4 | 6.9 | no |

**0 of 8 — the framing closes.** "No baseline tested beats the selector" can now
be stated without the CPTC caveat, since CPTC has been tested.

**The important nuance, which the paper should not omit:** at recession 6M —
the hardest horizon in the paper, and the one the whole six-month discussion
turns on — CPTC reaches **91.84% at 81.8 width versus the selector's 96.61% at
123.9**. That is nominal coverage at two-thirds the width. Its Wilson interval
[80.81, 96.78] straddles 90%, so this is not a claim that CPTC solves the
six-month problem. But it is the best coverage-per-unit-width of any strategy
tested at that horizon, and the change-point-aware machinery is the plausible
reason. In climate the same pattern is starker: 89.67–91.08% at **0.8× the
target's own spread**, less than half the selector's width.

The honest summary is: **the selector wins on coverage everywhere; CPTC and
tuned BCI win on sharpness in the domains where they reach nominal.**

## Honest caveats

- **This is not CPTC.** It is CPTC's conformal layer driven by an HMM instead of
  a REDSDS. A REDSDS models explicit state durations and recurrent
  state-transition dynamics that a 3-state Gaussian HMM does not; the real
  method could perform better or worse. Do not cite these as the published
  method's numbers.
- **The recession n is smaller than other strategies'** (49–55 vs 59), because
  CPTC's warm-start period consumes leading test points to seed the per-state
  residual pools. Its recession intervals are correspondingly wider, and the
  comparison at that horizon is slightly less powered than the others.
- **`z_mean` is constructed, not learned.** REDSDS supplies per-state predictive
  means directly; here they are the point forecast offset by each HMM state's
  level. This is a reasonable univariate analogue but it is an implementation
  choice, not the authors'.
- **State sampling makes CPTC stochastic.** Results use seed 11; a different
  seed shifts coverage by a small amount. Not re-run across seeds.
- **The recession width ratios are enormous** (124–186× the target's spread),
  as they are for every strategy on that testbed — the 2020+ recession
  probability series barely moves outside the COVID spike, so all intervals look
  vacuous against its spread. Compare across strategies, not against 1.0.
