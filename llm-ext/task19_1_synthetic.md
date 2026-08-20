# Task 19, Item 1 — Synthetic Mechanism Test for LLM-Style Scores

**Result: the correlated-severity generator reproduces the paper's pattern; the
independent-draw generator does not. The gate passes.**

Script: `llm-ext/task19_1_synthetic.py`
Data: `task19_1_synthetic.csv` (120 cells), `task19_1_ablation.csv`, `task19_1_verdict.csv`

## What was run

Two generators for LLM-style nonconformity scores, both swept over the same
plane as the paper's Section 3.5 (6 hard-example prevalences x 10 severity
couplings = 60 cells each), each cell evaluated over 200 random calibration
draws at N=254 with the paper's ACI loop (gamma=0.005, alpha=0.10).

**Generator 1 — independent-draw.** Functional form copied unchanged from
`fix-reg/task16c_phase_diagram.py::make_series()`. Hard-example placement is
clustered (length 6); score magnitude is an independent `|N(0,1)|` draw with a
constant `+Delta_S` shift applied to flagged examples. Which example is hard
and how large its score is are independent beyond that constant shift.

**Generator 2 — correlated-severity.** Functional form copied unchanged from
`fix-reg/task17_4_correlated_synthetic.py`:

    Z_t = 0.6*Z_{t-1} + eta_t            persistent latent difficulty
    P(H_t=1 | Z_t) = sigmoid(2.0*Z_t + b)   b solved for target prevalence
    S_t = |N(0,1) + Delta_S * Z_t|          score driven by the SAME Z

**Parameter provenance.** Every parameter (POOL/TEST/N_FIX/N_DRAWS = 635/65/254/200,
gamma, alpha, a=2.0, rho=0.6, CLUSTER=6, both sweep grids, all seeds) is carried
over unchanged from Task 16C / Task 17 Item 4. Nothing was chosen by looking at a
Task 19 result. The ablation operating point (freq=0.12, Delta_S=1.0) is the
centre of the swept plane, fixed before running.

**Fidelity check.** The correlated generator's `gap` reproduces
`fix-reg/task17_4_correlated_synthetic.csv` **exactly — max |difference| = 0.000
across all 60 shared cells**. The construction is faithful, not a re-derivation.

## The three-part diagnostic

### (B) Support width vs hard-count as a coverage predictor

`gap = rho(support_width, coverage) - rho(hard_count, coverage)`, positive when
support width is the better predictor. Median over non-degenerate cells:

| Generator | cells | gap > 0 | median gap | R^2 gap > 0 | median R^2 gap |
|---|---|---|---|---|---|
| independent | 60 | 47/60 (78%) | +0.233 | 46/60 | +0.216 |
| correlated | 60 | **60/60 (100%)** | **+0.408** | 60/60 | +0.243 |

The headline averages understate the separation. Restricting to the
high-severity regime where the real domains actually sit (measured
Delta_S = 1.26-2.97 per Task 16C):

| Generator | Delta_S >= 2.0: median gap | frac > 0 |
|---|---|---|
| independent | **-0.059** | 0.39 |
| correlated | **+0.434** | 1.00 |

The independent generator goes **wrong-signed** in 13 of 60 cells, worst at
freq=0.12 (gap = -0.359 at Delta_S=3.0, -0.352 at Delta_S=4.0) — reproducing the
exact failure Task 16C diagnosed, now in an LLM-styled setting. The correlated
generator is positive in every cell and strengthens as coupling rises.

### (A) Fixed-size ablation (N=254 held constant, hard-count k forced)

| Generator | k=0 | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|
| independent | 89.6 / 1.87 | 89.5 / 1.88 | 90.9 / 1.97 | 92.7 / 2.04 | 95.8 / 2.22 |
| correlated | 93.0 / 2.52 | 93.3 / 2.55 | 93.3 / 2.57 | 94.0 / 2.60 | 94.1 / 2.61 |

(coverage % / mean support width)

Under the independent generator, forcing hard examples in buys a large coverage
swing (+6.2pp from k=0 to k=32) because each forced example mechanically drags
the support width up (1.87 -> 2.22): count and width are yoked by construction,
which is precisely why count looks informative there. Under the correlated
generator, support width is already near-saturated at k=0 (2.52) and barely
moves (-> 2.61), so coverage moves only +1.1pp. Support width, not count, is
carrying the signal.

### (C) Within-support-tertile redundancy of hard-count

Median rho(hard_count, coverage) within support-width tertiles:
**independent +0.214, correlated +0.091.** Conditioning on support width leaves
hard-count with more than twice the residual signal under the independent
generator. Under the correlated generator count is close to redundant once width
is known — the paper's claim.

## Which generator reproduces the paper's pattern

The **correlated-severity** generator, unambiguously: 100% sign agreement,
roughly double the median gap, and hard-count near-redundant within tertiles.

The independent-draw generator does *not*. Its 78% headline is a plane-average
artefact: it holds only in the low-severity corner and inverts in the
high-Delta_S region where every real domain sits. This is the same wrong-sign
signature that falsified the paper's original construction, so LLM-style scores
behave no differently from the four domains already tested.

## What this does and does not establish

It establishes that the mechanism's **structural precondition** — correlated
latent severity rather than independent hardness-and-magnitude — is coherent for
LLM-style scores, and that the diagnostic can tell the two apart in this setting.

It does **not** establish that real LLM scores have correlated severity. That is
an empirical question about real data, and it is exactly what Item 2 exists to
test. A synthetic generator built to have the property will display it; the
result here is only that the precondition is not ruled out a priori, which is
the cheap disqualifying step this item was scoped to provide.
