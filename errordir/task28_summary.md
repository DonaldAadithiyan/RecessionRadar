# Task 28 — Geometry-Conditioned Calibration: Staged Implementation

## Did Items 1–2 support proceeding? **Split: Item 2 strongly yes, Item 1 no.** And the final answer to "does this beat current methods on coverage and width": **no — on climate it wins the combined tradeoff but ties RLCP and loses both individual axes; on energy it fails outright at 68.50% coverage.**

## The staging worked exactly as designed

The two cheap checks were decisive, and running them first changed what Item 3
was for.

**Item 1 (literature): the mechanism is occupied territory.** RLCP (Hore &
Barber, arXiv:2310.07850) already conditions conformal width on kernel distance
in a **learned latent embedding space** — the same structural idea — and carries
guarantees this project's method does not (marginal validity, relaxed local
coverage, validity under covariate shift). Localized quantile regression
(arXiv:2412.01098) is close to Item 3's `q_α(z)` construction. The honest framing
is not "a new conditional conformal method."

**Item 2 (scarcity sweep): the efficiency claim holds, and strongly.**
errordir at **n_cal = 50** beats every baseline at **full** calibration (551
climate / 1600 energy) on both domains — an 11× and 32× data-efficiency
advantage.

| domain | errordir @ 50 | dtaci @ full | mondrian @ full | cqr @ full |
|---|---|---|---|---|
| climate | **11.331** | 12.271 | 12.145 | 13.351 |
| energy | **29.566** | 29.702 | 29.960 | 29.810 |

With the honest mechanism stated: errordir degrades at **the same rate** as DtACI
(+1.0% vs +0.5% on climate), so this is *not* superior scarcity-robustness. The
advantage is starting at a better frontier position and holding it. The correct
claim is "the frontier advantage persists under scarcity," not "needs less data
to work."

## Item-by-item

| Item | Status | Finding |
|---|---|---|
| 1 — Literature | **Occupied territory** | RLCP does learned-representation localized conformal, with better theory. |
| 2 — Scarcity sweep | **Claim holds** | n_cal=50 beats all baselines at full CAL, both domains. |
| 3 — `q_α(z)` | **Built, narrowed purpose** | Isotonic window quantile; K = max(30, n_cal/10) fixed in advance, never swept against test. |
| 4 — Non-circular validation | **Passes climate, marginal energy** | Fit on CAL-A, checked on disjoint CAL-B. Flagged energy bin 1 at 81.88% **before** Item 5. |
| 5 — Full comparison | **Climate 1st (ties RLCP); energy 10th of 12** | Beats 9 of 10 baselines significantly on climate; fails on energy. |

## The central result

On **climate**, `q_α(z)` genuinely improves on the fixed multiplier — Winkler
11.220 → **10.770**, width 9.441 → **8.666** (−8.2%), beating nine of ten
baselines significantly with win rates all above 50%.

But it **ties RLCP** (−0.029, q=0.596) — the prior-work method Item 1 identified.
After building the redesign, the result is a statistical dead heat with a
published method that already has stronger guarantees.

And it did **not** do what the task set out to do. The coverage gap against
diversity_optimal **widened** (3.29 → 5.35 pp): `q_α(z)` traded coverage for
width, moving *along* the frontier rather than past it. Task 27 concluded the
frontier position looks structural; this task's redesign is further evidence for
that, not against it.

On **energy** it fails outright — 68.50% coverage, 21.5 points below nominal,
worse than the fixed multiplier it replaced. The mechanism is identified: the
fixed multiplier is anchored to the **live ACI quantile**, which adapts online;
`q_α(z)` replaces that anchor with an estimate frozen on calibration data and
loses the adaptivity. Energy's test period is a different price regime, so the
frozen estimate systematically under-provisions — the same failure Task 27
documented for CQR on this exact domain.

## What earned its place

**Item 4's non-circular design.** It flagged energy bin 1 at 81.88% *before*
Item 5 ran, giving the energy collapse a diagnosis rather than just a bad number.
A circular check reusing CAL-A would have shown a well-fit function and taught
nothing.

**Item 1 before Item 3.** Building `q_α(z)` first and discovering RLCP afterward
would have spent the expensive part of this task on a question the cheap part
answered — precisely the waste the staging existed to prevent.

## Standing claim, revised

> The error-direction method reaches the best coverage-width frontier position
> among twelve methods on climate, and **retains that position with 11–32× less
> calibration data than baselines require** — the strongest and most novel result
> in this line of work. It does not dominate any individual axis, ties RLCP once
> that prior-work method is included, and its learned-quantile variant fails
> under distribution shift where the adaptive fixed-multiplier version does not.

The data-efficiency result (Item 2) is the finding worth building on. The
`q_α(z)` redesign is not.

## Files

- `task28_1_literature.md` — search, closest prior work, framing consequences
- `task28_2_scarcity_sweep.py` / `.md`, `task28_2_scarcity_sweep.csv`
- `task28_3_qalpha.py` / `.md` — estimator, K rationale, correctness check
- `task28_4_validation.md`, `task28_4_bins.csv` — held-out per-bin coverage
- `task28_45.py`, `task28_5_comparison.py` results:
  `task28_5_comparison.csv` / `.md`, `task28_5_significance.csv`
