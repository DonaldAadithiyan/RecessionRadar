# Task 35, Item 2 — Locality Gate

## Result: **PASSES on all six fits, ridge included — the bandwidth fix worked. But a serious caveat is recorded here before Item 3: the gradient field sits only 0.3–0.9 sd above what purely random directions would produce.**

Script: `task35_2_locality_check.py` · Data: `task35_2_locality_check.csv`

## The bandwidth fix, measured against Task 33's

| domain | model | rule | h | **eff. neighbours** | **mean \|cos\|** | frac >30° | LOCAL |
|---|---|---|---|---|---|---|---|
| insurance | ridge | **silverman** | **0.567** | **2.23** | **0.284** | **1.000** | **Yes** |
| insurance | ridge | median (T33) | 4.357 | 1221 | 0.833 | 0.093 | Yes* |
| insurance | xgboost | **silverman** | 0.567 | 2.23 | 0.284 | 0.997 | **Yes** |
| insurance | xgboost | median (T33) | 4.357 | 1221 | 0.862 | 0.080 | Yes* |
| insurance | lightgbm | **silverman** | 0.567 | 2.23 | 0.285 | 1.000 | **Yes** |
| energy | ridge | **silverman** | **0.588** | **4.34** | **0.292** | **1.000** | **Yes** |
| energy | ridge | median (T33) | 4.210 | 1042 | 0.886 | 0.058 | Yes* |
| energy | xgboost | **silverman** | 0.588 | 4.34 | 0.290 | 0.997 | **Yes** |
| energy | xgboost | median (T33) | 4.210 | 1042 | **0.902** | 0.043 | **No** |
| energy | lightgbm | median (T33) | 4.210 | 1042 | **0.902** | 0.043 | **No** |

\* marginal — 0.83–0.89 against a 0.90 bar

**The fix is unambiguous on its own terms.** Silverman gives h ≈ 0.57–0.59 versus
the median heuristic's 4.21–4.36. Mean |cosine| falls from **0.83–0.90 → 0.29**,
and the fraction of gradients more than 30° from the mean direction rises from
**4–9% → ~100%**. Task 33's near-constant field is gone.

**Ridge passes on both domains**, so per the Item 2 guardrail Item 3 may proceed.

## The caveat that must travel with this result

**Effective neighbours: 2.23 (insurance) and 4.34 (energy) out of 1500 FIT
points — 0.2%.** A Nadaraya-Watson mean over ~2–4 points is extremely noisy, and
its gradient more so.

That raises the mirror-image concern to Task 33's. Task 33's `h` was too large,
making D(x) global and its gradients near-constant. Silverman's `h` may be too
small, making D(x) approximately a nearest-neighbour lookup and its gradients
near-**random**. Both are ways of failing to be meaningfully local.

Checked directly against the random-direction baseline at the actual dimensions:

| domain | d | random-field mean \|cos\| | observed | excess |
|---|---|---|---|---|
| insurance | 12 | 0.235 (sd 0.167) | 0.284 | **+0.049 (0.3 sd)** |
| energy | 21 | 0.177 (sd 0.129) | 0.292 | **+0.115 (0.9 sd)** |

The field is **above** the random baseline on both domains, so it is not pure
noise — there is genuine structure. But the margin is small, and on insurance it
is well within one standard deviation.

**What this means for interpreting Item 3.** The locality criterion was fixed in
advance and is met. But "local" here means "gradients differ from each other,"
and a random field would also satisfy that. The honest reading is that this
gradient field has **real but weak** directional structure. A negative Item 3
result therefore cannot cleanly distinguish "locality doesn't help" from "this
particular local estimate is too noisy to help" — and that ambiguity should be
stated in the verdict rather than resolved by assertion.

I am **not** adjusting the bandwidth in response to this. Silverman was fixed in
Item 1 with the over-smoothing risk stated in advance; discovering an
under-smoothing risk after the fact and re-tuning would be exactly the test-set
adaptation this project has refused since Task 22.
