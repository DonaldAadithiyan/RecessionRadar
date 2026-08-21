# Task 28, Item 2 — The Data-Scarcity Sweep

## Result: **the scarcity claim holds, and strongly. errordir at n_cal = 50 beats every baseline at FULL calibration, on both domains.**

Script: `task28_2_scarcity_sweep.py` · Data: `task28_2_scarcity_sweep.csv`

Uses the already-validated 1-D β from climate/ridge and energy/ridge — no new
β fitting. Only the calibration set size changes. 20 random subsamples per size
(1 for full); mean and sd reported.

## Winkler mean by calibration size (lower is better)

### Climate (full CAL = 551)

| n_cal | errordir | mondrian | dtaci | cqr |
|---|---|---|---|---|
| 50 | **11.331** | 12.182 | 12.329 | 13.865 |
| 100 | **11.273** | 12.252 | 12.305 | 13.628 |
| 200 | **11.224** | 12.346 | 12.252 | 13.381 |
| 400 | **11.221** | 12.304 | 12.248 | 13.381 |
| full | **11.220** | 12.145 | 12.271 | 13.351 |

### Energy (full CAL = 1600)

| n_cal | errordir | cqr | dtaci | mondrian |
|---|---|---|---|---|
| 50 | 29.566 | **29.205** | 30.420 | 31.025 |
| 100 | **29.731** | 29.829 | 30.618 | 31.277 |
| 200 | **29.212** | 29.849 | 30.175 | 30.380 |
| 400 | **28.962** | 29.771 | 29.857 | 30.213 |
| full | **28.775** | 29.810 | 29.702 | 29.960 |

## The headline comparison

**errordir at n_cal = 50 versus each baseline at FULL calibration:**

| domain | errordir @ 50 | dtaci @ full | mondrian @ full | cqr @ full | verdict |
|---|---|---|---|---|---|
| climate | **11.331** | 12.271 | 12.145 | 13.351 | **beats all three** |
| energy | **29.566** | 29.702 | 29.960 | 29.810 | **beats all three** |

errordir with **50** calibration points beats every baseline given **551**
(climate) or **1600** (energy) — an **11×** and **32×** data-efficiency
advantage respectively.

## The honest mechanism — this is not superior scarcity-robustness

The obvious reading is "errordir degrades more gracefully." That is **not** what
the data shows. Degradation from full → n_cal=50:

| domain | errordir | dtaci | mondrian | cqr |
|---|---|---|---|---|
| climate | +1.0% | **+0.5%** | **+0.3%** | +3.8% |
| energy | +2.7% | +2.4% | +3.6% | −2.0% |

errordir degrades at roughly the **same rate** as DtACI, and slightly *worse*
than Mondrian on climate. The efficiency advantage comes from **starting at a
better frontier position and holding it**, not from being more robust to
shrinking calibration data.

That is still a real and useful property — it means the advantage is not an
artifact of large calibration sets, and it survives into the small-n regime that
has been this project's persistent bottleneck (recession at n=59 never reached
significance). But the correct claim is "the frontier advantage persists under
scarcity," not "this method needs less data than others to work."

## Two caveats

**Energy at n_cal=50 is the one exception.** CQR edges errordir there
(29.205 vs 29.566) — the only cell in either domain where errordir is not first.
CQR also *improves* as calibration shrinks on energy (−2.0%), which is not a real
efficiency property: Task 27 established CQR badly under-covers on energy
(56.0%) because the test period is a different price regime, and a smaller
calibration set makes its already-broken intervals slightly narrower in a way
Winkler happens to reward. Coverage tells the true story — CQR sits at 55.8–58.1%
across all sizes while errordir holds 83.5–86.3%.

**Coverage is stable for errordir at every size**: climate 90.99–91.56%, energy
83.45–86.25%. The method does not achieve its Winkler position by sacrificing
coverage as data shrinks.

## Gate decision

Item 2 supports proceeding. Item 1 does not, unambiguously — the mechanism is
occupied territory (RLCP, localized quantile regression), which lowers Item 3's
expected value. Combined: **proceed to Item 3, but with its purpose narrowed** to
the single empirical question of whether `q_α(z)` closes the coverage gap the
fixed multiplier could not — not as a novel-method contribution.
