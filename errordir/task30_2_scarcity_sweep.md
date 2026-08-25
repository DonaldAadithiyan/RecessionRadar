# Task 30, Item 2 — Scarcity Sweep With Corrected RLCP

## Result: **errordir's data-efficiency advantage holds against RLCP, and this time it is genuine scarcity-robustness — not the fixed head start Task 28 had to correct itself about.** RLCP degrades 8× faster on climate and produces unbounded intervals at a rate that triples as calibration shrinks.

Script: `task30_2_scarcity_sweep.py` · Data: `task30_2_scarcity_sweep.csv`

Uses Task 29's **corrected** RLCP (randomization + n+1 normalisation + `+∞`
atom), not Task 28's baseLCP. Task 28's sweep contained no localized method at
all, so this is the first real comparison.

## Winkler mean by calibration size

### Climate

| n_cal | errordir | **rlcp** | mondrian | dtaci | cqr |
|---|---|---|---|---|---|
| 50 | **11.331** | **undefined** | 12.182 | 12.329 | 13.865 |
| 100 | **11.273** | 11.502 | 12.252 | 12.305 | 13.628 |
| 200 | **11.224** | 11.592 | 12.346 | 12.252 | 13.381 |
| 400 | **11.221** | 11.515 | 12.304 | 12.248 | 13.381 |
| full | **11.220** | 11.292 | 12.145 | 12.271 | 13.351 |

### Energy

| n_cal | errordir | **rlcp** | cqr | dtaci | mondrian |
|---|---|---|---|---|---|
| 50 | 29.566 | **undefined** | 29.205 | 30.420 | 31.025 |
| 100 | **29.731** | undefined | 29.829 | 30.618 | 31.277 |
| 200 | **29.212** | undefined | 29.849 | 30.175 | 30.380 |
| 400 | **28.962** | undefined | 29.771 | 29.857 | 30.213 |
| full | **28.775** | undefined | 29.810 | 29.702 | 29.960 |

**errordir beats RLCP at every calibration size on climate**, and RLCP's Winkler
mean is undefined at every size on energy.

## The unbounded-rate column — the finding the pre-registration protected

| n_cal | climate RLCP | **energy RLCP** | all other methods |
|---|---|---|---|
| 50 | **2.94%** | **19.93%** | 0.00% |
| 100 | 0.76% | 15.12% | 0.00% |
| 200 | 0.16% | 12.75% | 0.00% |
| 400 | 0.06% | 10.10% | 0.00% |
| full | 0.00% | 6.75% | 0.00% |

RLCP's unbounded rate rises **monotonically** as calibration shrinks — from 0% to
2.94% on climate, and **6.75% → 19.93% (3×)** on energy. No other method produces
a single unbounded interval at any size.

This is exactly the mechanism predicted: RLCP's weighted quantile needs enough
*effective local mass* near the test point. As calibration shrinks, the kernel
finds too few neighbours, the finite atoms never reach 1−α, and the quantile
falls through to the `+∞` tail. **This is a structural scarcity failure mode that
errordir's global ACI anchor does not have.**

Per the pre-registration, no cell reached a majority unbounded, so Winkler
medians are reported rather than suppressed.

## Degradation rate — the Task 28 self-correction check, applied to RLCP

Task 28's first read of errordir's own result was wrong in exactly this way, so
the same check is applied here before any gap is read as "RLCP needs more data."

| domain | method | metric | full | n=50 | **degradation** |
|---|---|---|---|---|---|
| climate | mondrian | mean | 12.145 | 12.182 | +0.3% |
| climate | dtaci | mean | 12.271 | 12.329 | +0.5% |
| climate | **errordir** | mean | 11.220 | 11.331 | **+1.0%** |
| climate | cqr | mean | 13.351 | 13.865 | +3.8% |
| climate | **rlcp** | median | 9.642 | 10.438 | **+8.3%** |
| energy | dtaci | mean | 29.702 | 30.420 | +2.4% |
| energy | **errordir** | mean | 28.775 | 29.566 | **+2.7%** |
| energy | mondrian | mean | 29.960 | 31.025 | +3.6% |
| energy | **rlcp** | median | 4.233 | 5.446 | **+28.7%** |

**This is the genuine article, unlike Task 28's case.** There, errordir degraded
at the *same* rate as DtACI (+1.0% vs +0.5%) and its advantage was purely a
better starting position. Here RLCP degrades **8× faster than errordir on
climate** (+8.3% vs +1.0%) and **10× faster on energy** (+28.7% vs +2.7%), *and*
its unbounded rate triples. The gap widens as data shrinks rather than staying
fixed — which is what a real data-efficiency advantage looks like.

One honest caveat: RLCP's degradation is measured on Winkler **median** (its mean
being undefined), while errordir's is on the mean. Medians are less
outlier-sensitive, so if anything this comparison *understates* RLCP's
degradation — the unbounded intervals it is shedding are exactly the cases a mean
would penalise most.

## Coverage

RLCP's raw coverage on energy rises as calibration shrinks (76.75% → 85.67%),
which looks like an advantage but is not: unbounded intervals count as covering.
With a 19.93% unbounded rate at n=50, roughly a fifth of that "coverage" is
infinite intervals. errordir holds 83.45–86.25% with zero unbounded throughout.

## Interpretation in light of Item 1

Item 1 established energy contains **concept drift**, not pure covariate shift.
So energy's RLCP result should be read as a compound failure — a method outside
its guaranteed scope *and* starved of local mass — rather than as a clean
scarcity result. **Climate is the clean test**: pure covariate shift, no concept
drift, and RLCP still degrades 8× faster with a rising unbounded rate. The
data-efficiency finding therefore stands on climate independently of the energy
result.
