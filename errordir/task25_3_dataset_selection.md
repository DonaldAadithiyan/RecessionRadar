# Task 25, Item 3 — Dataset Selection

**Criteria were fixed before any dataset was evaluated against the method.**
Search process, candidates considered, and rejections are reported in full.

## The four criteria (committed in advance, in priority order)

1. Publicly available, no cost or access barrier.
2. Genuine rare/severe-event structure (not uniform error severity).
3. A real named practical domain where XGBoost/LightGBM are standard.
4. Materially more usable rows than recession's n=59 at 6M.

## Candidates evaluated

| # | Candidate | Source | Rows | Tail structure | Verdict |
|---|---|---|---|---|---|
| 1 | **French Motor TPL claim severity** (`freMTPL2sev` + `freMTPL2freq`) | OpenML 41215 / 41214 | **26,639** claims joined to 678,013 policies | p99/p50 = **14.3**, max 4,075,400 vs median 1,172 | **SELECTED** |
| 2 | **Australian electricity market price** (`electricity`) | OpenML 151 | **45,312** half-hourly | p99/p50 = 3.3; **0.37%** above 5× median | **SELECTED** |
| 3 | Bike-sharing hourly (`electricity-hourly`, id 44063) | OpenML 44063 | 17,379 | Demand is diurnal/seasonal, **not rare-event** | **REJECTED — criterion 2** |
| 4 | UCI credit default | UCI | ~30,000 | Binary default label, not a continuous severity target; would need a synthetic severity construction | **REJECTED — criterion 2** (would manufacture the structure the task exists to test) |

Candidates 3 and 4 were rejected on criterion 2 specifically. Bike-sharing has
smooth, predictable seasonality — an evenly-distributed-severity problem, which
the task explicitly warns is "testing a different, broader, less-motivated
question." Credit default is binary; deriving a continuous severity would mean
constructing the rare-event structure rather than finding it, and this project
has repeatedly declined to manufacture signals (Task 19 Item 2, Task 24 Item 1).

## Selected datasets

### D1 — Insurance claim severity (freMTPL2)
- **Target:** `ClaimAmount` (log-transformed; severity is heavy-tailed by nature)
- **Features:** 10 real policy attributes joined on `IDpol` — Exposure, Area,
  VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Density, Region
- **Rare-event structure:** p99/p50 = 14.3; the largest claim is 3,477× the median
- **Practical fit:** claim-severity modelling is a canonical GBM application in
  actuarial practice
- **Power:** 26,639 claims — ~450× recession's n=59

### D2 — Energy price spikes (electricity)
- **Target:** `nswprice` (NSW electricity spot price)
- **Features:** day, period, nswdemand, vicprice, vicdemand, transfer + lagged
  price/demand terms constructed with strict `shift(1)` discipline
- **Rare-event structure:** 0.37% of half-hours exceed 5× the median price;
  p99/p50 = 3.3
- **Practical fit:** price-spike forecasting is a standard GBM task in energy trading
- **Power:** 45,312 half-hourly observations

## Note on the weaker tail in D2

D2's tail ratio (3.3) is materially thinner than D1's (14.3). This is recorded
now, before any result, because it sets an expectation in the same way Task 24's
healthcare pre-registration did: D2 has less severity dispersion for a difficulty
signal to exploit, so a weaker effect there would be unsurprising. It still
clears criterion 2 on the spike-frequency measure (0.37% above 5× median is
genuine rare-event structure), so it is retained rather than swapped.

**Per the Item 3 guardrail, both selected datasets are tested and reported
regardless of outcome. No third replacement candidate will be substituted.**
