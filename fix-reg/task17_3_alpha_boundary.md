# Task 17 Item 3 — Sweeping the α Boundary Directly

**Verdict: the proof is verified in substance, but its boundary is stated one
character too loosely. The ceiling ratio is exactly 1.0 for every α < 0.5 and
breaks AT α = 0.5, not after it. Section 4.3 should read q > 0.5 (strict), not
q ≥ 0.5. This is a real correction to a claim currently in the paper.**

The spec pre-registered that disagreement inside α ≤ 0.5 is a critical finding
to report immediately rather than smooth over. There is one, it is narrow, and
it is reported here.

Script: `fix-reg/task17_135_boundary_bootstrap_qc.py`.
Data: `task17_3_alpha_boundary.csv`.

---

## Result

Ceiling ratio (selector's operative quantile ÷ the theoretical maximum for any
N-subset), recession testbed, N = 254:

| α | 1M | 3M | 6M | Current |
|---|---|---|---|---|
| 0.010 – 0.490 | **1.00000** | **1.00000** | **1.00000** | **1.00000** |
| **0.500** | **0.52375** | **0.51477** | **0.51844** | **0.51222** |
| 0.550 | 0.03178 | 0.02560 | 0.02184 | 0.03079 |
| 0.600 | 0.03680 | 0.03236 | 0.02645 | 0.03548 |
| 0.700 | 0.04152 | 0.02938 | 0.03279 | 0.03479 |
| 0.800 | 0.01135 | 0.01370 | 0.01750 | 0.01785 |

- **α ≤ 0.5: 40 of 44 configurations at the ceiling; 4 violations, all at
  exactly α = 0.500.**
- α > 0.5: all 16 below the ceiling, as predicted (min ratio 0.011).

## What the violation is — and it is not a bug

The oversupply argument says the alternating-tail rule supplies ~N/2 extreme
scores while the (1−α) quantile depends on the top α·N. At N = 254:

| α | Scores needed | Scores supplied | Ratio |
|---|---|---|---|
| 0.4900 | 124.5 | 127 | 1.00000 |
| 0.4950 | 125.7 | 127 | 1.00000 |
| **0.4999** | **127.0** | **127** | **0.54310** |
| **0.5000** | **127.0** | **127** | **0.51844** |
| 0.5001 | 127.0 | 127 | 0.49375 |
| 0.5050 | 128.3 | 127 | 0.02401 |

**The inequality becomes tight exactly at α = 0.5** — supply equals demand at
127 — and a tight inequality is not a strict one. The selector needs *strictly
more* extreme scores than the quantile consumes, and at α = 0.5 it has exactly
as many, which is not enough.

So the proof's mechanism is confirmed precisely as argued; only the boundary's
inclusivity is wrong. The correction is:

> **q > 0.5 (equivalently α < 0.5)**, not q ≥ 0.5.

## Why this matters, and why it does not

**It matters** because the paper states a boundary and the boundary is off by
the endpoint. A reader checking the algebra would find the same tightness, and
an uncorrected claim invites the objection.

**It does not matter practically.** ACI's measured operating range is
[0.073, 0.132] (Task 10), which is nowhere near 0.5, and every result in the
paper sits at α ≤ 0.25 (Task 16D). The optimality guarantee covers everything
the paper actually does, with an enormous margin. No published number changes.

## Honest caveats

- **The failure is exact-arithmetic, not statistical.** It reproduces
  identically across all four horizons because it follows from N and α alone,
  not from the data. That is why it is reported as a proof-statement correction
  rather than a measurement.
- **N = 254 throughout.** The tightness point is where α·N = N/2, i.e. α = 0.5
  for any even N; for odd N the floor in N//2 shifts it fractionally. Not swept
  over N here — Task 12A already established the ratio holds at every N from 20
  to 254 at α = 0.10.
- **α > 0.5 ratios are not smoothly decreasing** (0.032, 0.037, 0.042, 0.011).
  Beyond the boundary the selector's low-tail scores dominate the quantile and
  the ratio becomes essentially arbitrary; the values there are not meaningful
  beyond "well below 1".
- **Primary testbed only**, as specified. The argument is arithmetic rather than
  data-dependent, so it should transfer, but it was not run on healthcare or
  climate.
