# Task 18 Item 6 — Parameter-Free Online Conformal Baseline

**Verdict: it undercovers badly — 77.97–81.54% against a 90% target, losing to
even the plain trailing baseline at all four horizons. Being parameter-free
removes a tuning burden but costs coverage on this data. Attempted despite being
the spec's lowest-priority item, since the others finished.**

Script: `fix-reg/task18_6_upocp.py`. Data: `task18_6_upocp.csv`.

---

## What was implemented

Standard ACI needs a step size γ set in advance; DtACI removes it but introduces
η and σ instead. A parameter-free method derives its step size from the observed
gradient history, so nothing is user-set:

```
α_{t+1} = α_t + η_t·(α_target − err_t)
η_t = D / sqrt(1 + Σ_{s≤t} g_s²)          AdaGrad-style, D = 1
```

There is genuinely nothing to tune — η_t is determined entirely by the data seen
so far.

## Result

| Horizon | Trailing | **UP-OCP** | Selector | UP-OCP width | η (first → last) |
|---|---|---|---|---|---|
| Current | 93.85 | **81.54** | 93.85 | 38.69 | 0.743 → 0.298 |
| 1M | 89.06 | **81.25** | 96.88 | 41.98 | 0.743 → 0.298 |
| 3M | 90.32 | **80.65** | 95.16 | 49.98 | 0.995 → 0.298 |
| 6M | 84.75 | **77.97** | 96.61 | 88.42 | 0.743 → 0.289 |

**Beats the trailing baseline at 0 of 4 horizons.** It closes a *negative*
fraction of the selector's gain — it moves in the wrong direction.

## Why it fails here

The adaptive step starts large (η ≈ 0.74–1.0) and decays only to ≈ 0.30 over 65
test months. Against ACI's fixed γ = 0.005, that is **60× larger**, so α
oscillates violently in response to individual misses instead of tracking a
stable level. On a 65-point test stream the gradient history never accumulates
enough for η to decay into a useful range.

This is a small-sample failure of the parameter-free approach, not evidence
against parameter-free methods generally. AdaGrad-style scaling is designed for
long horizons where Σg² grows large; a 65-month test window is far too short.

## What the paper should say

One row in the baseline table plus one sentence: *a parameter-free online
conformal variant with an AdaGrad-scaled step was also tested and undercovers
substantially (77.97–81.54%), because its adaptive step size does not decay
sufficiently over a 65-month test window.* That answers the reviewer's question
directly and honestly.

**It should not be presented as a strong refutation of parameter-free methods.**
The failure mode is transparently about test-stream length, and a fair
evaluation would need a much longer stream than this testbed provides.

## Honest caveats

- **One parameterisation of "parameter-free".** UP-OCP proper and other
  parameter-free online-learning schemes differ in their step-size derivation;
  this implements the standard AdaGrad-style form rather than a specific
  published algorithm, and is labelled "UP-OCP style" for that reason.
- **D = 1 was set from the domain of α** (a probability), which is the natural
  choice and not a tuned value — but it does scale η directly, so a smaller D
  would produce a gentler and possibly better-behaved trajectory.
- **65 test months is short** for any method whose step size depends on
  accumulated gradient history. This result should not transfer to settings with
  thousands of test points.
- **Recession testbed only**, per the spec.
