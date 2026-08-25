# Task 36, Item 1 — Is δᵢ Constant on the Four Ridge Fits?

## Result: **YES — exactly constant. All four ridge fits produce ONE unique δ value across 400 test points (CV ≈ 2×10⁻¹⁵, floating-point zero). DS is structurally void on these fits. Item 2 does not run.**

Script: `task36_1_degeneracy.py` · Data: `task36_1_degeneracy.csv`

## Direct measurement

| domain | model | class | mechanism | mean δ | **CV** | **unique δ values** | degenerate |
|---|---|---|---|---|---|---|---|
| insurance | ridge | linear | beta | 0.1209 | **3.4e−15** | **1 / 400** | **Yes** |
| insurance | ridge | linear | A_local_knn | 0.0990 | **3.2e−15** | **1 / 400** | **Yes** |
| energy | ridge | linear | beta | 2.3301 | **6.6e−16** | **1 / 400** | **Yes** |
| energy | ridge | linear | A_local_knn | 2.5967 | **7.8e−16** | **1 / 400** | **Yes** |
| insurance | lightgbm | tree | beta | 0.2284 | 0.802 | 381 / 400 | No |
| insurance | lightgbm | tree | A_local_knn | 0.2244 | 0.824 | 381 / 400 | No |
| energy | lightgbm | tree | beta | 0.9722 | 0.607 | 398 / 400 | No |
| energy | lightgbm | tree | A_local_knn | 1.4753 | 0.493 | 400 / 400 | No |

The tree rows are the positive control and they behave exactly as they must —
CV 0.49–0.82, 381–400 distinct values. **The measurement instrument works.** The
ridge rows are not "small variation," they are *one number repeated 400 times*.

## The pipeline trace — why this is exact, not approximate

Task 34's prediction path for ridge is:

```python
def pred(Zs):
    raw = scb.inverse_transform(Zs)
    return mfull.predict(sc0.transform(raw))     # mfull = Ridge(alpha=1.0)
```

Three affine maps composed: `scb.inverse_transform` (affine), `sc0.transform`
(affine), `Ridge.predict` (affine). **No clipping, no flooring, no bound, no
post-hoc nonlinearity anywhere** — verified by grep over the script; the only
`Ridge(` occurrences are bare constructors with no wrapping.

So the whole path is a single affine function `f(z) = a·z + c`, and

    δᵢ = |f(xᵢ + hv) − f(xᵢ − hv)| = |2h·(a·v)|

which has **no xᵢ dependence at all**. The constant is not an empirical finding
about these datasets; it is forced by the composition. The measured CV of
~10⁻¹⁵ is floating-point noise around an exact identity.

This is what the concern predicted, and it is confirmed as a fact about the code
rather than an inference about OLS in the abstract.

## What follows immediately

DS is `Spearman(δᵢ, eᵢ)`. With δᵢ constant, the ranks of δ are all tied, the
Spearman denominator degenerates, and the returned value is determined entirely
by how `scipy` breaks ties on a constant vector — it is **not a measurement of
anything about the fit**.

The same applies to every magnitude-matched null direction, which are also
constant-δ. So both the statistic and its null are void, and the reported
percentile is an artifact of tie-breaking noise.

**Task 34's four ridge rows measured nothing.** They cannot support the claim
Task 34 drew from them.

## Item 2 is not run

Per the guardrail: *"Do not run this item if Item 1 finds degeneracy — testing
climate under a broken instrument would just produce a second uninformative
number."* Climate/ridge/β would have δᵢ constant for exactly the same structural
reason. It is left untested rather than given a meaningless number.

**Consequence for the escalation question:** the urgent branch dissolves. Task
34's ridge rows are an instrument artifact, not evidence against the mechanistic
claim. The mechanistic claim for β on ridge is **neither supported nor undermined
by DS** — DS simply cannot speak to it.
