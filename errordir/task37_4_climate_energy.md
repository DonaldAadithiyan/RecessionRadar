# Task 37, Item 4 — HOS on Climate/ridge/β and Energy/ridge/β

## Result: **BOTH VALIDATE.** Climate percentile 1.000, energy 0.990. β's held-out correlation with realized error is 2.7–3.0× what a permutation-fitted direction achieves on the same holdout.

Data: `task37_4_real.csv`

| domain | model | n_fit | n_holdout | **HOS(β)** | null mean | **percentile** | validated |
|---|---|---|---|---|---|---|---|
| **climate** | ridge | 597 | 344 | **0.3583** | 0.1175 | **1.000** | **Yes** |
| **energy** | ridge | 1734 | 1000 | **0.4023** | 0.1499 | **0.990** | **Yes** |

Climate's β beats **all 200** permutation-fitted directions; energy's beats 198
of 200.

## What this establishes

These are the two fits behind this project's positive claim — the ones that had
never had a working instrument applied to their mechanistic validity. Task 36
left the question explicitly open ("neither supported nor undermined"; climate
"untested, not exonerated").

**It is now answered, affirmatively.** β carries genuine, held-out,
difficulty-specific information beyond what an identically-fitted direction
achieves by chance, on both domains.

The null is the reason this is meaningful. Permutation-fitted directions use the
same estimator, same alphas, same features, same normalization, and the same
error values — merely shuffled. So the comparison isolates the one thing that
differs: whether the target carried a real feature↔error relationship. β's
separation from that null is the evidence.

## Placing this alongside the other instruments

| instrument | ridge verdict | status |
|---|---|---|
| **Check 2** (magnitude perturbation) | **passes** 0.995 both | Valid — Item 1 confirmed fitted-but-meaningless directions score 0.18–0.48 and never pass |
| **DS** (magnitude-matched specificity) | uninformative | **Void on linear models** — δᵢ constant (Task 36) |
| **HOS** (held-out specificity) | **passes** 1.000 / 0.990 | Verified both directions on synthetic ground truth |

Two independent, non-degenerate instruments now support the mechanistic claim on
ridge, by different routes: Check 2 says perturbing along β moves the prediction
more than fitted-but-irrelevant directions do; HOS says β *orders test points by
difficulty* better than such directions do. Neither alone would be decisive;
together they are reasonably strong.

## Honest qualifications

1. **HOS is correlational.** It shows β's ordering tracks error better than
   chance. It does not show that β is *causally* adjacent to difficulty — that
   framing is what degenerated on linear models, and HOS does not restore it.
2. **The known-null synthetic landed at 0.825**, not near 0.5. The instrument has
   less headroom than ideal. Both real results (1.000, 0.990, with HOS at
   2.7–3.0× the null mean) sit well clear of that, but a marginal future result
   near 0.95 should be treated cautiously.
3. **n=1 per domain for β.** β is deterministic given the fit, so each domain
   contributes one candidate against 200 null draws. The percentile is a valid
   permutation p-value, but there is no across-fit replication within a domain.
4. **This says nothing about tree models.** HOS was not run on trees here. The
   0/20 tree record from Tasks 32–34 stands untouched.
