# Task 15 — Healthcare Power and Headroom

**Verdict: healthcare is LOW-HEADROOM, not underpowered — and the separations
that appear at larger n do not survive the noise guard. Study B (a second
dataset) is NOT warranted. The paper should state that healthcare is
constitutionally unable to rank calibration strategies, rather than continuing
to present it as an underpowered version of climate.**

All decision criteria were fixed in `task15_healthcare_power_spec.md` **before**
this ran. The result below is awkward for the obvious narrative, and the
pre-registration is the reason it is being reported the way it is.

Script: `fix-reg/task15_healthcare_power.py`.
Data: `task15_healthcare_threshold_sweep.csv`, `task15_headroom_targets.csv`.

---

## Study A — the threshold sweep

Lowering `min_encounters` yields more cohorts and a longer test stream, at the
cost of noisier per-cohort rates.

| min_enc | n_test | noise/signal | Baseline | Selector | Separates? | Aug | Separates? |
|---|---|---|---|---|---|---|---|
| 25 | 146 | 0.88 | 88.36 / 90.41 | 92.47 / 93.84 | no | 92.47 / 94.52 | no |
| 20 | 174 | 0.89 | 90.80 / 90.80 | 94.83 / 94.25 | no | 94.25 / 93.68 | no |
| **15** | **214** | **0.96** | 89.25 / 89.72 | 94.39 / 94.86 | **yes (both)** | 92.99 / 92.52 | no |
| 12 | 250 | 1.01 | 92.80 / 91.60 | 96.80 / 94.80 | yes (ridge) | 95.60 / 93.60 | no |
| 10 | 286 | 1.03 | 91.96 / 89.86 | 96.50 / 95.10 | yes (both) | 93.36 / 93.01 | no |

*(ridge / gradboost)*

### The finding that decides the task

**Separations by noise band:**

| Band | Cells | Separations |
|---|---|---|
| Evidence-grade (noise/signal < 0.95) | 4 | **0** |
| Borderline (0.95–1.0) | 2 | 2 |
| Artifact-prone (≥ 1.0) | 4 | 3 |

**Zero separations occur at any threshold where the target is clean. Every
separation appears only once the cohort rates are at or past the noise floor.**

That is precisely the pattern the spec pre-registered as artifact-prone. Had the
criteria been chosen after seeing this table, it would have been very tempting
to report "the selector separates at n=214" as a win.

### One nuance that distinguishes two failure modes

The spec's second warning sign — *the diversity gap growing as noise rises,
which would indicate noise inflating the diversity signal* — **does not occur**.
The gap ρ(support) − ρ(rare) moves the other way:

| Model | ρ(noise/signal, gap) |
|---|---|
| ridge | **−0.500** |
| gradboost | **−0.400** |

So noise is **diluting** the diversity signal, not manufacturing it. This is
reassuring for the paper's central diagnostic: the mechanism is not an artifact
of noisy targets, and the largest, cleanest gap is at the strictest threshold
(min=25: +0.196 / +0.292).

But it also means the separations at min≤12 are not evidence *for* the selector.
They arise because a noisier target inflates the score spread, which widens
intervals and lifts coverage mechanically — not because the selector is working
better. The distinction matters, and only the two-part guard catches it.

## Study C — headroom, with the A2 guard applied

Four candidate targets. Per the guard, each filters on a **pre-specified
covariate** (never on the outcome itself, which would let selection and score
share a term), and each is checked for collinearity before its headroom is read.

| Target | n | Mean rate | Baseline | ρ(rare, support) | Verdict |
|---|---|---|---|---|---|
| All cohorts (min=25) | 146 | 11.07% | 90.41 | **+0.509** | **DISCARDED — A2 degenerate** |
| Emergency admissions only | 60 | 11.57% | **88.33** | +0.127 | admissible |
| Long stays (≥5 days) | 63 | 12.51% | 93.65 | **+0.611** | **DISCARDED — A2 degenerate** |
| High prior utilisation | 60 | 16.06% | 93.33 | +0.478 | admissible |

**Two of four candidate targets failed the A2 guard** and were discarded rather
than reported — including, notably, the paper's own default construction at
ρ = 0.509, marginally over the 0.5 line.

That last point deserves attention rather than burial: the published healthcare
domain sits just past the collinearity threshold this task pre-registered. It is
not degenerate in the way the original binary target was (ρ = 0.68–0.72), but it
is closer to that boundary than is comfortable, and the paper should say so.

**No admissible target pushes the baseline below 88%.** The lowest achievable is
88.33% (emergency admissions), and that is with n=60 — too small to test
anything. Pre-registered conclusion: **low-headroom = True**.

## Verdict against the committed decision table

| Test | Pre-registered criterion | Result |
|---|---|---|
| Underpowered | +3.4pp effect fails to separate at min=15 | **False** — it does separate (94.86, Wilson [91.03, 97.11] vs baseline 89.72) |
| Low-headroom | no admissible target pushes baseline < 88% | **True** |

Per the spec's committed table: **not underpowered → Study A resolved it → no
Study B.** A second healthcare dataset is not warranted.

**But the honest reading is more careful than "not underpowered".** The min=15
separation that triggers that verdict sits at noise/signal = 0.96 — *above* the
0.95 evidence-grade line the same spec set. The two criteria conflict, and the
conflict is itself the result:

> Healthcare can be made to separate, but only by degrading the target to the
> point where the separation is no longer trustworthy. It is not that more data
> is needed — it is that this domain's baseline sits at 88–91% under every
> admissible target, leaving ~10pp of headroom, most of which is overcoverage.

That is a **low-headroom domain**, and it is a legitimate finding rather than a
gap. It also explains, retroactively, why healthcare has separated nothing
across Tasks 7, 8 and 11: there was never enough room for a strategy to
distinguish itself.

## What the paper should say

1. **Reframe healthcare.** Not "underpowered, cannot rank methods" but
   "low-headroom: the baseline is already near nominal under every admissible
   target, so no calibration strategy has room to distinguish itself." This is
   a property of the domain, not a limitation of the study.
2. **Keep min_encounters = 25.** It is the cleanest threshold (noise/signal
   0.88) and produces the largest, most trustworthy diversity gap. The published
   results do not change.
3. **Record the ρ = 0.509 finding.** The default healthcare construction sits
   marginally past the collinearity line this task pre-registered. Worth one
   honest sentence.
4. **Record the pooling rejection** (spec §2, Study B) verbatim: between-dataset
   variance would appear in the diagnostic *as diversity*, manufacturing the
   paper's headline effect rather than merely biasing it conservatively. If a
   second cohort is ever added it must be a separate domain row, never a merge.

## Honest caveats

- **The two pre-registered criteria conflict at min=15**, and the write-up
  resolves that conflict by weighting the noise guard over the separation test.
  That is a judgment, made visible rather than hidden — a reader who weights
  them the other way would conclude "not underpowered, full stop."
- **Study C's admissible targets have n = 60–63**, far too small to test
  anything beyond the baseline level itself. The headroom conclusion rests on
  where the baseline *sits*, not on any coverage comparison at those n.
- **Only four candidate targets were tried.** A more aggressive target might
  find headroom, but the two that moved the baseline most were both A2-degenerate
  — which suggests headroom and degeneracy are hard to separate in this dataset.
- **The noise/signal ratio uses a worst-case binomial SE** at the minimum cohort
  size; cohorts above the threshold are less noisy, so the ratio is conservative.
  A per-cohort weighting would place the boundary somewhat lower.
