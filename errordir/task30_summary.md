# Task 30 — Covariate-Shift Diagnostic, Then the RLCP Scarcity Sweep

## Item 1 first: **energy is covariate shift AND concept drift.** Claim 1 must be narrowed — RLCP was tested partly outside its guarantee's scope. Item 2 then: **errordir's data-efficiency advantage over RLCP is real, and unlike Task 28's case it is genuine scarcity-robustness rather than a fixed head start.**

## Item 1 — which kind of shift?

| test | energy | climate (control) |
|---|---|---|
| P(X) shift — classifier AUC | 0.799 (moderate) | **0.958 (strong)** |
| P(error&#124;z) drift — fraction of matched z-bins with KS p<0.01 | **0.40** | **0.00** |
| median error ratio TEST/FIT at matched z | 1.315 | 1.276 |
| did methods break? | **yes** (RLCP, CQR, q_α) | **no** (all tie) |

The control makes this conclusive: **climate has stronger covariate shift than
energy and nothing breaks; energy has weaker covariate shift but real concept
drift and three methods fail.** Energy's bin 5 shows median error rising **2.25×**
at matched difficulty (0.978 → 2.202, p<0.0001) — the feature→error relationship
itself moved.

### How claim 1 changes, stated before Item 2 was framed

**Was:** "RLCP, CQR and q_α all fail on energy under distribution shift" —
implying RLCP failed a guarantee it claims.

**Now:** RLCP's guarantee covers *covariate shift*. Energy contains concept
drift, which no covariate-shift guarantee addresses. **RLCP's energy failure is
not a violation of its stated guarantee.**

This narrows rather than retracts. What survives:

> Under **concept drift**, methods that freeze a calibration-period estimate
> (RLCP, CQR, q_α) degrade while errordir's online ACI anchor adapts. This is a
> gap in what current methods *offer*, not a failure to deliver what they
> promise.

Weaker as a criticism of RLCP; still a real scope statement. And claim 1 remains
**n=1** — energy is the only domain with detected concept drift.

## Item 2 — the sweep

**Winkler mean, climate:** errordir beats RLCP at every calibration size
(11.331 vs undefined at n=50; 11.273 vs 11.502 at n=100; 11.220 vs 11.292 at
full). On energy RLCP's mean is undefined at every size.

**The unbounded-rate column — where the real finding is:**

| n_cal | climate RLCP | energy RLCP | every other method |
|---|---|---|---|
| 50 | **2.94%** | **19.93%** | 0.00% |
| full | 0.00% | 6.75% | 0.00% |

RLCP's unbounded rate rises monotonically as calibration shrinks — **3× on
energy**. The kernel needs effective local mass near the test point; when
calibration thins, the finite atoms never reach 1−α and the quantile falls
through to `+∞`. errordir's global ACI anchor has no such failure mode.

**Degradation rate — the check Task 28 had to apply to itself:**

| method | climate | energy |
|---|---|---|
| mondrian | +0.3% | +3.6% |
| dtaci | +0.5% | +2.4% |
| **errordir** | **+1.0%** | **+2.7%** |
| cqr | +3.8% | −2.0% |
| **rlcp** | **+8.3%** | **+28.7%** |

**This is the genuine article.** In Task 28, errordir degraded at the *same* rate
as DtACI (+1.0% vs +0.5%) and its advantage was purely a better starting
position — I corrected that reading there. Here RLCP degrades **8× faster on
climate and 10× faster on energy**, *and* its unbounded rate triples. The gap
**widens** as data shrinks, which is what data efficiency actually looks like.

Caveat: RLCP's degradation is on Winkler median (its mean undefined), errordir's
on the mean. Medians are less outlier-sensitive, so this likely *understates*
RLCP's degradation.

**Climate is the clean test.** Item 1 showed it has pure covariate shift and no
concept drift, so RLCP is fully within its guaranteed scope there — and still
degrades 8× faster. The data-efficiency finding stands independently of energy's
compounded failure.

## Where the claim now stands

| claim | status |
|---|---|
| **Ties on accuracy** | Confirmed (Task 29): climate Winkler 11.220 vs 11.292, q=0.556. |
| **Wins on data efficiency** | **Confirmed against RLCP, on climate, with degradation-rate evidence.** |
| **Wins on adaptivity** | **Narrowed:** concept drift, not covariate shift; n=1 domain. |

The three-part claim is now two-thirds solid. Claim 1 is the weak leg — not
because the mechanism is unclear (three methods failing identically, with a
control isolating the cause, is strong mechanistic evidence) but because it rests
on a single domain. **Task 31's second regime-shifting domain is the right next
investment**, and Item 1 gives it a precise selection criterion it did not have
before: the domain must exhibit **concept drift** (drift in P(error|z) at matched
z), not merely covariate shift — testable with `task30_1_shift.py` before any
comparison is run.

## Files

- `task30_1_shift.py`, `task30_1_shift_diagnostic.md`, `task30_1_shift.csv`
- `task30_2_scarcity_sweep.py` / `.md`, `task30_2_scarcity_sweep.csv`
