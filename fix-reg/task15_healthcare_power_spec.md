# Task 15 — Statistical Power for the Healthcare Domain

**Status: specification. The feasibility work was done before writing, and it
changed the recommended approach — read §1 before §2.**

## Why this task exists

Healthcare has separated **zero** strategies across every task in the paper.
Task 7 (8 strategies), Task 8 (4 underlying models incl. LACE/HOSPITAL), Task 11
(augmentation) — nothing has ever cleared the confidence interval in that
domain, while climate separates routinely. Every healthcare conclusion is
currently "confirms the mechanism, cannot rank methods."

The original proposal was to pool MIMIC-IV-derived cohorts with UCI Diabetes 130
to raise n. **The feasibility check below says that is the wrong instrument for
this problem, and would introduce a worse one.**

---

## §1 — What the diagnostic actually found (do this before considering MIMIC)

### The binding constraint is a design parameter, not a data shortage

Healthcare's test stream is n=146. That is not a property of the UCI dataset —
it is a consequence of the `min_encounters = 25` cohort threshold chosen in
`domain_healthcare.py`. Relaxing it yields more cohorts from the *same* data:

| min_encounters | Cohorts | Test stream (~25%) | Cohort-rate SD |
|---|---|---|---|
| 25 (current) | 584 | **146** | 7.19pp |
| 15 | 857 | **214** | 8.46pp |
| 10 | 1144 | **286** | 9.69pp |
| 5 | 1947 | **486** | 12.88pp |

### How much n is actually needed

At the effect sizes healthcare actually exhibits, separation requires far less
than a second dataset. Wilson lower bound of the treatment vs the baseline
coverage:

| Scenario (observed in Task 8/11) | n=146 | n=250 | n=286 | n=400 |
|---|---|---|---|---|
| base 88.4% → +4.1pp (ridge, augmented) | no | **yes** | **yes** | yes |
| base 90.4% → +3.4pp (gradboost, selector) | no | no | no | **yes** |
| base 90.9% → +7.8pp (climate-sized effect) | yes | yes | yes | yes |

**n ≈ 250–400 is sufficient**, and `min_encounters = 10–15` already delivers
214–286 from existing data. MIMIC-IV is not required to fix the power problem.

### But lowering the threshold has a real cost, and it is nearly fatal at min=10

Cohort readmission rates are binomial estimates. Their noise floor grows as the
threshold falls, and it must stay well below the signal it is meant to carry:

| min_encounters | Noise floor (binomial SE) | Signal (rate SD) | Noise/signal |
|---|---|---|---|
| 25 | 6.31pp | 7.19pp | 0.88 |
| **15** | **8.14pp** | **8.46pp** | **0.96** |
| 10 | 9.97pp | 9.69pp | **1.03 — noise exceeds signal** |
| 5 | 14.10pp | 12.88pp | 1.09 — worse |

**At min=10 the cohort rates are mostly noise.** Any "improved separation" found
there would likely be an artifact of a degraded target, which is the exact
failure mode Appendix A2 already documents for the binary-target construction.

**This is the central tension of the task**: more n is available, but only by
degrading the thing being measured. The spec below resolves it by treating the
threshold as an experimental variable rather than picking one.

---

## §2 — What to build

### Study A (primary) — threshold sweep with a noise-floor guard

Re-run the healthcare domain at `min_encounters ∈ {25, 20, 15, 12, 10}`, and at
each threshold report:

1. n, cohort count, and the noise/signal ratio above;
2. the diagnostic (ρ(support) vs ρ(rare-count), 200 draws);
3. the selector and augmentation vs the trailing baseline, with Wilson intervals;
4. **whether anything separates**, which is the question the domain has never
   been able to answer.

**Pre-registered interpretation rule, to prevent reading noise as signal:**

- Separation appearing at a threshold where **noise/signal < 0.95** is evidence.
- Separation appearing *only* at noise/signal ≥ 1.0 must be reported as **likely
  an artifact of target degradation**, not as a finding.
- If the diagnostic's ρ(support) − ρ(rare) gap *grows* as the threshold falls,
  treat that as a warning sign rather than a result: it is what would happen if
  noise were inflating the diversity signal.

The honest expected outcome is that min=15 (n=214, noise/signal 0.96) is the
best available compromise, and that it separates the larger effects but not the
+3.4pp ones. That would be a real improvement over "nothing ever separates."

### Pre-registered decision thresholds (committed BEFORE seeing any results)

The spec previously said Study B runs "if A and C leave the domain genuinely
underpowered" — a judgment call that would be made *after* seeing the sweep,
which is exactly the sequencing Task 13 avoided. The numbers are therefore fixed
here, in advance:

**"Underpowered"** — the domain has a detectable effect that n is too small to
resolve. Operationally:

> At `min_encounters = 15` (n≈214, the largest noise/signal-safe threshold),
> the **+3.4pp gradboost-selector effect** measured in Task 8
> (90.41 → 93.84) fails to separate: its Wilson lower bound does not exceed
> the baseline coverage.

**"Low-headroom"** — the domain has no room for an effect regardless of n:

> Under Study C's hardest target attempt, the pooled/trailing baseline remains
> **≥ 88%**. If the baseline cannot be pushed below 88% by any admissible target
> definition, additional n cannot help, because the maximum attainable gain is
> smaller than the effects the paper reports elsewhere.

**Decision rule:**

| Underpowered? | Low-headroom? | Conclusion | Study B? |
|---|---|---|---|
| yes | no | genuinely needs more data | **run B** (separate domain row) |
| yes | yes | both — but headroom binds first | **do not run B**; report as low-headroom |
| no | no | Study A fixed it | no |
| no | yes | separates but only into overcoverage | no; report the ceiling |

Study B runs in exactly one cell of that table. This is committed now so the
answer cannot be reverse-engineered from the sweep.

### Study B (secondary) — a genuinely independent healthcare cohort

Only in the single case identified above. **Do not pool with UCI.**

**Pooling is rejected, and the reason should be recorded in the paper.** UCI
Diabetes 130 defines readmission as `<30 days` over 1999–2008 across 130 US
hospitals, restricted to diabetic encounters. Any MIMIC-derived cohort would
differ in outcome definition, coding vintage, care setting (ICU-weighted), and
population. Concatenating cohorts whose rates mean different things creates
between-dataset variance that would appear in the diagnostic as *diversity* —
manufacturing exactly the signal the paper claims to detect. That is not a
conservative bias; it is a mechanism for producing a false positive.

If a second cohort is used, it must be a **separate domain row** (like the
five-country table in Table 2), reported alongside UCI rather than merged into
it. Requirements:

- PhysioNet credentialing must already be in place. **Do not delay on approval**
  — the same instruction the original revision spec gave for MIMIC.
- Build cohorts with the same aggregation logic and an equivalent
  `min_encounters` threshold, so the two rows are comparable in construction
  even though their populations differ.
- Report the readmission-rate distributions of both side by side, so a reader
  can see how different the targets are before comparing diagnostics.

### Study C (cheap, do regardless) — is the ceiling the real problem?

Healthcare's baseline sits at 88–90%, i.e. already near nominal. Even with
infinite n, a strategy can only gain ~10pp there, and most gains would be
overcoverage. Test directly: **construct a harder healthcare target** by
tightening the rare-event definition (e.g. top-decile-rate cohorts, or a shorter
readmission window if derivable) and check whether the baseline drops enough to
create headroom.

If the baseline stays near nominal under every reasonable target, then
**healthcare is intrinsically a low-headroom domain**, and the paper should say
so plainly rather than continuing to present it as an underpowered version of
climate. That is a legitimate and reportable finding.

#### Mandatory guard: does the harder target reintroduce the Appendix A2 trap?

"Top-decile-rate cohorts" is itself a rare-event filter. If applied so that
cohort *selection* correlates mechanically with the *outcome*, it recreates the
degeneracy that made the original binary target unusable — rare-event count and
score diversity become the same variable, and any resulting headroom is fake.

**This must be checked, not assumed safe by analogy to the continuous-rate fix.**
Concretely, for every candidate target the write-up must report:

1. **Spearman ρ(rare-count, support-width) across the 200 draws.** The A2
   failure showed ρ = 0.68–0.72. Anything above **0.5** means the two predictors
   are collinear and the target is degenerate — report it as such and discard
   that target rather than reporting its headroom.
2. **Whether the target remains continuous.** Selecting *which cohorts* are in
   the pool is admissible; redefining the target itself as a binary
   in-decile/out-of-decile indicator is not, and would reproduce A2 exactly.
3. **Whether the filter is applied using outcome information that the
   nonconformity score also depends on.** If cohorts are selected *by their
   readmission rate* and then scored on error against that same rate, selection
   and outcome share a term. The admissible construction filters on a
   pre-specified covariate (e.g. cohort size, acuity mix, admission type) and
   lets the rate vary freely within the filtered population.

A target failing any of these is reported as attempted-and-rejected with the
measured ρ, in the same style as the dropped 1999 cutoff in Task 13 — not
silently replaced with a friendlier one.

---

## Deliverables

- `fix-reg/task15_healthcare_threshold_sweep.csv` — per threshold: cohorts, n,
  noise/signal, diagnostic ρ values, selector/augmentation coverage with Wilson
  intervals, and separation flags.
- `fix-reg/task15_healthcare_power.md` — write-up containing:
  - the noise/signal table and the chosen operating threshold, with reasoning;
  - whether anything separates, evaluated against the pre-registered rule;
  - an explicit verdict on whether healthcare is underpowered, low-headroom, or
    both — these have different implications for the paper;
  - the recorded rejection of dataset pooling and why.

## Guardrails

- **Do not pool datasets with different outcome definitions into one domain.**
  Between-dataset variance masquerades as diversity and would manufacture the
  paper's headline effect.
- Do not report a separation found only at noise/signal ≥ 1.0 as a finding.
- Do not change the cohort threshold in `domain_healthcare.py` itself — the
  published results use min=25 and must remain reproducible. The sweep is a
  parallel experiment.
- If MIMIC credentialing is not already in place, report Study B as not
  attempted with that reason, per the convention used for CPTC in Task 7.

## Suggested order

1. **Study A** — uses only existing data, answers the question directly.
2. **Study C** — cheap, and may reframe the problem entirely.
3. **Study B** — only if A and C leave the domain genuinely underpowered, and
   only as a separate domain row.
