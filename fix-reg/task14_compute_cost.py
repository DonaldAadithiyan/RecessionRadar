"""
TASK 14 — Compute cost of every calibration strategy.

Task 9 established a coverage-vs-sharpness tradeoff across the nine strategies.
A practitioner reading that section will also want the compute-vs-sharpness
axis: DtACI runs k experts in parallel, Bellman CI solves a dynamic program at
every step, and CPTC needs a fitted latent-state model, while the diversity
selector is a single two-pointer sweep. Those are very different cost profiles
and nobody has measured them.

What is measured, per strategy:
  - selection/setup time   one-off cost before the stream starts
                           (e.g. the selector's sweep, BCI's grid construction)
  - per-step time          amortised cost of producing one interval
  - total wall time        on a fixed-length stream
Timings are the median of several repeats to damp scheduler noise, and all
strategies are run on the SAME scores and stream so the comparison is like-for-
like.

Deliberately excluded: CPTC's HMM fitting is reported separately rather than
folded into its per-step cost, because it is a one-off training cost of a kind
no other strategy incurs — averaging it into per-step numbers would obscure
exactly the thing that makes CPTC different.

Output: fix-reg/task14_compute_cost.csv
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ensemble_stubs  # noqa: E402
ensemble_stubs.install()

from domain_common import run_aci, GAMMA_DEFAULT  # noqa: E402
import baselines_lib as B  # noqa: E402
import selector_lib as SEL  # noqa: E402
import augment_lib as AUG  # noqa: E402

OUT = "fix-reg"
N_FIX = 254
REPEATS = 5

rows = []


def timeit(fn, repeats=REPEATS):
    """Median wall time over `repeats` runs."""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def measure(domain, scores_pool, y_te, p_te):
    s_all = np.asarray(scores_pool, dtype=float)
    valid = np.where(np.isfinite(s_all))[0]
    trailing = s_all[valid[-N_FIX:]]
    T = len(y_te)

    def add(name, setup_s, stream_s, note=""):
        rows.append(dict(domain=domain, strategy=name, n_test=T,
                         setup_ms=round(setup_s * 1000, 3),
                         stream_ms=round(stream_s * 1000, 3),
                         per_step_us=round(stream_s / max(T, 1) * 1e6, 1),
                         total_ms=round((setup_s + stream_s) * 1000, 3),
                         note=note))

    # 1. pooled/trailing ACI — the reference
    add("pooled_trailing", 0.0,
        timeit(lambda: run_aci(y_te, p_te, trailing, gamma=GAMMA_DEFAULT)))

    # 2. Mondrian
    reg_pool = np.zeros(len(trailing), dtype=bool)
    reg_pool[-len(trailing) // 4:] = True
    cal_by = {False: trailing[~reg_pool], True: trailing[reg_pool]}
    te_reg = [bool(x) for x in (np.arange(T) % 2 == 0)]
    add("mondrian", 0.0,
        timeit(lambda: B.run_mondrian(y_te, p_te, cal_by, te_reg)))

    # 3. PID-conformal
    add("pid_conformal", 0.0, timeit(lambda: B.run_pid(y_te, p_te, trailing)))

    # 4. EVT-tail — refits a GPD at every step, so cost lives in the stream
    add("evt_tail", 0.0, timeit(lambda: B.run_evt(y_te, p_te, trailing)))

    # 5. diversity-optimal — one-off selection sweep, then plain ACI
    setup = timeit(lambda: SEL.support_width_selector(s_all, N_FIX))
    sel = s_all[SEL.support_width_selector(s_all, N_FIX)]
    sel = sel[np.isfinite(sel)]
    add("diversity_optimal", setup,
        timeit(lambda: run_aci(y_te, p_te, sel, gamma=GAMMA_DEFAULT)))

    # 6. DtACI — k experts updated per step
    add("dtaci", 0.0, timeit(lambda: B.run_dtaci(y_te, p_te, trailing)))

    # 7. AcMCP
    add("acmcp", 0.0, timeit(lambda: B.run_acmcp(y_te, p_te, trailing)))

    # 8. Bellman CI — dynamic program at every step
    add("bellman_ci", 0.0, timeit(lambda: B.run_bci(y_te, p_te, trailing),
                                  repeats=max(2, REPEATS // 2)))

    # 9. pool augmentation — one-off GPD fit + draws, then plain ACI
    setup_a = timeit(lambda: AUG.augment_pool(trailing, synth_frac=0.20,
                                              seed=3))
    aug, _ = AUG.augment_pool(trailing, synth_frac=0.20, seed=3)
    add("augmented_trailing", setup_a,
        timeit(lambda: run_aci(y_te, p_te, aug, gamma=GAMMA_DEFAULT)))

    got = [r for r in rows if r["domain"] == domain]
    print(f"\n  {domain} (T={T}):")
    for r in sorted(got, key=lambda x: x["total_ms"]):
        print(f"    {r['strategy']:22s} setup={r['setup_ms']:8.3f}ms  "
              f"stream={r['stream_ms']:9.3f}ms  "
              f"per-step={r['per_step_us']:9.1f}us")


print("=" * 92)
print("TASK 14 — compute cost per calibration strategy")
print("=" * 92)
print(f"  median of {REPEATS} repeats; identical scores and stream per domain")

import task_oof_and_probit as T_  # noqa: E402
measure("Recession-6M", np.abs(T_.oof_pred[:, 3] - T_.y_train[:, 3]),
        T_.y_test[:, 3], T_.preds_test[:, 3])

import runpy  # noqa: E402
g = runpy.run_path("fix-reg/domain_healthcare.py", run_name="_hc")
measure("Healthcare", g["SCORES"]["gradboost"], g["Y_TEST"],
        g["PREDS_TEST"]["gradboost"])

gc = runpy.run_path("fix-reg/domain_climate.py", run_name="_cl")
measure("Climate", gc["SCORES"]["gradboost"], gc["Y_TEST"],
        gc["PREDS_TEST"]["gradboost"])

# CPTC's one-off state-model cost, reported separately and honestly
print("\n" + "-" * 92)
print("CPTC's latent-state model — a one-off cost no other strategy incurs")
print("-" * 92)
try:
    sys.argv = ["task9c"]
    from task9c_cptc import GaussianHMM  # noqa: E402
    probe = np.abs(T_.oof_pred[:, 3] - T_.y_train[:, 3])
    probe = probe[np.isfinite(probe)]
    t_hmm = timeit(lambda: GaussianHMM(n_states=3).fit(probe), repeats=3)
    print(f"  Gaussian HMM fit (3 states, {len(probe)} points): "
          f"{t_hmm*1000:.1f}ms")
    print("  NOTE: the authors' method uses a REDSDS, which is a deep")
    print("  switching-dynamics model — its training cost is orders of")
    print("  magnitude above this HMM stand-in and is NOT represented here.")
    rows.append(dict(domain="(all)", strategy="cptc_state_model_fit",
                     n_test=np.nan, setup_ms=round(t_hmm * 1000, 3),
                     stream_ms=np.nan, per_step_us=np.nan,
                     total_ms=round(t_hmm * 1000, 3),
                     note="one-off HMM fit; real REDSDS would be far costlier"))
except Exception as e:
    print(f"  [skipped] {e}")

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/task14_compute_cost.csv", index=False)

print("\n" + "=" * 92)
print("PER-STEP COST (microseconds), relative to pooled/trailing ACI")
print("=" * 92)
d = df[df["domain"] != "(all)"]
piv = d.pivot_table(index="strategy", columns="domain", values="per_step_us")
base = piv.loc["pooled_trailing"]
rel = (piv / base).round(1)
print("\nabsolute (us/step):"); print(piv.round(1).to_string())
print("\nrelative to pooled ACI:"); print(rel.to_string())
print(f"\nSaved {OUT}/task14_compute_cost.csv")
