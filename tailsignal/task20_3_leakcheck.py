"""
Independent no-leakage audit of Item 3. Three checks:

  1. POSITIVE CONTROL — an oracle signal that DOES use the outcome must show a
     huge lift. If it doesn't, the quartile-test machinery is broken and the
     null result is uninterpretable.
  2. SHIFT INVARIANCE — Signal B at step t must be unchanged when the realized
     score at t and all later steps are overwritten with garbage. If it changes,
     the window is reaching forward.
  3. Signal A must be independent of all outcomes by construction.
"""
import os, re, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import task20_2_signalB as SB

T = pd.read_csv(os.path.join(HERE, "task20_3_traces.csv"))

# ---- 1. POSITIVE CONTROL ----
sub = T[(T.horizon == "6M") & (T.strategy == "pooled_trailing")].reset_index(drop=True)
d = sub[sub["miss"].notna()]
# A binary oracle makes quantile(.75)=0 flag everything, so jitter it into a
# continuous risk score that still encodes the outcome. This tests the quartile
# machinery on a signal known to be predictive.
rng0 = np.random.default_rng(1)
oracle = d["miss"].values.astype(float) + rng0.normal(0, 1e-3, len(d))
th = np.quantile(oracle, 0.75)
flag = oracle >= th
lift = 100 * d["miss"].values[flag].mean() - 100 * d["miss"].values.mean()
print(f"1. POSITIVE CONTROL (oracle = the outcome itself):")
print(f"   lift = {lift:+.2f} pp  -> {'machinery DETECTS a real signal' if lift > 30 else 'MACHINERY BROKEN'}")

# ---- 2. SHIFT INVARIANCE ----
rng = np.random.default_rng(0)
resolved = list(rng.random(40))
t = 20
q_c = 0.5
clean = SB.drift_ratio(resolved, t, q_c)
poisoned = list(resolved)
for i in range(t, len(poisoned)):
    poisoned[i] = 999.0                      # corrupt present + future
dirty = SB.drift_ratio(poisoned, t, q_c)
print(f"\n2. SHIFT INVARIANCE (corrupt scores at steps >= t):")
print(f"   clean={clean:.6f}  corrupted={dirty:.6f}  "
      f"-> {'NO forward reach' if clean == dirty else 'LEAKAGE DETECTED'}")

# ---- 3. Signal A independence ----
print(f"\n3. Signal A inputs: N (constant), alpha_t (state from steps < t), "
      f"cal scores (pre-test pool).")
# Only count CODE lines, not docstring prose (the docstring states it never
# uses Q_G, which would otherwise trip a naive grep).
_src = open(os.path.join(HERE, "task20_1_signalA.py")).read()
_code = re.sub(r'"""[\s\S]*?"""', "", _src)
print(f"   Q_G referenced in Signal A executable code: "
      f"{'NO' if 'Q_G' not in _code else 'YES'}")
_v = open(os.path.join(HERE, "task20_3_validation.py")).read()
_vcode = re.sub(r'"""[\s\S]*?"""', "", _v)
_vcode = "\n".join(l for l in _vcode.split("\n") if not l.strip().startswith("#"))
print(f"   Q_G referenced in Item 3 executable code:   "
      f"{'NO' if 'Q_G' not in _vcode else 'YES'}")
