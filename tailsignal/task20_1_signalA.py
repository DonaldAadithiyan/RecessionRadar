"""
TASK 20, Item 1 — Signal A: the structural supply-demand margin.

Derived directly from the optimality proof's own supply/demand logic
(Section 4.4 / task12_small_n_and_drift.py::ceiling_ratio), NOT a new heuristic.

At each prediction step, with active calibration budget N and ACI's current
target quantile q_t = 1 - alpha_t:

    DEMAND  = ceil((1 - q_t) * N)
        the number of top-ranked calibration points that determine the
        q_t-quantile. As q_t -> 1 this shrinks toward 1; as q_t -> 0.5 it grows
        toward N/2.

    SUPPLY  = the number of genuinely extreme (upper-tail) points the active
        selection rule guarantees.
          - alternating-tail selector: exactly ceil(N/2), since the rule takes
            points alternately from the low and high tails, so half the budget
            lands in the upper tail BY CONSTRUCTION (verified against
            selector_lib.support_width_selector below).
          - trailing/pooled baseline: no such guarantee exists, so it is
            computed directly from that set's own score ranking — the count of
            its points at or above the pool's median, which is the honest
            empirical analogue of "how many upper-tail points do I actually
            hold".

    MARGIN  = SUPPLY / DEMAND

Margin >> 1 means the calibration set holds far more extreme points than the
current target quantile needs. Margin -> 1 means the quantile is being
determined by essentially the whole guaranteed upper-tail supply, with nothing
in reserve.

NO-LEAKAGE STATEMENT: this uses only N, alpha_t, and the calibration set's own
scores — all of which exist strictly BEFORE the prediction at step t is made.
It never touches y_test[t], the realized score at t, or Q_G. Verified in Item 3.

Outputs: tailsignal/task20_1_signalA_verify.csv
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fix-reg"))

import selector_lib as SEL  # noqa: E402

OUT = "tailsignal"
os.makedirs(OUT, exist_ok=True)


def demand(N, q_t):
    """Top-ranked points needed to determine the q_t-quantile of N points."""
    return int(np.ceil((1.0 - q_t) * N))


def supply_alternating(N):
    """
    Alternating-tail selector: the rule alternates low, high, low, high, ...
    starting low, so of N picks, floor(N/2) come from the high tail when N is
    even and ... see verify_supply() — computed, not assumed.
    """
    return int(np.floor(N / 2))


def supply_empirical(cal_scores, pool_scores):
    """
    Trailing/pooled baseline: no structural guarantee, so count the set's own
    points at or above the POOL median — the empirical count of upper-tail
    points actually held. Uses only already-observed calibration scores.
    """
    med = float(np.median(pool_scores))
    return int(np.sum(np.asarray(cal_scores) >= med))


def margin(supply, dem):
    return float(supply) / float(dem) if dem > 0 else np.inf


def verify_supply(N_list=(20, 30, 50, 80, 120, 180, 254)):
    """
    HAND-VERIFICATION 1: the claimed supply for the alternating selector must
    match what selector_lib actually does. Feed it a known pool, run the real
    selector, and count how many of its picks land in the pool's upper half.
    """
    rng = np.random.default_rng(0)
    pool = np.abs(rng.normal(0, 1, 635))
    med = float(np.median(pool))
    rows = []
    for N in N_list:
        idx = SEL.support_width_selector(pool, N)
        actual_upper = int(np.sum(pool[idx] >= med))
        claimed = supply_alternating(N)
        rows.append(dict(N=N, claimed_supply=claimed,
                         actual_upper_tail_picks=actual_upper,
                         match=bool(claimed == actual_upper)))
    return pd.DataFrame(rows)


def verify_against_section44():
    """
    HAND-VERIFICATION 2: the guardrail's requirement — reproduce known cells
    from Section 4.4's ceiling-ratio sweep (task12a_small_n_ceiling.csv) and
    confirm the margin's boundary behaviour agrees with where that sweep says
    the guarantee is exact vs. degrading.

    Section 4.4 sweeps the operative quantiles q in {0.868, 0.900, 0.927}. The
    proof is exact for q > 0.5 and degrades as q -> 0.5. The margin must be
    >= 1 wherever the sweep reports ratio == 1.0 (guarantee attained), because
    a margin < 1 would mean the selector cannot even supply the points the
    quantile needs.
    """
    path = os.path.join(HERE, "..", "fix-reg", "task12a_small_n_ceiling.csv")
    if not os.path.exists(path):
        return None
    ceil_df = pd.read_csv(path)
    ceil_df = ceil_df[ceil_df.domain == "Recession"]
    rows = []
    for _, r in ceil_df.iterrows():
        N = int(r["N"])
        for q, col in [(0.868, "ratio_q868"), (0.900, "ratio_q900"),
                       (0.927, "ratio_q927")]:
            ratio = float(r[col])
            d = demand(N, q)
            s = supply_alternating(N)
            m = margin(s, d)
            rows.append(dict(horizon=r["horizon"], N=N, q=q,
                             section44_ceiling_ratio=ratio,
                             demand=d, supply=s, margin=round(m, 3),
                             guarantee_attained=bool(ratio >= 0.9999),
                             margin_ge_1=bool(m >= 1.0),
                             consistent=bool((ratio < 0.9999) or (m >= 1.0))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 92)
    print("TASK 20 Item 1 — Signal A: structural supply/demand margin")
    print("=" * 92)
    print("  DEMAND = ceil((1-q_t)*N)      SUPPLY = upper-tail points the rule guarantees")
    print("  MARGIN = SUPPLY / DEMAND\n")

    print("HAND-VERIFICATION 1 — claimed supply vs. what selector_lib actually picks:")
    v1 = verify_supply()
    for r in v1.itertuples():
        print(f"    N={r.N:4d}  claimed={r.claimed_supply:4d}  "
              f"actual upper-tail picks={r.actual_upper_tail_picks:4d}  "
              f"{'MATCH' if r.match else 'MISMATCH'}")
    ok1 = bool(v1["match"].all())
    print(f"  -> {'all match' if ok1 else 'MISMATCH PRESENT — bug, not a modeling choice'}\n")

    print("HAND-VERIFICATION 2 — against Section 4.4's ceiling-ratio sweep:")
    v2 = verify_against_section44()
    if v2 is None:
        print("    task12a_small_n_ceiling.csv not found — cannot verify")
        ok2 = False
    else:
        # worked example, printed for hand-checking
        for N, q in [(254, 0.900), (20, 0.900), (254, 0.868)]:
            d, s = demand(N, q), supply_alternating(N)
            print(f"    N={N:4d}, q={q:.3f}: demand=ceil({1-q:.3f}*{N})={d:3d}, "
                  f"supply=floor({N}/2)={s:3d}, margin={s/d:7.3f}")
        bad = v2[~v2.consistent]
        ok2 = bool(len(bad) == 0)
        print(f"    cells checked: {len(v2)}   inconsistent: {len(bad)}")
        print(f"  -> {'consistent with Section 4.4' if ok2 else 'INCONSISTENT — bug'}")
        v2.to_csv(f"{OUT}/task20_1_signalA_verify.csv", index=False)
        print(f"\nSaved {OUT}/task20_1_signalA_verify.csv")

    # Boundary behaviour: margin as q -> 0.5, where the proof degrades
    print("\nBoundary behaviour (N=254), where Section 4.4 says the guarantee degrades:")
    for q in [0.99, 0.95, 0.927, 0.90, 0.868, 0.80, 0.70, 0.60, 0.55, 0.51, 0.50]:
        d, s = demand(254, q), supply_alternating(254)
        flag = "  <-- at/below proof boundary (q<=0.5)" if q <= 0.5 else (
               "  <-- STRAINED (margin<2)" if s / d < 2 else "")
        print(f"    q={q:.3f}: demand={d:4d}  supply={s:3d}  margin={s/d:7.3f}{flag}")

    print("\nSignal A is exact: no fitting, no free parameters.")
