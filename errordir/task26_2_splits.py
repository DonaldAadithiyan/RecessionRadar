"""
TASK 26, Item 2 — build the Beijing PM2.5 dataset and BOTH split constructions.

Both conditions draw from the SAME underlying rows and use the SAME
fit/calibration/test counts. The only difference is how rows are assigned:

  TEMPORAL : chronological. FIT = earliest, CAL = next, TEST = latest.
             Train/calibrate on the past, test on the future.
  RANDOM   : the same total counts, assigned by random shuffle.

Matched sizes are asserted at runtime, not assumed.

FEATURES are strictly lagged (shift>=1) so neither condition can see the
current hour's own pollution. This matters especially for the RANDOM
condition: without lagging, a random split would leak neighbouring hours'
targets through contemporaneous weather and the comparison would be
meaningless.

Outputs: data/domains/task26/pm25.csv, errordir/task26_2_splits.csv
"""
import os
import io
import urllib.request
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
os.chdir(ROOT)

OUT_D = "data/domains/task26"
OUT = "errordir"
os.makedirs(OUT_D, exist_ok=True)
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00381/"
       "PRSA_data_2010.1.1-2014.12.31.csv")
PATH = f"{OUT_D}/pm25.csv"

SEED = 26
N_TEST = 600
N_CAL_TARGET = 1200
N_FIT_TARGET = 1800


def build():
    if os.path.exists(PATH):
        return pd.read_csv(PATH)
    raw = urllib.request.urlopen(URL, timeout=120).read()
    df = pd.read_csv(io.BytesIO(raw))
    df = df.sort_values(["year", "month", "day", "hour"]).reset_index(drop=True)
    df["cbwd"] = df["cbwd"].astype("category").cat.codes
    df["target"] = df["pm2.5"].astype(float)

    # STRICTLY LAGGED predictors — no same-hour information of any kind.
    for lag in [1, 2, 3, 24]:
        df[f"pm_lag{lag}"] = df["target"].shift(lag)
    df["pm_roll24"] = df["target"].shift(1).rolling(24).mean()
    df["pm_roll168"] = df["target"].shift(1).rolling(168).mean()
    for c in ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir", "cbwd"]:
        df[f"{c}_lag1"] = df[c].shift(1)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["mon_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["mon_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    feats = ([c for c in df.columns if c.startswith(("pm_lag", "pm_roll"))]
             + [f"{c}_lag1" for c in ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir", "cbwd"]]
             + ["hour_sin", "hour_cos", "mon_sin", "mon_cos"])
    d = df[["year", "month", "day", "hour"] + feats + ["target"]].dropna()
    d = d.reset_index(drop=True)
    d.to_csv(PATH, index=False)
    return d


def make_splits(d, seed=SEED):
    """
    Returns (temporal, random) index dicts. Both use identical counts.
    A contiguous block of the most recent rows is used for the temporal
    condition; the random condition shuffles the SAME row set.
    """
    n_need = N_FIT_TARGET + N_CAL_TARGET + N_TEST
    # use the most recent n_need rows so both conditions share identical rows
    base = np.arange(len(d) - n_need, len(d))
    assert len(base) == n_need

    temporal = dict(fit=base[:N_FIT_TARGET],
                    cal=base[N_FIT_TARGET:N_FIT_TARGET + N_CAL_TARGET],
                    test=base[N_FIT_TARGET + N_CAL_TARGET:])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(base)
    random_ = dict(fit=np.sort(perm[:N_FIT_TARGET]),
                   cal=np.sort(perm[N_FIT_TARGET:N_FIT_TARGET + N_CAL_TARGET]),
                   test=np.sort(perm[N_FIT_TARGET + N_CAL_TARGET:]))
    return temporal, random_, base


if __name__ == "__main__":
    d = build()
    T, R, base = make_splits(d)
    print("=" * 92)
    print("TASK 26 Item 2 — matched split construction (Beijing PM2.5)")
    print("=" * 92)
    print(f"  usable rows after lagging: {len(d):,}")
    print(f"  rows used (shared by BOTH conditions): {len(base):,}\n")

    rows = []
    for nm, S in [("temporal", T), ("random", R)]:
        print(f"  {nm:9s} fit={len(S['fit']):5d}  cal={len(S['cal']):5d}  "
              f"test={len(S['test']):5d}")
        rows.append(dict(condition=nm, n_fit=len(S["fit"]), n_cal=len(S["cal"]),
                         n_test=len(S["test"])))

    # ── matched-size verification (the guardrail) ──────────────────────────
    ok_sizes = all(len(T[k]) == len(R[k]) for k in ["fit", "cal", "test"])
    ok_rows = bool(np.array_equal(np.sort(np.concatenate(list(T.values()))),
                                  np.sort(np.concatenate(list(R.values())))))
    ok_disj_T = len(np.intersect1d(T["fit"], T["cal"])) == 0 and \
        len(np.intersect1d(T["cal"], T["test"])) == 0 and \
        len(np.intersect1d(T["fit"], T["test"])) == 0
    ok_disj_R = len(np.intersect1d(R["fit"], R["cal"])) == 0 and \
        len(np.intersect1d(R["cal"], R["test"])) == 0 and \
        len(np.intersect1d(R["fit"], R["test"])) == 0

    print(f"\n  MATCHED SIZES across conditions:      {ok_sizes}")
    print(f"  IDENTICAL underlying row set:         {ok_rows}")
    print(f"  fit/cal/test disjoint (temporal):     {ok_disj_T}")
    print(f"  fit/cal/test disjoint (random):       {ok_disj_R}")

    # temporal ordering sanity: temporal test must be strictly later than fit
    later = bool(T["test"].min() > T["fit"].max())
    overlap_r = float(np.mean(R["test"] < np.median(R["fit"])))
    print(f"  temporal TEST strictly after FIT:     {later}")
    print(f"  random TEST rows interleaved with FIT: "
          f"{overlap_r:.2f} fraction before FIT median (≈0.5 expected)")

    for r in rows:
        r.update(matched_sizes=ok_sizes, identical_rowset=ok_rows,
                 disjoint=ok_disj_T and ok_disj_R)
    pd.DataFrame(rows).to_csv(f"{OUT}/task26_2_splits.csv", index=False)
    print(f"\nSaved {PATH}, {OUT}/task26_2_splits.csv")
