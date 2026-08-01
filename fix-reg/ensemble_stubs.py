"""
Stub classes required to unpickle the saved stacking ensemble.

The ensemble in fix-reg/models/full_chain_stacking.pkl was pickled from a script
running as __main__, so pickle looks for these class definitions in the
__main__ module of whatever process loads it. Any script that loads the model
must therefore register them into __main__ first.

Use `install()` rather than copying the class definitions again:

    import ensemble_stubs
    ensemble_stubs.install()
    ensemble = pickle.load(open(MODEL_PATH, "rb"))

This replaces the copy-paste blocks that previously appeared in task1_aci.py,
task_phase*.py and task_oof_and_probit.py, which is what made loading the model
from a new entry point fail.
"""

import re
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

recession_targets = [
    "recession_probability", "1_month_recession_probability",
    "3_month_recession_probability", "6_month_recession_probability",
]
LABELS = ["Current", "1M", "3M", "6M"]
eps = 1e-8


def safe_logit(y):
    p = np.clip(np.clip(y, 0, 100) / 100, eps, 1 - eps)
    return np.log(p / (1 - p))


def safe_inv_logit(z):
    return np.clip(1 / (1 + np.exp(-np.clip(z, -50, 50))) * 100, 0, 100)


def sanitize_columns(df):
    df = df.copy()
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', c) for c in df.columns]
    return df


class LGBMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, params=None, num_boost_round=500, early_stopping_rounds=50):
        self.params = params or {}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.model.predict(X)


class FullChainCatBoostModel:
    def __init__(self):
        self.chain_model = None; self.scaler = None

    def predict(self, X):
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return np.clip(safe_inv_logit(self.chain_model.predict(Xs)), 0, 100)


class FullChainLightGBMModel:
    def __init__(self):
        self.chain_model = None; self.scaler = None

    def predict(self, X):
        X = sanitize_columns(X)
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return np.clip(safe_inv_logit(self.chain_model.predict(Xs)), 0, 100)


class FullChainRandomForestModel:
    def __init__(self):
        self.chain_model = None; self.scaler = None

    def predict(self, X):
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return np.clip(safe_inv_logit(self.chain_model.predict(Xs)), 0, 100)


class FullChainStackingEnsemble:
    def __init__(self, cv_folds=8, use_feature_engineering=True):
        self.base_models = {'CatBoost': FullChainCatBoostModel,
                            'LightGBM': FullChainLightGBMModel,
                            'RandomForest': FullChainRandomForestModel}
        self.meta_models = {}; self.cv_folds = cv_folds
        self.use_feature_engineering = use_feature_engineering
        self.meta_scaler = {}; self.fitted_base_models = {}

    def _engineer_meta_features(self, *bp):
        f = list(bp)
        if self.use_feature_engineering:
            f += [np.mean(bp, axis=0), 0.4 * bp[0] + 0.35 * bp[1] + 0.25 * bp[2],
                  np.std(bp, axis=0), np.min(bp, axis=0), np.max(bp, axis=0)]
            for i in range(len(bp)):
                for j in range(i + 1, len(bp)):
                    f.append(np.abs(bp[i] - bp[j]))
        return np.column_stack(f)

    def predict(self, X):
        bp = {n: m.predict(X) for n, m in self.fitted_base_models.items()}
        fp = np.zeros_like(list(bp.values())[0])
        for i, t in enumerate(recession_targets):
            bpt = [bp[n][:, i] for n in self.base_models]
            mf = self._engineer_meta_features(*bpt)
            fp[:, i] = self.meta_models[t].predict(self.meta_scaler[t].transform(mf))
        return np.clip(fp, 0, 100)


_STUBS = [FullChainStackingEnsemble, FullChainCatBoostModel,
          FullChainLightGBMModel, FullChainRandomForestModel, LGBMWrapper,
          safe_logit, safe_inv_logit, sanitize_columns]


def install():
    """Register the stub classes into __main__ so pickle.load can find them."""
    main = sys.modules["__main__"]
    for obj in _STUBS:
        if not hasattr(main, obj.__name__):
            setattr(main, obj.__name__, obj)
