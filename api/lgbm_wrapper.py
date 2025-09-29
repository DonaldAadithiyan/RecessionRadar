import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin

class LGBMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, params=None, num_boost_round=500):
        self.params = params or {}
        self.num_boost_round = num_boost_round
        self.model = None

    def fit(self, X, y):
        dtrain = lgb.Dataset(X, label=y)
        self.model = lgb.train(self.params, dtrain, num_boost_round=self.num_boost_round)
        return self

    def predict(self, X):
        return self.model.predict(X)