import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings('ignore')

recession_targets = [
    "recession_probability",
    "1_month_recession_probability", 
    "3_month_recession_probability",
    "6_month_recession_probability",
]

def sanitize_columns(df):
    """Clean column names for LightGBM compatibility"""
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    return df

epsilon = 1e-8
def safe_logit_transform(y):
    """Transform probabilities to logit space for better modeling"""
    y_clipped = np.clip(y, 0, 100)
    y_scaled = y_clipped / 100.0
    y_scaled = np.clip(y_scaled, epsilon, 1 - epsilon)
    return np.log(y_scaled / (1 - y_scaled))

def safe_inv_logit_transform(y_logit):
    """Transform predictions back from logit to probability space"""
    y_logit_clipped = np.clip(y_logit, -50, 50)
    y_prob = 1 / (1 + np.exp(-y_logit_clipped))
    return np.clip(y_prob * 100, 0, 100)


class LGBMWrapper(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible wrapper for LightGBM"""
    def __init__(self, params=None, num_boost_round=500, early_stopping_rounds=50):
        self.params = params or {
            "objective": "regression",
            "metric": "rmse",
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 30,
            "reg_alpha": 0.3,
            "reg_lambda": 0.3,
            "seed": 42,
            "verbose": -1,
        }
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, X, y):
        val_size = max(10, int(0.1 * len(X)))
        train_X, val_X = X[:-val_size], X[-val_size:]
        train_y, val_y = y[:-val_size], y[-val_size:]
        
        dtrain = lgb.Dataset(train_X, label=train_y)
        dval = lgb.Dataset(val_X, label=val_y, reference=dtrain)
        
        self.model = lgb.train(
            self.params, 
            dtrain, 
            num_boost_round=self.num_boost_round,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

# ========================================================================================
# FULL CHAIN BASE MODELS
# ========================================================================================

class FullChainCatBoostModel:
    """CatBoost with RegressorChain for ALL 4 targets"""
    def __init__(self):
        self.chain_model = None
        self.scaler = None
        
    def fit(self, X_train, y_train):
        self.scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transform all 4 targets
        y_train_transformed = safe_logit_transform(y_train[recession_targets].values)
        
        # Single chain for all 4 targets: 0 → 1 → 2 → 3
        base_model = CatBoostRegressor(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            subsample=0.8,
            random_seed=42,
            loss_function="RMSE",
            verbose=False
        )
        
        self.chain_model = RegressorChain(base_model, order=[0, 1, 2, 3])
        self.chain_model.fit(X_train_scaled, y_train_transformed)
        
    def predict(self, X_test):
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        preds_logit = self.chain_model.predict(X_test_scaled)
        preds = safe_inv_logit_transform(preds_logit)
        
        return np.clip(preds, 0, 100)

class FullChainLightGBMModel:
    """LightGBM with RegressorChain for ALL 4 targets"""
    def __init__(self):
        self.chain_model = None
        self.scaler = None
        self.lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 30,
            "reg_alpha": 0.3,
            "reg_lambda": 0.3,
            "seed": 42,
            "verbose": -1,
        }
        
    def fit(self, X_train, y_train):
        X_train = sanitize_columns(X_train)
        
        self.scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transform all 4 targets
        y_train_transformed = safe_logit_transform(y_train[recession_targets].values)
        
        # Single chain for all 4 targets
        chain_base = LGBMWrapper(params=self.lgb_params, num_boost_round=500)
        self.chain_model = RegressorChain(chain_base, order=[0, 1, 2, 3])
        self.chain_model.fit(X_train_scaled, y_train_transformed)
        
    def predict(self, X_test):
        X_test = sanitize_columns(X_test)
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        preds_logit = self.chain_model.predict(X_test_scaled)
        preds = safe_inv_logit_transform(preds_logit)
        
        return np.clip(preds, 0, 100)

class FullChainRandomForestModel:
    """Random Forest with RegressorChain for ALL 4 targets"""
    def __init__(self):
        self.chain_model = None
        self.scaler = None
        
    def fit(self, X_train, y_train):
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transform all 4 targets
        y_train_transformed = safe_logit_transform(y_train[recession_targets].values)
        
        rf_base = RandomForestRegressor(
            n_estimators=500,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features=0.8,
            max_samples=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        self.chain_model = RegressorChain(base_estimator=rf_base, order=[0, 1, 2, 3])
        self.chain_model.fit(X_train_scaled, y_train_transformed)
        
    def predict(self, X_test):
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        preds_logit = self.chain_model.predict(X_test_scaled)
        preds = safe_inv_logit_transform(preds_logit)
        
        return np.clip(preds, 0, 100)

# ========================================================================================
# STACKING ENSEMBLE
# ========================================================================================

class FullChainStackingEnsemble:
    """
    Stacking ensemble combining CatBoost, LightGBM, and Random Forest
    with full chain architecture
    """
    def __init__(self, cv_folds=8, use_feature_engineering=True):
        self.base_models = {
            'CatBoost': FullChainCatBoostModel,
            'LightGBM': FullChainLightGBMModel,
            'RandomForest': FullChainRandomForestModel
        }
        self.meta_models = {}
        self.cv_folds = cv_folds
        self.use_feature_engineering = use_feature_engineering
        self.meta_scaler = {}
        self.fitted_base_models = {}
            
    def _engineer_meta_features(self, *base_preds):
        """Create engineered features from base model predictions"""
        features = list(base_preds)
        
        if self.use_feature_engineering:
            # Mean prediction
            mean_pred = np.mean(base_preds, axis=0)
            features.append(mean_pred)
            
            # Weighted average (CatBoost=0.4, LightGBM=0.35, RF=0.25)
            weighted_avg = 0.4 * base_preds[0] + 0.35 * base_preds[1] + 0.25 * base_preds[2]
            features.append(weighted_avg)
            
            # Variance/disagreement
            pred_std = np.std(base_preds, axis=0)
            features.append(pred_std)
            
            # Min and Max
            features.append(np.min(base_preds, axis=0))
            features.append(np.max(base_preds, axis=0))
            
            # Pairwise differences
            for i in range(len(base_preds)):
                for j in range(i+1, len(base_preds)):
                    features.append(np.abs(base_preds[i] - base_preds[j]))
        
        return np.column_stack(features)
    
    def _get_cv_predictions(self, X_train, y_train):
        """Generate stratified cross-validation predictions"""
        recession_prob = y_train['recession_probability'].values
        bins = np.quantile(recession_prob, [0, 0.25, 0.5, 0.75, 1.0])
        stratify_labels = np.digitize(recession_prob, bins) - 1
        
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        n_samples = X_train.shape[0]
        n_targets = len(recession_targets)
        
        cv_predictions = {name: np.zeros((n_samples, n_targets)) 
                         for name in self.base_models.keys()}
        
        print(f"Generating CV predictions ({self.cv_folds} folds)...")
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, stratify_labels)):
            print(f"  Fold {fold_idx + 1}/{self.cv_folds}", end='\r')
            
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            
            for model_name, model_class in self.base_models.items():
                model = model_class()
                model.fit(X_fold_train, y_fold_train)
                cv_predictions[model_name][val_idx] = model.predict(X_fold_val)
        
        print(f"  ✓ Completed {self.cv_folds} folds" + " " * 20)
        return cv_predictions
    
    def fit(self, X_train, y_train):
        """Fit the stacking ensemble"""
        print(f"\n{'='*80}")
        print("TRAINING: FULL CHAIN STACKING ENSEMBLE")
        print("="*80)
        
        print("\n[1/3] Training base models...")
        for model_name, model_class in self.base_models.items():
            print(f"  Training {model_name}...")
            model = model_class()
            model.fit(X_train, y_train)
            self.fitted_base_models[model_name] = model
        
        print("\n[2/3] Generating CV predictions...")
        cv_predictions = self._get_cv_predictions(X_train, y_train)
        
        print("\n[3/3] Training meta-models...")
        for i, target in enumerate(recession_targets):
            print(f"  {target}...", end='')
            
            # Extract predictions for this target from all base models
            base_preds_for_target = [cv_predictions[name][:, i] 
                                    for name in self.base_models.keys()]
            
            # Create meta features
            meta_features = self._engineer_meta_features(*base_preds_for_target)
            
            # Scale
            scaler = StandardScaler()
            meta_features_scaled = scaler.fit_transform(meta_features)
            self.meta_scaler[target] = scaler
            
            # Nested CV for hyperparameter tuning
            kf_inner = KFold(n_splits=3, shuffle=True, random_state=42)
            best_score = -np.inf
            best_alpha = 0.1
            
            for alpha in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
                meta_model_test = ElasticNet(alpha=alpha, l1_ratio=0.5, 
                                            random_state=42, max_iter=2000)
                cv_scores = []
                
                for train_inner_idx, val_inner_idx in kf_inner.split(meta_features_scaled):
                    meta_model_test.fit(
                        meta_features_scaled[train_inner_idx], 
                        y_train[target].values[train_inner_idx]
                    )
                    pred = meta_model_test.predict(meta_features_scaled[val_inner_idx])
                    score = r2_score(y_train[target].values[val_inner_idx], pred)
                    cv_scores.append(score)
                
                avg_score = np.mean(cv_scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_alpha = alpha
            
            # Train final meta-model
            meta_model = ElasticNet(alpha=best_alpha, l1_ratio=0.5, 
                                   random_state=42, max_iter=2000)
            meta_model.fit(meta_features_scaled, y_train[target])
            self.meta_models[target] = meta_model
            
            print(f" α={best_alpha:.2f}, R²={best_score:.4f}")
        
        print(f"\n{'='*80}")
        print("✓ TRAINING COMPLETED")
        print("="*80)
        
    def predict(self, X_test):
        """Make predictions"""
        # Get predictions from all base models
        base_predictions = {}
        for model_name, model in self.fitted_base_models.items():
            base_predictions[model_name] = model.predict(X_test)
        
        # Generate final predictions
        final_predictions = np.zeros_like(list(base_predictions.values())[0])
        
        for i, target in enumerate(recession_targets):
            # Extract predictions for this target
            base_preds_for_target = [base_predictions[name][:, i] 
                                    for name in self.base_models.keys()]
            
            # Create meta features
            meta_features = self._engineer_meta_features(*base_preds_for_target)
            meta_features_scaled = self.meta_scaler[target].transform(meta_features)
            
            # Predict
            final_predictions[:, i] = self.meta_models[target].predict(meta_features_scaled)
            
        return np.clip(final_predictions, 0, 100)
