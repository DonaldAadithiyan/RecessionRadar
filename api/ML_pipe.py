import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.multioutput import RegressorChain
# from statsmodels.tsa.seasonal import STL
from lgbm_wrapper import LGBMWrapper
from scipy.stats import boxcox
import lightgbm, catboost

import pickle
import os
import warnings
import re
warnings.filterwarnings("ignore")

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_reg_models():
    if os.path.exists('../models'):
        base_model = load_model('../models/recession_chain_model.pkl')
        one_three_month_model = load_model('../models/lgbm_recession_chain_model.pkl')
        six_month_model = load_model('../models/lgbm_recession_6m_model.pkl')
    else:
        base_model = load_model('models/catboost_recession_chain_model.pkl')
        one_three_month_model = load_model('models/lgbm_recession_chain_model.pkl')
        six_month_model = load_model('models/lgbm_recession_6m_model.pkl')

    return base_model, one_three_month_model, six_month_model

def is_anomaly(col_name: str, value: float, stats_dict) -> bool:
    """
    Check if a single value is an anomaly based on precomputed stats.

    Parameters
    ----------
    col_name : str
        Name of the column (e.g., 'CPI').
    value : float
        The new observation to check.
    stats_dict : dict
        Dictionary containing mean, std, lower_bound, upper_bound.

    Returns
    -------
    bool
        True if value is outside the allowed bounds, else False.
    """
    bounds = stats_dict[col_name]
    return not (bounds["lower_bound"] <= value <= bounds["upper_bound"])

def logit_transform(y):
    epsilon = 1e-6
    y_scaled = np.clip(y / 100, epsilon, 1 - epsilon)
    return np.log(y_scaled / (1 - y_scaled))

def inv_logit_transform(y_logit):
    y_prob = 1 / (1 + np.exp(-y_logit))
    return y_prob * 100

def sanitize_columns(df):
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    return df

def clean_data(X_or_y):
    X_or_y = X_or_y.replace([np.inf, -np.inf], np.nan)
    X_or_y = X_or_y.ffill().bfill()
    X_or_y = X_or_y.fillna(0)
    return X_or_y


def time_series_feature_eng():
    pass


def time_series_feature_reduction():
    pass


def regresstion_feature_engineering(input_data = None):
    if os.path.exists('../data/fix'):
        train_df = pd.read_csv('../data/fix/feature_selected_recession_train.csv')
        test_df = pd.read_csv('../data/fix/feature_selected_recession_test.csv')
    else:
        train_df = pd.read_csv('data/fix/feature_selected_recession_train.csv')
        test_df = pd.read_csv('data/fix/feature_selected_recession_test.csv')

    # Ensure 'date' is datetime
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])

    # Combine into one DataFrame
    full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    full_df = full_df.sort_values('date').reset_index(drop=True)
    
    ######
    recession_targets = [
        "recession_probability",
        "1_month_recession_probability",
        "3_month_recession_probability",
        "6_month_recession_probability"
    ]
    anomaly_cols = [
        'CPI', 'unemployment_rate', 
        'share_price', '3_months_rate', '6_months_rate', '10_year_rate'
    ]
    selected_features = [
        'share_price', 'OECD_CLI_index', 'CSI_index', 'gdp_per_capita_residual', 'gdp_per_capita_rollstd12', 'PPI_diff1', 
        'gdp_per_capita_diff1', 'gdp_per_capita_diff3', 'gdp_per_capita_pct_change1', '1_year_rate', '3_months_rate', 
        '6_months_rate', 'CPI', 'INDPRO', '10_year_rate', 'unemployment_rate', 'PPI', 'gdp_per_capita', 'CSI_index_trend', 
        'OECD_CLI_index_trend', 'PPI_CPI_diff', '3_months_rate_rollmin12', 'INDPRO_rollstd12', 'PPI_rollstd12', 
        '3_months_rate_diff3', 'OECD_CLI_index_diff1', 'CSI_index_rollmin12', 'OECD_CLI_index_diff3', 'OECD_CLI_index_residual', 
        'share_price_trend', 'CSI_index_lag6', 'unemployment_rate_rollstd6', 'PPI_rollstd6', 'OECD_CLI_index_rollmax12', 'CSI_index_rollmax6', 
        'INDPRO_diff3', 'unemployment_rate_diff3', 'CSI_index_rollmin6', 'OECD_CLI_index_pct_change1', '10_year_rate_residual'
    ]
    if os.path.exists('../anomaly_models'):
        with open('../anomaly_models/anomaly_stats.pkl', 'rb') as f:
            anomaly_stats = pickle.load(f)
    else:
        with open('anomaly_models/anomaly_stats.pkl', 'rb') as f:
            anomaly_stats = pickle.load(f)
            
    lags = [1, 3, 6]  # months
    windows = [3, 6, 12]
    indicators = [
        '1_year_rate', '3_months_rate', '6_months_rate', 'CPI', 'INDPRO', 
        '10_year_rate', 'share_price', 'unemployment_rate', 'PPI', 
        'OECD_CLI_index', 'CSI_index', 'gdp_per_capita'
    ]
    ########
    
    if input_data:      # apply boxcox transform to share_price if provided
        if 'share_price' in input_data:
            share_price_val = input_data['share_price']
            if share_price_val > 0:
                transformed, _ = boxcox([share_price_val])
                input_data['share_price'] = transformed[0]
                
        for col in input_data:
            if col in full_df.columns:
                full_df.at[full_df.index[-1], col] = input_data[col]
 
    # Interaction features
    full_df['CPI_unemployment_interaction'] = full_df['CPI'] * full_df['unemployment_rate']
    full_df['INDPRO_CPI_ratio'] = full_df['INDPRO'] / (full_df['CPI'] + 1e-6)
    full_df['share_gdp_ratio'] = full_df['share_price'] / (full_df['gdp_per_capita'] + 1e-6)
    full_df['PPI_CPI_diff'] = full_df['PPI'] - full_df['CPI']
    full_df['interest_spread'] = full_df['10_year_rate'] - full_df['3_months_rate']
    
    for col in indicators:
        for lag in lags:
            full_df[f"{col}_lag{lag}"] = full_df[col].shift(lag)
            
        for window in windows:
            full_df[f"{col}_rollmean{window}"] = full_df[col].shift(1).rolling(window).mean()
            full_df[f"{col}_rollstd{window}"]  = full_df[col].shift(1).rolling(window).std()
            full_df[f"{col}_rollmax{window}"]  = full_df[col].shift(1).rolling(window).max()
            full_df[f"{col}_rollmin{window}"]  = full_df[col].shift(1).rolling(window).min()

        full_df[f"{col}_diff1"] = full_df[col] - full_df[col].shift(1)
        full_df[f"{col}_diff3"] = full_df[col] - full_df[col].shift(3)
        full_df[f"{col}_pct_change1"] = full_df[col].pct_change(1)
        
    df_reduced = full_df[selected_features + recession_targets + ["date"]].copy()
    
    for col in anomaly_cols:
        df_reduced[f"{col}_anomaly"] = full_df[col].apply(lambda x: is_anomaly(col, x, anomaly_stats)).astype(int)
                

    return df_reduced


def time_series_prediction():
    pass

def regression_prediction(input_data=None):
    # Feature engineering
    fe_data = regresstion_feature_engineering(input_data)
    
    # Load models
    base_model, one_three_month_model, six_month_model = load_reg_models()
    
    #### prepare x data for base, 1m, 3m
    recession_targets = [
        "recession_probability",
        "1_month_recession_probability",
        "3_month_recession_probability",
        "6_month_recession_probability",
    ]

    chain_targets = recession_targets[:3]
    features = [col for col in fe_data.columns if col not in [*recession_targets,"date"]]
    
    X = fe_data[features].iloc[[-1]]
    X_base = clean_data(X.copy())
    X_1m_3m = sanitize_columns(clean_data(X.copy()))
    
    ## base, 1m, 3m prediction
    base_pred, *_ = inv_logit_transform(base_model.predict(X_base)).clip(0, 100)[-1]
    _, one_month_pred, three_month_pred = inv_logit_transform(one_three_month_model.predict(X_1m_3m)).clip(0, 100)[-1]
    
    ## prepare x data for 6m
    X_6m = X_1m_3m.copy()
    X_6m[chain_targets] = [base_pred, one_month_pred, three_month_pred]
    
    six_month_pred = inv_logit_transform(six_month_model.predict(X_6m)).clip(0, 100)[-1]
    
    
    print(f"Base: {base_pred}, 1m: {one_month_pred}, 3m: {three_month_pred}, 6m: {six_month_pred}")
    
    return base_pred/100, one_month_pred/100, three_month_pred/100, six_month_pred/100


if __name__ == "__main__":
    base_pred, one_month_pred, three_month_pred, six_month_pred = regression_prediction()
    print(f"Base Prediction: {base_pred}")
    print(f"1-Month Prediction: {one_month_pred}")
    print(f"3-Month Prediction: {three_month_pred}")
    print(f"6-Month Prediction: {six_month_pred}")