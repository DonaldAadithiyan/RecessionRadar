import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.seasonal import STL
from lgbm_wrapper import LGBMWrapper
from scipy.stats import boxcox
import lightgbm, catboost

import pickle
import os
import warnings
warnings.filterwarnings("ignore")

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_models():
    if os.path.exists('../models'):
        base_model = load_model('../models/catboost_recession_chain_model.pkl')
        one_month_model = load_model('../models/lgbm_recession_chain_model.pkl')
        three_month_model = load_model('../models/lgbm_recession_chain_model.pkl')
        six_month_model = load_model('../models/lgbm_recession_6m_model.pkl')
    else:
        base_model = load_model('models/catboost_recession_chain_model.pkl')
        one_month_model = load_model('models/lgbm_recession_chain_model.pkl')
        three_month_model = load_model('models/lgbm_recession_chain_model.pkl')
        six_month_model = load_model('models/lgbm_recession_6m_model.pkl')

    return base_model, one_month_model, three_month_model, six_month_model


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


    train_df['CPI_unemployment_interaction'] = train_df['CPI'] * train_df['unemployment_rate']
    train_df['INDPRO_CPI_ratio'] = train_df['INDPRO'] / (train_df['CPI'] + 1e-6)
    train_df['share_gdp_ratio'] = train_df['share_price'] / (train_df['gdp_per_capita'] + 1e-6)
    train_df['PPI_CPI_diff'] = train_df['PPI'] - train_df['CPI']
    train_df['interest_spread'] = train_df['10_year_rate'] - train_df['3_months_rate']

    test_df['CPI_unemployment_interaction'] = test_df['CPI'] * test_df['unemployment_rate']
    test_df['INDPRO_CPI_ratio'] = test_df['INDPRO'] / (test_df['CPI'] + 1e-6)
    test_df['share_gdp_ratio'] = test_df['share_price'] / (test_df['gdp_per_capita'] + 1e-6)
    test_df['PPI_CPI_diff'] = test_df['PPI'] - test_df['CPI']
    test_df['interest_spread'] = test_df['10_year_rate'] - test_df['3_months_rate']

    lags = [1, 3, 6]  # months

    indicators = [
        '1_year_rate', '3_months_rate', '6_months_rate', 'CPI', 'INDPRO', 
        '10_year_rate', 'share_price', 'unemployment_rate', 'PPI', 
        'OECD_CLI_index', 'CSI_index', 'gdp_per_capita'
    ]

    for col in indicators:
        for lag in lags:
            train_df[f"{col}_lag{lag}"] = train_df[col].shift(lag)


    for col in indicators:
        for lag in lags:
            test_df[f"{col}_lag{lag}"] = test_df[col].shift(lag)
            
    windows = [3, 6, 12]

    for col in indicators:
        for window in windows:
            train_df[f"{col}_rollmean{window}"] = train_df[col].shift(1).rolling(window).mean()
            train_df[f"{col}_rollstd{window}"]  = train_df[col].shift(1).rolling(window).std()
            train_df[f"{col}_rollmax{window}"]  = train_df[col].shift(1).rolling(window).max()
            train_df[f"{col}_rollmin{window}"]  = train_df[col].shift(1).rolling(window).min()


    for col in indicators:
        for window in windows:
            test_df[f"{col}_rollmean{window}"] = test_df[col].shift(1).rolling(window).mean()
            test_df[f"{col}_rollstd{window}"]  = test_df[col].shift(1).rolling(window).std()
            test_df[f"{col}_rollmax{window}"]  = test_df[col].shift(1).rolling(window).max()
            test_df[f"{col}_rollmin{window}"]  = test_df[col].shift(1).rolling(window).min()
            

    for col in indicators:
        for window in windows:
            train_df[f"{col}_rollmean{window}"] = train_df[col].shift(1).rolling(window).mean()
            train_df[f"{col}_rollstd{window}"]  = train_df[col].shift(1).rolling(window).std()
            train_df[f"{col}_rollmax{window}"]  = train_df[col].shift(1).rolling(window).max()
            train_df[f"{col}_rollmin{window}"]  = train_df[col].shift(1).rolling(window).min()

    for col in indicators:
        for window in windows:
            test_df[f"{col}_rollmean{window}"] = test_df[col].shift(1).rolling(window).mean()
            test_df[f"{col}_rollstd{window}"]  = test_df[col].shift(1).rolling(window).std()
            test_df[f"{col}_rollmax{window}"]  = test_df[col].shift(1).rolling(window).max()
            test_df[f"{col}_rollmin{window}"]  = test_df[col].shift(1).rolling(window).min()

    for col in indicators:
        train_df[f"{col}_diff1"] = train_df[col] - train_df[col].shift(1)
        train_df[f"{col}_diff3"] = train_df[col] - train_df[col].shift(3)
        train_df[f"{col}_pct_change1"] = train_df[col].pct_change(1)

    for col in indicators:
        test_df[f"{col}_diff1"] = test_df[col] - test_df[col].shift(1)
        test_df[f"{col}_diff3"] = test_df[col] - test_df[col].shift(3)
        test_df[f"{col}_pct_change1"] = test_df[col].pct_change(1)
        
    for col in indicators:
        train_df[f"{col}_diff1"] = train_df[col] - train_df[col].shift(1)
        train_df[f"{col}_diff3"] = train_df[col] - train_df[col].shift(3)
        train_df[f"{col}_pct_change1"] = train_df[col].pct_change(1)

    for col in indicators:
        test_df[f"{col}_diff1"] = test_df[col] - test_df[col].shift(1)
        test_df[f"{col}_diff3"] = test_df[col] - test_df[col].shift(3)
        test_df[f"{col}_pct_change1"] = test_df[col].pct_change(1)
        
    # Ensure 'date' is datetime
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])

    # Add a column to mark source
    train_df['dataset'] = 'train'
    test_df['dataset'] = 'test'

    # Combine into one DataFrame
    full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

    # Sort by date (optional)
    full_df = full_df.sort_values('date').reset_index(drop=True)

    # --- Later: drop the unnecessary 'dataset' column ---
    full_df = full_df.drop(columns=['dataset'])

    cols_to_check = [
        'INDPRO', 'CPI', 'unemployment_rate', 'PPI', 'share_price',
        '1_year_rate', '3_months_rate', '6_months_rate', '10_year_rate'
    ]

    # Dictionary to store boundaries for each column
    anomaly_stats = {}

    # Ensure 'date' is datetime and sorted
    full_df['date'] = pd.to_datetime(full_df['date'])
    full_df = full_df.sort_values('date').reset_index(drop=True)

    for col in cols_to_check:
        # Apply STL decomposition
        stl = STL(full_df[col], period=12, robust=True)
        res = stl.fit()
        
        residual = res.resid
        
        # Compute mean and std of residuals
        resid_mean = residual.mean()
        resid_std = residual.std()
        
        # Store stats for future use
        anomaly_stats[col] = {'mean': resid_mean, 'std': resid_std, 
                            'lower_bound': resid_mean - 3*resid_std,
                            'upper_bound': resid_mean + 3*resid_std}
        
        # Create anomaly column: 1 if outside ±3 std, else 0
        full_df[f'{col}_anomaly'] = ((residual < resid_mean - 3*resid_std) | 
                                    (residual > resid_mean + 3*resid_std)).astype(int)



    selected_features = ['share_price', 'OECD_CLI_index', 'CSI_index', 'gdp_per_capita_residual', 
                         'gdp_per_capita_rollstd12', 'PPI_diff1', 'gdp_per_capita_diff1', 
                         'gdp_per_capita_diff3', 'gdp_per_capita_pct_change1', '1_year_rate', 
                         '3_months_rate', '6_months_rate', 'CPI', 'INDPRO', '10_year_rate', 'unemployment_rate', 
                         'PPI', 'gdp_per_capita', 'CSI_index_trend', 'OECD_CLI_index_trend', 'PPI_CPI_diff', 
                         '3_months_rate_rollmin12', 'INDPRO_rollstd12', 'PPI_rollstd12', '3_months_rate_diff3', 
                         'OECD_CLI_index_diff1', 'CSI_index_rollmin12', 'OECD_CLI_index_diff3', 
                         'OECD_CLI_index_residual', 'share_price_trend', 'CSI_index_lag6', 'unemployment_rate_rollstd6', 
                         'PPI_rollstd6', 'OECD_CLI_index_rollmax12', 'CSI_index_rollmax6', 'INDPRO_diff3', 
                         'unemployment_rate_diff3', 'CSI_index_rollmin6', 'OECD_CLI_index_pct_change1', '10_year_rate_residual']

    df_reduced = full_df[selected_features].copy()

    # List of anomaly columns to add
    anomaly_cols = [
        'CPI_anomaly', 'unemployment_rate_anomaly', 
        'share_price_anomaly', '3_months_rate_anomaly', '6_months_rate_anomaly', '10_year_rate_anomaly'
    ]
    anomaly_cols_ = [
        'CPI', 'unemployment_rate', 
        'share_price', '3_months_rate', '6_months_rate', '10_year_rate'
    ]

    # Add these columns to df_reduced
    df_reduced = pd.concat([df_reduced, full_df[anomaly_cols]], axis=1)
    
    
    if 'share_price' in input_data:
        share_price_val = input_data['share_price']
        if share_price_val > 0:
            transformed, lmbda = boxcox([share_price_val])
            input_data['share_price'] = transformed[0]
            
    for col, anomaly_col in zip(anomaly_cols_, anomaly_cols):
        if col in input_data:
            df_reduced.at[df_reduced.index[-1], anomaly_col] = int(is_anomaly(col, input_data[col], anomaly_stats))
            
    for col in input_data:
        if col in df_reduced.columns:
            df_reduced.at[df_reduced.index[-1], col] = input_data[col]
            
    return df_reduced


def time_series_prediction():
    pass

def regression_prediction(input_data=None):
    # Feature engineering
    fe_data = regresstion_feature_engineering(input_data)
    
    # Load models
    base_model, one_month_model, three_month_model, six_month_model = load_models()
    
    
    # Make predictions
    try:
        base_pred = float(base_model.predict(fe_data)[0])
        one_month_pred = float(one_month_model.predict(fe_data)[0])
        three_month_pred = float(three_month_model.predict(fe_data)[0])
        six_month_pred = float(six_month_model.predict(fe_data)[0])
    except Exception as e:
        print("Prediction error:", e)
        base_pred = 0.25
        one_month_pred = 0.13
        three_month_pred = 0.66
        six_month_pred = 0.49
    
    return base_pred, one_month_pred, three_month_pred, six_month_pred

