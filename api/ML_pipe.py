import pandas as pd
import numpy as np

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf
from scipy.stats import boxcox
from datetime import datetime

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
    model_dir = '../models/' if os.path.exists('../models/') else 'models/'
    
    base_model = load_model(os.path.join(model_dir, 'catboost_recession_chain_model.pkl'))
    one_three_month_model = load_model(os.path.join(model_dir, 'lgbm_recession_chain_model.pkl'))
    six_month_model = load_model(os.path.join(model_dir, 'lgbm_recession_6m_model.pkl'))

    return base_model, one_three_month_model, six_month_model

def load_ts_models():
    models_dict = {}
    
    model_dir = '../models/enhanced_hybrid_ARIMA_models/' if os.path.exists('../models/enhanced_hybrid_ARIMA_models/') else 'models/enhanced_hybrid_ARIMA_models/'
    for file in os.listdir(model_dir):
        if not file.endswith('.pkl'):
            continue
        models_dict[f'{os.path.splitext(file)[0]}_arima'] = load_model(os.path.join(model_dir, file))

    models_dir = '../models/enhanced_hybrid_prophet_models/' if os.path.exists('../models/enhanced_hybrid_prophet_models/') else 'models/enhanced_hybrid_prophet_models/'
    for file in os.listdir(models_dir):
        if not file.endswith('.pkl'):
            continue
        models_dict[f'{os.path.splitext(file)[0]}_prophet'] = load_model(os.path.join(models_dir, file))

    return models_dict

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

def production_forecasting_pipeline(input_data, models_dict, forecast_steps, date_col='date', freq='M'):
    """
    Production forecasting pipeline that matches your training process
    
    Key insight: Your models were trained with specific feature sets:
    - ARIMA models: Use ALL features except recession targets as exogenous variables
    - Prophet models: Use ALL features except recession targets as regressors
    
    Strategy: Use iterative forecasting approach similar to your training
    """
    
    # Recession targets to exclude (from your training code)
    recession_targets = [
        'recession_probability', '1_month_recession_probability',
        '3_month_recession_probability', '6_month_recession_probability'
    ]
    
    # Financial indicators (your target variables)
    financial_indicators = [
        '1_year_rate', '3_months_rate', '6_months_rate', 'CPI', 'INDPRO',
        '10_year_rate', 'share_price', 'unemployment_rate', 'PPI',
        'OECD_CLI_index', 'CSI_index', 'gdp_per_capita'
    ]
    
    # 1. Create future date range
    if date_col in input_data.columns:
        last_date = pd.to_datetime(input_data[date_col].max())
        if freq == 'M':
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                       periods=forecast_steps, freq='M')
        elif freq == 'Q':
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=3), 
                                       periods=forecast_steps, freq='Q')
        else:
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), 
                                       periods=forecast_steps, freq='M')
        
        result_data = pd.DataFrame({date_col: future_dates})
    else:
        result_data = pd.DataFrame(index=range(forecast_steps))
    
    # 2. Separate models by type
    arima_models = {}
    prophet_models = {}
    
    for indicator, model in models_dict.items():
        model_type = str(type(model)).lower()
        if 'arima' in model_type:
            arima_models[indicator] = model
        elif 'prophet' in model_type:
            prophet_models[indicator] = model
        else:
            arima_models[indicator] = model  # Default to ARIMA treatment
    
    
    # 3. Forecast ARIMA models first (they need exogenous variables)
    for indicator, model in arima_models.items():
        try:
            # Prepare exogenous variables for ARIMA (same as training)
            features_to_exclude = [date_col] + recession_targets + [indicator]
            available_features = [c for c in input_data.columns if c not in features_to_exclude]
            
            if len(available_features) == 0:
                # No exogenous variables - simple ARIMA
                predictions = model.forecast(steps=forecast_steps)
                if hasattr(predictions, 'values'):
                    predictions = predictions.values
            else:
                # ARIMA with exogenous variables
                exog_data = input_data[available_features].copy()
                
                # Clean exogenous data (same as your training)
                exog_data = exog_data.fillna(method='ffill').fillna(method='bfill')
                exog_data = exog_data.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
                
                # Remove non-varying columns
                varying_cols = [c for c in exog_data.columns if exog_data[c].nunique() > 1]
                exog_data = exog_data[varying_cols]
                
                if len(varying_cols) == 0:
                    # No varying exogenous variables
                    predictions = model.forecast(steps=forecast_steps)
                else:
                    # Create future exogenous variables (forward fill last values)
                    last_exog_values = exog_data.iloc[-1:].copy()
                    future_exog = pd.concat([last_exog_values] * forecast_steps, ignore_index=True)
                    
                    # Check if model expects more exogenous variables than we have
                    expected_exog_count = getattr(model.model, 'k_exog', 0)
                    
                    if expected_exog_count > 0 and len(varying_cols) != expected_exog_count:
                        if len(varying_cols) < expected_exog_count:
                            # Pad with zeros to match expected shape
                            missing_cols = expected_exog_count - len(varying_cols)
                            for i in range(missing_cols):
                                future_exog[f'missing_exog_{i}'] = 0.0
                        else:
                            # Take only the first expected_exog_count columns
                            future_exog = future_exog.iloc[:, :expected_exog_count]
                    
                    # Make forecast with exogenous variables
                    predictions = model.forecast(steps=forecast_steps, exog=future_exog)
                
                if hasattr(predictions, 'values'):
                    predictions = predictions.values
            
            # Ensure correct length
            if len(predictions) != forecast_steps:
                if len(predictions) > forecast_steps:
                    predictions = predictions[:forecast_steps]
                else:
                    last_val = predictions[-1] if len(predictions) > 0 else 0
                    predictions = list(predictions) + [last_val] * (forecast_steps - len(predictions))
            
            result_data[indicator] = predictions
            
        except Exception as e:
            # Use trend-based fallback
            predictions = trend_based_forecast(input_data, indicator, forecast_steps)
            result_data[indicator] = predictions
    
    # 4. Now forecast Prophet models (they need ALL features as regressors)
    for indicator, model in prophet_models.items():
        try:
            # Create future dataframe for Prophet
            future_df = model.make_future_dataframe(periods=forecast_steps, freq=freq)
            
            # Get regressor names from the model
            regressors = []
            if hasattr(model, 'extra_regressors'):
                regressors = list(model.extra_regressors.keys())
            
            # Prepare regressor values for the entire future_df
            historical_length = len(future_df) - forecast_steps
            
            for regressor in regressors:
                try:
                    if regressor in result_data.columns:
                        # Use ARIMA predictions for this regressor
                        if regressor in input_data.columns:
                            hist_values = input_data[regressor].fillna(method='ffill').fillna(method='bfill').tolist()
                        else:
                            hist_values = [0.0] * historical_length
                        
                        future_values = result_data[regressor].tolist()
                        all_values = hist_values + future_values
                        
                        # Adjust length to match future_df
                        if len(all_values) > len(future_df):
                            all_values = all_values[:len(future_df)]
                        elif len(all_values) < len(future_df):
                            last_val = all_values[-1] if all_values else 0
                            all_values.extend([last_val] * (len(future_df) - len(all_values)))
                        
                        future_df[regressor] = all_values
                        
                    elif regressor in input_data.columns:
                        # Use historical data + forward fill
                        hist_values = input_data[regressor].fillna(method='ffill').fillna(method='bfill').tolist()
                        
                        # Forward fill for future
                        last_val = hist_values[-1] if hist_values else 0
                        future_values = [last_val] * forecast_steps
                        all_values = hist_values + future_values
                        
                        # Adjust length
                        if len(all_values) > len(future_df):
                            all_values = all_values[:len(future_df)]
                        elif len(all_values) < len(future_df):
                            last_val = all_values[-1] if all_values else 0
                            all_values.extend([last_val] * (len(future_df) - len(all_values)))
                        
                        future_df[regressor] = all_values
                        
                    else:
                        # Missing regressor - use zeros
                        future_df[regressor] = [0.0] * len(future_df)
                        
                except Exception as reg_error:
                    future_df[regressor] = [0.0] * len(future_df)
            
            # Make Prophet prediction
            forecast_result = model.predict(future_df)
            predictions = forecast_result['yhat'].tail(forecast_steps).values
            
            result_data[indicator] = predictions
            
        except Exception as e:
            # Use trend-based fallback
            predictions = trend_based_forecast(input_data, indicator, forecast_steps)
            result_data[indicator] = predictions
    
    # 5. Apply STL decomposition (same as your training)
    indicators = list(models_dict.keys())
    for indicator in indicators:
        if indicator in result_data.columns:
            try:
                series = pd.Series(result_data[indicator])
                
                # Combine with historical data for better STL
                if indicator in input_data.columns:
                    historical_series = input_data[indicator].fillna(method='ffill').fillna(method='bfill')
                    combined_series = pd.concat([historical_series, series], ignore_index=True)
                else:
                    combined_series = series
                
                if len(combined_series.dropna()) >= 24:
                    # Use STL with same parameters as your training
                    stl = STL(combined_series, seasonal=13, period=12)
                    decomposition = stl.fit()
                    
                    # Extract forecast portion
                    trend = decomposition.trend.iloc[-forecast_steps:].values
                    residual = decomposition.resid.iloc[-forecast_steps:].values
                    
                    result_data[f'{indicator}_trend'] = trend
                    result_data[f'{indicator}_residual'] = residual
                else:
                    # Short series fallback
                    mean_val = series.mean()
                    result_data[f'{indicator}_trend'] = [mean_val] * forecast_steps
                    result_data[f'{indicator}_residual'] = (series - mean_val).values
                
            except Exception as e:
                # Simple fallback
                result_data[f'{indicator}_trend'] = result_data[indicator].values
                result_data[f'{indicator}_residual'] = [0.0] * forecast_steps
    
    # 6. Extract required features AND financial indicators
    required_features = [
        'CSI_index_trend', '10_year_rate_trend', '3_months_rate_trend', 
        '1_year_rate_trend', 'unemployment_rate_trend', '6_months_rate_trend',
        'PPI_trend', 'CPI_trend', 'gdp_per_capita_trend', 'gdp_per_capita_residual',
        'OECD_CLI_index_trend', 'OECD_CLI_index_residual', '3_months_rate_residual',
        'INDPRO_trend', 'share_price_trend', '6_months_rate_residual',
        '1_year_rate_residual', '10_year_rate_residual'
    ]
    
    # Financial indicators (raw predictions)
    financial_indicators = [
        '1_year_rate', '3_months_rate', '6_months_rate', 'CPI', 'INDPRO',
        '10_year_rate', 'share_price', 'unemployment_rate', 'PPI',
        'OECD_CLI_index', 'CSI_index', 'gdp_per_capita'
    ]
    
    # Combine date + financial indicators + required features
    all_output_columns = [date_col] + financial_indicators + required_features
    
    # Keep available columns from result_data
    final_columns = [col for col in all_output_columns if col in result_data.columns]
    final_data = result_data[final_columns].copy()
    
    # Fill missing financial indicators with 0
    for indicator in financial_indicators:
        if indicator not in final_data.columns:
            final_data[indicator] = [0.0] * forecast_steps
    
    # Fill missing features with 0
    for feature in required_features:
        if feature not in final_data.columns:
            final_data[feature] = [0.0] * forecast_steps
    
    return final_data

def trend_based_forecast(input_data, indicator, forecast_steps):
    """
    Trend-based forecasting fallback (matches your training approach)
    """
    if indicator not in input_data.columns or input_data[indicator].isna().all():
        return [0.0] * forecast_steps
    
    series = input_data[indicator].fillna(method='ffill').fillna(method='bfill').dropna()
    
    if len(series) < 3:
        return [series.iloc[-1] if len(series) > 0 else 0.0] * forecast_steps
    
    # Simple linear trend
    x = np.arange(len(series))
    y = series.values
    
    try:
        slope, intercept = np.polyfit(x, y, 1)
        last_index = len(series) - 1
        
        predictions = []
        for step in range(1, forecast_steps + 1):
            pred = slope * (last_index + step) + intercept
            predictions.append(pred)
        
        return predictions
        
    except:
        # Ultimate fallback - last value
        return [series.iloc[-1]] * forecast_steps


def time_series_feature_eng(df):
    """
    Feature engineering pipeline for time series data.
    Only computes features that are in selected_columns (if provided).
    Returns:
        pd.DataFrame: Feature engineered DataFrame
    """
    
    selected_columns = [
        "date",
        "recession_probability",
        "1_month_recession_probability",
        "3_month_recession_probability",
        "6_month_recession_probability",
        "1_year_rate",
        "3_months_rate",
        "6_months_rate",
        "CPI",
        "INDPRO",
        "10_year_rate",
        "share_price",
        "unemployment_rate",
        "PPI",
        "OECD_CLI_index",
        "CSI_index",
        "gdp_per_capita",
        "CSI_index_trend",
        "10_year_rate_trend",
        "3_months_rate_trend",
        "1_year_rate_trend",
        "unemployment_rate_trend",
        "6_months_rate_trend",
        "PPI_trend",
        "CPI_trend",
        "gdp_per_capita_trend",
        "gdp_per_capita_residual",
        "OECD_CLI_index_trend",
        "OECD_CLI_index_residual",
        "3_months_rate_residual",
        "INDPRO_trend",
        "share_price_trend",
        "6_months_rate_residual",
        "1_year_rate_residual",
        "10_year_rate_residual"
    ]

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    train_fe = df[df['date'].dt.year < 2020].copy()
    test_fe = df[df['date'].dt.year >= 2020].copy()

    # Identify which STL, residual, and trend features are needed
    stl_needed = [col for col in selected_columns if col.endswith('_trend') or col.endswith('_residual') or col.endswith('_seasonal')]
    stl_base = set([col.replace('_trend', '').replace('_residual', '').replace('_seasonal', '') for col in stl_needed])

    # Identify which anomaly features are needed
    anomaly_needed = [col for col in selected_columns if col.endswith('_anomaly')]
    anomaly_base = set([col.replace('_anomaly', '') for col in anomaly_needed])

    # Identify which ACF features are needed
    acf_suffixes = ['first_acf_original', 'sumsq_acf_original', 'first_acf_diff1', 'sumsq_acf_diff1', 'first_acf_diff2', 'sumsq_acf_diff2', 'seasonal_acf']
    acf_needed = [col for col in selected_columns if any(col.endswith('_' + suf) for suf in acf_suffixes)]
    acf_base = set([col[:-(len(suffix)+1)] for col in acf_needed for suffix in acf_suffixes if col.endswith('_' + suffix)])

    # STL decomposition (only for needed columns)
    stl_params = {}
    # for train
    for col in stl_base:
        if col not in df.columns:
            continue
        train_series = train_fe[col].fillna(method='ffill').fillna(method='bfill')
        if train_series.isna().all() or len(train_series.dropna()) < 24:
            continue
        try:
            stl = STL(train_series, seasonal=13, period=12)
            decomposition = stl.fit()
            if f"{col}_trend" in selected_columns:
                train_fe[f'{col}_trend'] = decomposition.trend
            if f"{col}_seasonal" in selected_columns:
                train_fe[f'{col}_seasonal'] = decomposition.seasonal
            if f"{col}_residual" in selected_columns:
                train_fe[f'{col}_residual'] = decomposition.resid
                
            stl_params[col] = {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'last_trend': decomposition.trend.iloc[-1],
                'seasonal_pattern': decomposition.seasonal.iloc[-12:].values,  # Last year pattern
                'residual_mean': decomposition.resid.mean(),
                'residual_std': decomposition.resid.std()
            }
        except Exception as e:
            print(f"STL failed for {col}: {e}")
            continue
    # for test
    for col in stl_base:
        if col not in df.columns:
            continue
        try:
            test_series = test_fe[col].fillna(method='ffill').fillna(method='bfill')
            
            last_trend = stl_params[col]['last_trend']
            if f"{col}_trend" in selected_columns:
                test_fe[f'{col}_trend'] = last_trend
                
            seasonal_pattern = stl_params[col]['seasonal_pattern']
            n_test = len(test_fe)
            seasonal_test = np.tile(seasonal_pattern, (n_test // 12) + 1)[:n_test]
            if f"{col}_seasonal" in selected_columns:
                test_fe[f'{col}_seasonal'] = seasonal_test
                
            expected = last_trend + seasonal_test
            if f"{col}_residual" in selected_columns:
                test_fe[f'{col}_residual'] = test_series - expected
                
        except Exception as e:
            print(f"STL failed for {col}: {e}")
            continue

    # Anomaly detection (3-sigma on residuals, only for needed columns)
    anomaly_params = {}
    # train
    for col in anomaly_base:
        resid_col = f'{col}_residual'
        if resid_col not in df.columns:
            continue
        
        residuals = train_fe[resid_col].dropna()
        if len(residuals) < 10:
            continue
        mean_resid = residuals.mean()
        std_resid = residuals.std() if residuals.std() != 0 else 1e-6
        
        anomaly_params[col] = {
            'mean': mean_resid,
            'std': std_resid,
            'lower_threshold': mean_resid - 3 * std_resid,
            'upper_threshold': mean_resid + 3 * std_resid
        }
        
        # Apply to training data
        train_fe[f'{col}_anomaly'] = (
            (train_fe[f'{col}_residual'] < anomaly_params[col]['lower_threshold']) |
            (train_fe[f'{col}_residual'] > anomaly_params[col]['upper_threshold'])
        ).astype(int)
        
        # Apply to test data using training thresholds
        if f'{col}_residual' in test_fe.columns:
            test_fe[f'{col}_anomaly'] = (
                (test_fe[f'{col}_residual'] < anomaly_params[col]['lower_threshold']) |
                (test_fe[f'{col}_residual'] > anomaly_params[col]['upper_threshold'])
            ).astype(int)
            
    # ACF features (only for needed columns)
    acf_params = {}
    def compute_acf_features(series, max_lags=10):
        if len(series.dropna()) < max_lags + 5:
            return None
        
        try:
            clean_series = series.fillna(method='ffill').fillna(method='bfill').dropna()
            if len(clean_series) < max_lags + 5:
                return None
            acf_original = acf(clean_series, nlags=max_lags, fft=True)
            first_acf_original = acf_original[1] if len(acf_original) > 1 else 0
            sumsq_acf_original = np.sum(acf_original[1:] ** 2) if len(acf_original) > 1 else 0
            diff1_series = clean_series.diff().dropna()
            if len(diff1_series) >= max_lags + 2:
                acf_diff1 = acf(diff1_series, nlags=max_lags, fft=True)
                first_acf_diff1 = acf_diff1[1] if len(acf_diff1) > 1 else 0
                sumsq_acf_diff1 = np.sum(acf_diff1[1:] ** 2) if len(acf_diff1) > 1 else 0
            else:
                first_acf_diff1 = 0
                sumsq_acf_diff1 = 0
            diff2_series = diff1_series.diff().dropna()
            if len(diff2_series) >= max_lags + 2:
                acf_diff2 = acf(diff2_series, nlags=max_lags, fft=True)
                first_acf_diff2 = acf_diff2[1] if len(acf_diff2) > 1 else 0
                sumsq_acf_diff2 = np.sum(acf_diff2[1:] ** 2) if len(acf_diff2) > 1 else 0
            else:
                first_acf_diff2 = 0
                sumsq_acf_diff2 = 0
            seasonal_lag = min(12, len(acf_original) - 1)
            seasonal_acf = acf_original[seasonal_lag] if seasonal_lag > 0 else 0
            return {
                'first_acf_original': first_acf_original,
                'sumsq_acf_original': sumsq_acf_original,
                'first_acf_diff1': first_acf_diff1,
                'sumsq_acf_diff1': sumsq_acf_diff1,
                'first_acf_diff2': first_acf_diff2,
                'sumsq_acf_diff2': sumsq_acf_diff2,
                'seasonal_acf': seasonal_acf
            }
        except Exception as e:
            print(f"ACF failed: {e}")
            return None

    for col in acf_base:
        if col not in df.columns:
            continue
        train_series = train_fe[col]
        acf_features = compute_acf_features(train_series)
        if acf_features is None:
            continue
        acf_params[col] = acf_features
        for feat_name, feat_val in acf_features.items():
            full_feat_name = f'{col}_{feat_name}'
            if full_feat_name in selected_columns:
                train_fe[full_feat_name] = feat_val
                
        for feat_name, feat_val in acf_features.items():
            full_feat_name = f'{col}_{feat_name}'
            if full_feat_name in selected_columns:
                test_fe[full_feat_name] = feat_val

    # Only keep selected columns (plus 'date' if not already included)
    full_df = pd.concat([train_fe, test_fe], axis=0).reset_index(drop=True)
    full_df = full_df.sort_values('date').reset_index(drop=True)

    final_df = full_df[selected_columns].copy()
    return final_df


def regresstion_feature_engineering(ts_fe_data, ts_prediction):
    full_df = pd.concat([ts_fe_data, ts_prediction], axis=0).reset_index(drop=True)
    print(f"Feature Engineering Input Shape: {full_df.shape}")
    # Ensure 'date' is datetime
    full_df['date'] = pd.to_datetime(full_df['date'])
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
    
    # Interaction features
    if 'CPI_unemployment_interaction' in selected_features:
        full_df['CPI_unemployment_interaction'] = full_df['CPI'] * full_df['unemployment_rate']
    if 'INDPRO_CPI_ratio' in selected_features:
        full_df['INDPRO_CPI_ratio'] = full_df['INDPRO'] / (full_df['CPI'] + 1e-6)
    if 'share_gdp_ratio' in selected_features:
        full_df['share_gdp_ratio'] = full_df['share_price'] / (full_df['gdp_per_capita'] + 1e-6)
    if 'PPI_CPI_diff' in selected_features:
        full_df['PPI_CPI_diff'] = full_df['PPI'] - full_df['CPI']
    if 'interest_spread' in selected_features:
        full_df['interest_spread'] = full_df['10_year_rate'] - full_df['3_months_rate']
    
    for col in indicators:
        for lag in lags:
            if f"{col}_lag{lag}" in selected_features:
                full_df[f"{col}_lag{lag}"] = full_df[col].shift(lag)
            
        for window in windows:
            if f"{col}_rollmean{window}" in selected_features:
                full_df[f"{col}_rollmean{window}"] = full_df[col].shift(1).rolling(window).mean()
            if f"{col}_rollstd{window}" in selected_features:
                full_df[f"{col}_rollstd{window}"]  = full_df[col].shift(1).rolling(window).std()
            if f"{col}_rollmax{window}" in selected_features:
                full_df[f"{col}_rollmax{window}"]  = full_df[col].shift(1).rolling(window).max()
            if f"{col}_rollmin{window}" in selected_features:
                full_df[f"{col}_rollmin{window}"]  = full_df[col].shift(1).rolling(window).min()
        
        if f"{col}_diff1" in selected_features:
            full_df[f"{col}_diff1"] = full_df[col] - full_df[col].shift(1)
        if f"{col}_diff3" in selected_features:
            full_df[f"{col}_diff3"] = full_df[col] - full_df[col].shift(3)
        if f"{col}_pct_change1" in selected_features:
            full_df[f"{col}_pct_change1"] = full_df[col].pct_change(1)
        
    df_reduced = full_df[selected_features + ["date"]].copy()
    
    for col in anomaly_cols:
        df_reduced[f"{col}_anomaly"] = full_df[col].apply(lambda x: is_anomaly(col, x, anomaly_stats)).astype(int)
                

    return df_reduced.iloc[[-1]]


def time_series_prediction(input_data):
    models = load_ts_models()
    
    models_dict = {
        'CSI_index': models['CSI_index_hybrid_prophet'], 
        '10_year_rate': models['10_year_rate_hybrid_prophet'], 
        '3_months_rate': models['3_months_rate_hybrid_arima'], 
        '1_year_rate': models['1_year_rate_hybrid_prophet'],
        'unemployment_rate': models['unemployment_rate_hybrid_arima'], 
        '6_months_rate': models['6_months_rate_hybrid_prophet'], 
        'PPI': models['PPI_hybrid_arima'], 
        'CPI': models['CPI_hybrid_prophet'], 
        'gdp_per_capita': models['gdp_per_capita_hybrid_prophet'], 
        'OECD_CLI_index': models['OECD_CLI_index_hybrid_prophet'], 
        'INDPRO': models['INDPRO_hybrid_arima'], 
        'share_price': models['share_price_hybrid_prophet']
    }
    
    ## calculate how many months to forecast
    last_date = pd.to_datetime(input_data.iloc[-1]['date'])
    now = datetime.now()
    month_diff = (now.year - last_date.year) * 12 + (now.month - last_date.month)
    input_data_ = input_data[input_data['date'].dt.year >= 2020].copy()
    prediction = production_forecasting_pipeline(
        input_data=input_data_, 
        models_dict=models_dict, 
        forecast_steps=month_diff, 
        date_col='date', 
        freq='M'
    )
    
    return prediction


def regression_prediction(fe_data):
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
        
    return base_pred, one_month_pred, three_month_pred, six_month_pred


if __name__ == "__main__":
    time_series_prediction(None)