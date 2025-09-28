import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.multioutput import RegressorChain
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf
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
    model_dir = '../models/' if os.path.exists('../models/') else 'models/'
    
    base_model = load_model(os.path.join(model_dir, 'catboost_recession_chain_model.pkl'))
    one_three_month_model = load_model(os.path.join(model_dir, 'lgbm_recession_chain_model.pkl'))
    six_month_model = load_model(os.path.join(model_dir, 'lgbm_recession_6m_model.pkl'))

    return base_model, one_three_month_model, six_month_model

def load_ts_models():
    models_dict = {}
    model_dir = '../models/ts_models/' if os.path.exists('../models/ts_models/') else 'models/ts_models/'
    
    models_dict['csi_index_prophet_model'] = load_model(os.path.join(model_dir, 'CSI_index_prophet_model.pkl'))
    models_dict['ten_year_rate_prophet_model'] = load_model(os.path.join(model_dir, '10_year_rate_prophet_model.pkl'))
    models_dict['three_months_rate_arima_model'] = load_model(os.path.join(model_dir, '3_months_rate_arima_model.pkl'))
    models_dict['one_year_rate_prophet_model'] = load_model(os.path.join(model_dir, '1_year_rate_prophet_model.pkl'))
    models_dict['unemployment_rate_arima_model'] = load_model(os.path.join(model_dir, 'unemployment_rate_arima_model.pkl'))
    models_dict['six_months_rate_arima_model'] = load_model(os.path.join(model_dir, '6_months_rate_arima_model.pkl'))
    models_dict['ppi_prophet_model'] = load_model(os.path.join(model_dir, 'PPI_prophet_model.pkl'))
    models_dict['cpi_prophet_model'] = load_model(os.path.join(model_dir, 'CPI_prophet_model.pkl'))
    models_dict['gdp_per_capita_arima_model'] = load_model(os.path.join(model_dir, 'gdp_per_capita_arima_model.pkl'))
    models_dict['oecd_cli_index_prophet_model'] = load_model(os.path.join(model_dir, 'OECD_CLI_index_prophet_model.pkl'))
    models_dict['indpro_prophet_model'] = load_model(os.path.join(model_dir, 'INDPRO_prophet_model.pkl'))
    models_dict['share_price_prophet_model'] = load_model(os.path.join(model_dir, 'share_price_prophet_model.pkl'))
    
    
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
    
    print(f"Production Forecasting Pipeline - {forecast_steps} steps ahead")
    print("="*60)
    
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
    
    print(f"ARIMA models: {list(arima_models.keys())}")
    print(f"Prophet models: {list(prophet_models.keys())}")
    
    # 3. Forecast ARIMA models first (they need exogenous variables)
    print(f"\nStep 1: Forecasting ARIMA models...")
    
    for indicator, model in arima_models.items():
        try:
            print(f"  Processing ARIMA: {indicator}...")
            
            # Prepare exogenous variables for ARIMA (same as training)
            features_to_exclude = [date_col] + recession_targets + [indicator]
            available_features = [c for c in input_data.columns if c not in features_to_exclude]
            
            print(f"    Available exog features: {len(available_features)}")
            
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
                
                print(f"    Using {len(varying_cols)} exog variables")
                
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
                        print(f"    Warning: Model expects {expected_exog_count} exog vars, but have {len(varying_cols)}")
                        
                        if len(varying_cols) < expected_exog_count:
                            # Pad with zeros to match expected shape
                            missing_cols = expected_exog_count - len(varying_cols)
                            for i in range(missing_cols):
                                future_exog[f'missing_exog_{i}'] = 0.0
                            print(f"    Padded with {missing_cols} zero columns")
                        else:
                            # Take only the first expected_exog_count columns
                            future_exog = future_exog.iloc[:, :expected_exog_count]
                            print(f"    Truncated to {expected_exog_count} columns")
                    
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
            print(f"    ✓ ARIMA forecast: {indicator} ({len(predictions)} values)")
            
        except Exception as e:
            print(f"    ✗ ARIMA failed for {indicator}: {str(e)}")
            # Use trend-based fallback
            predictions = trend_based_forecast(input_data, indicator, forecast_steps)
            result_data[indicator] = predictions
            print(f"    ✓ Fallback forecast: {indicator}")
    
    # 4. Now forecast Prophet models (they need ALL features as regressors)
    print(f"\nStep 2: Forecasting Prophet models...")
    
    for indicator, model in prophet_models.items():
        try:
            print(f"  Processing Prophet: {indicator}...")
            
            # Create future dataframe for Prophet
            future_df = model.make_future_dataframe(periods=forecast_steps, freq=freq)
            
            # Get regressor names from the model
            regressors = []
            if hasattr(model, 'extra_regressors'):
                regressors = list(model.extra_regressors.keys())
            
            print(f"    Model needs {len(regressors)} regressors")
            print(f"    Sample regressors: {regressors[:5]}..." if len(regressors) > 5 else f"    Regressors: {regressors}")
            
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
                    print(f"      Warning: Error with regressor {regressor}: {str(reg_error)}")
                    future_df[regressor] = [0.0] * len(future_df)
            
            # Make Prophet prediction
            forecast_result = model.predict(future_df)
            predictions = forecast_result['yhat'].tail(forecast_steps).values
            
            result_data[indicator] = predictions
            print(f"    ✓ Prophet forecast: {indicator} ({len(predictions)} values)")
            
        except Exception as e:
            print(f"    ✗ Prophet failed for {indicator}: {str(e)}")
            # Use trend-based fallback
            predictions = trend_based_forecast(input_data, indicator, forecast_steps)
            result_data[indicator] = predictions
            print(f"    ✓ Fallback forecast: {indicator}")
    
    # 5. Apply STL decomposition (same as your training)
    print(f"\nStep 3: Applying STL decomposition...")
    
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
                
                print(f"  ✓ STL: {indicator}")
                
            except Exception as e:
                print(f"  ✗ STL failed for {indicator}: {str(e)}")
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
    
    print(f"\n" + "="*60)
    print(f"PRODUCTION FORECASTING COMPLETE")
    print(f"="*60)
    print(f"Final shape: {final_data.shape}")
    if date_col in final_data.columns:
        print(f"Forecast period: {final_data[date_col].min()} to {final_data[date_col].max()}")
    print(f"Generated {len(financial_indicators)} financial indicators + {len(required_features)} engineered features")
    print(f"Financial indicators: {financial_indicators}")
    print(f"Sample features: {required_features[:5]}...")
    
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


def time_series_feature_eng(selected_columns=None):
    """
    Feature engineering pipeline for time series data.
    Only computes features that are in selected_columns (if provided).
    Returns:
        pd.DataFrame: Feature engineered DataFrame
    """
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import acf
    import os

    # Load data
    if os.path.exists('../data/combined/'):
        df = pd.read_csv('../data/combined/recession_probability.csv')
    else:
        df = pd.read_csv('data/combined/recession_probability.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # If no selected_columns provided, compute all features as before
    if selected_columns is None:
        selected_columns = df.columns.tolist()  # fallback: keep all

    # Identify which STL, residual, and trend features are needed
    stl_needed = [col for col in selected_columns if col.endswith('_trend') or col.endswith('_residual') or col.endswith('_seasonal')]
    stl_base = set([col.replace('_trend', '').replace('_residual', '').replace('_seasonal', '') for col in stl_needed])

    # Identify which anomaly features are needed
    anomaly_needed = [col for col in selected_columns if col.endswith('_anomaly')]
    anomaly_base = set([col.replace('_anomaly', '') for col in anomaly_needed])

    # Identify which ACF features are needed
    acf_suffixes = ['first_acf_original', 'sumsq_acf_original', 'first_acf_diff1', 'sumsq_acf_diff1', 'first_acf_diff2', 'sumsq_acf_diff2', 'seasonal_acf']
    acf_needed = [col for col in selected_columns if any(col.endswith('_' + suf) for suf in acf_suffixes)]
    acf_base = set([col[:-(len(suf)+1)] for col in acf_suffixes for col in acf_needed if col.endswith('_' + suf)])

    # STL decomposition (only for needed columns)
    for col in stl_base:
        if col not in df.columns:
            continue
        series = df[col].fillna(method='ffill').fillna(method='bfill')
        if series.isna().all() or len(series.dropna()) < 24:
            continue
        try:
            stl = STL(series, seasonal=13, period=12)
            decomposition = stl.fit()
            if f"{col}_trend" in selected_columns:
                df[f'{col}_trend'] = decomposition.trend
            if f"{col}_seasonal" in selected_columns:
                df[f'{col}_seasonal'] = decomposition.seasonal
            if f"{col}_residual" in selected_columns:
                df[f'{col}_residual'] = decomposition.resid
        except Exception as e:
            print(f"STL failed for {col}: {e}")

    # Anomaly detection (3-sigma on residuals, only for needed columns)
    for col in anomaly_base:
        resid_col = f'{col}_residual'
        if resid_col not in df.columns:
            continue
        residuals = df[resid_col].dropna()
        if len(residuals) < 10:
            continue
        mean_resid = residuals.mean()
        std_resid = residuals.std() if residuals.std() != 0 else 1e-6
        lower = mean_resid - 3 * std_resid
        upper = mean_resid + 3 * std_resid
        if f"{col}_anomaly" in selected_columns:
            df[f'{col}_anomaly'] = ((df[resid_col] < lower) | (df[resid_col] > upper)).astype(int)

    # ACF features (only for needed columns)
    def compute_acf_features(series, max_lags=10):
        if len(series.dropna()) < max_lags + 5:
            return {}
        try:
            clean_series = series.fillna(method='ffill').fillna(method='bfill').dropna()
            if len(clean_series) < max_lags + 5:
                return {}
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
            return {}

    for col in acf_base:
        if col not in df.columns:
            continue
        acf_feats = compute_acf_features(df[col])
        for feat_name, feat_val in acf_feats.items():
            full_feat_name = f'{col}_{feat_name}'
            if full_feat_name in selected_columns:
                df[full_feat_name] = feat_val

    # Only keep selected columns (plus 'date' if not already included)
    keep_cols = [col for col in selected_columns if col in df.columns]
    if 'date' not in keep_cols and 'date' in df.columns:
        keep_cols = ['date'] + keep_cols
    return df[keep_cols].copy()


def time_series_feature_reduction(input_df=None, dataset_type='test'):
    """
    Loads the feature-selected columns from the output of the feature selection pipeline.
    If input_df is provided, returns only the selected columns.
    Otherwise, loads the reduced test/train set from CSV.
    """
    import pandas as pd
    import os

    # List of selected columns after feature reduction (from your output)
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

    # If input_df is provided, just select the columns
    if input_df is not None:
        # If input_df is raw, run feature engineering only for needed columns
        if not set(selected_columns).issubset(set(input_df.columns)):
            return time_series_feature_eng(selected_columns=selected_columns)
        return input_df[selected_columns].copy()

    # Otherwise, load from CSV
    base_path = '../data/fix/' if os.path.exists('../data/fix/') else 'data/fix/'
    if dataset_type == 'train':
        csv_path = os.path.join(base_path, 'feature_selected_recession_train.csv')
    else:
        csv_path = os.path.join(base_path, 'feature_selected_recession_test.csv')
    df = pd.read_csv(csv_path)
    return df[selected_columns].copy()


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
    models = load_ts_models()
    
    models_dict = {
        'CSI_index': models['csi_index_prophet_model'], 
        '10_year_rate': models['ten_year_rate_prophet_model'], 
        '3_months_rate': models['three_months_rate_arima_model'], 
        '1_year_rate': models['one_year_rate_prophet_model'],
        'unemployment_rate': models['unemployment_rate_arima_model'], 
        '6_months_rate': models['six_months_rate_arima_model'], 
        'PPI': models['ppi_prophet_model'], 
        'CPI': models['cpi_prophet_model'], 
        'gdp_per_capita': models['gdp_per_capita_arima_model'], 
        'OECD_CLI_index': models['oecd_cli_index_prophet_model'], 
        'INDPRO': models['indpro_prophet_model'], 
        'share_price': models['share_price_prophet_model']
    }
    data_dir = '../data' if os.path.exists('../data') else 'data'
    input_data = pd.reaa_cd_csv(os.path.join(data_dir, 'fix', 'feature_selected_recession_test.csv'))
    
    prediction = production_forecasting_pipeline(
        input_data=input_data, 
        models_dict=models_dict, 
        forecast_steps=4, 
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
    
    
    print(f"Base: {base_pred}, 1m: {one_month_pred}, 3m: {three_month_pred}, 6m: {six_month_pred}")
    
    return base_pred/100, one_month_pred/100, three_month_pred/100, six_month_pred/100


if __name__ == "__main__":
    input_ = {'1-Year Rate': 99.54, '3-Month Rate': 3.89, '6-Month Rate': 3.75, '10-Year Rate': 4.18, 'CPI': 393.364, 'PPI': 262.443, 'Industrial Production': 103.9203, 'Share Price': 6643.7, 'Unemployment Rate': 4.3, 'OECD CLI Index': 98.3101894908785, 'CSI Index': 58.2}
    base_pred, one_month_pred, three_month_pred, six_month_pred = regression_prediction(input_)
    # base_pred, one_month_pred, three_month_pred, six_month_pred = regression_prediction()
    print(f"Base Prediction: {base_pred}")
    print(f"1-Month Prediction: {one_month_pred}")
    print(f"3-Month Prediction: {three_month_pred}")
    print(f"6-Month Prediction: {six_month_pred}")