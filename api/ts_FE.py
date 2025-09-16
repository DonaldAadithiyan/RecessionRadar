import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf
import os

import warnings
warnings.filterwarnings('ignore')


# 1️⃣ Define the split function
def prepare_data_split(df, date_col='date', split_year=2020):
    # Ensure the date column is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])

    # Sort by date
    df_sorted = df.sort_values(date_col).reset_index(drop=True)

    # Split
    train_data = df_sorted[df_sorted[date_col].dt.year < split_year].copy()
    test_data = df_sorted[df_sorted[date_col].dt.year >= split_year].copy()

    return train_data, test_data

# 2️⃣ Load your CSV
df = pd.read_csv("data/combined/recession_probability.csv")

# 3️⃣ Call the function
train_df, test_df = prepare_data_split(df, date_col="date", split_year=2020)


train_df.to_csv("data/fix/recession_train.csv", index=False)
test_df.to_csv("data/fix/recession_test.csv", index=False)

def safe_feature_engineering_pipeline(train_df, test_df):
    """
    Safe feature engineering pipeline that prevents data leakage.
    Only uses training data to compute all statistics and parameters.
    """
    
    print("=" * 60)
    print("SAFE FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    # Make copies to avoid modifying originals
    train_fe = train_df.copy()
    test_fe = test_df.copy()
    
    # Ensure date columns are datetime
    train_fe['date'] = pd.to_datetime(train_fe['date'])
    test_fe['date'] = pd.to_datetime(test_fe['date'])
    
    # Sort by date to ensure proper time series order
    train_fe = train_fe.sort_values('date').reset_index(drop=True)
    test_fe = test_fe.sort_values('date').reset_index(drop=True)
    
    print(f"Training data: {train_fe['date'].min()} to {train_fe['date'].max()}")
    print(f"Test data: {test_fe['date'].min()} to {test_fe['date'].max()}")
    
    # Define feature columns
    financial_indicators = [
        '1_year_rate', '3_months_rate', '6_months_rate', 'CPI', 'INDPRO', 
        '10_year_rate', 'share_price', 'unemployment_rate', 'PPI', 
        'OECD_CLI_index', 'CSI_index', 'gdp_per_capita'
    ]
    
    anomaly_columns = [
        'INDPRO', 'CPI', 'unemployment_rate', 'PPI', 'share_price',
        '1_year_rate', '3_months_rate', '6_months_rate', '10_year_rate'
    ]
    
    acf_columns = financial_indicators  # Same as financial_indicators
    
    print(f"\nProcessing {len(financial_indicators)} financial indicators...")
    print(f"Anomaly detection for {len(anomaly_columns)} columns...")
    print(f"ACF features for {len(acf_columns)} columns...")
    
    # ==========================================
    # 1. STL DECOMPOSITION (TRAINING DATA ONLY)
    # ==========================================
    print("\n1. STL Decomposition (Training Data Only)...")
    
    stl_params = {}  # Store parameters computed from training
    
    for col in financial_indicators:
        print(f"   Processing {col}...")
        
        if col not in train_fe.columns:
            print(f"   WARNING: {col} not found in training data, skipping...")
            continue
            
        # Clean training data
        train_series = train_fe[col].fillna(method='ffill').fillna(method='bfill')
        
        if train_series.isna().all() or len(train_series.dropna()) < 24:
            print(f"   WARNING: Insufficient data for {col}, skipping STL...")
            continue
            
        try:
            # STL decomposition on training data
            stl = STL(train_series, seasonal=13, period=12)  # Monthly seasonality
            decomposition = stl.fit()
            
            # Extract components for training
            train_fe[f'{col}_trend'] = decomposition.trend
            train_fe[f'{col}_seasonal'] = decomposition.seasonal  
            train_fe[f'{col}_residual'] = decomposition.resid
            
            # Store parameters for test application
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
            print(f"   ERROR in STL for {col}: {str(e)}")
            continue
    
    # Apply STL-derived features to test data (using training parameters)
    print("\n   Applying STL features to test data...")
    
    for col in financial_indicators:
        if col not in stl_params or col not in test_fe.columns:
            continue
            
        try:
            # For test data, we can't do full STL decomposition as it would use future data
            # Instead, we estimate components using training patterns
            
            test_series = test_fe[col].fillna(method='ffill').fillna(method='bfill')
            
            # Trend: Simple extrapolation from last training trend
            last_trend = stl_params[col]['last_trend']
            test_fe[f'{col}_trend'] = last_trend  # Constant trend assumption
            
            # Seasonal: Repeat the seasonal pattern from training
            seasonal_pattern = stl_params[col]['seasonal_pattern']
            n_test = len(test_fe)
            seasonal_test = np.tile(seasonal_pattern, (n_test // 12) + 1)[:n_test]
            test_fe[f'{col}_seasonal'] = seasonal_test
            
            # Residual: Estimate as deviation from trend + seasonal
            expected = test_fe[f'{col}_trend'] + test_fe[f'{col}_seasonal']
            test_fe[f'{col}_residual'] = test_series - expected
            
        except Exception as e:
            print(f"   ERROR applying STL to test for {col}: {str(e)}")
            continue
    
    # ==========================================
    # 2. ANOMALY DETECTION (TRAINING DATA ONLY)
    # ==========================================
    print("\n2. Anomaly Detection (Training Data Only)...")
    
    anomaly_params = {}
    
    for col in anomaly_columns:
        if f'{col}_residual' not in train_fe.columns:
            print(f"   WARNING: No residual for {col}, skipping anomaly detection...")
            continue
            
        print(f"   Processing anomalies for {col}...")
        
        # Compute anomaly thresholds from training residuals only
        residuals = train_fe[f'{col}_residual'].dropna()
        
        if len(residuals) < 10:
            continue
            
        mean_resid = residuals.mean()
        std_resid = residuals.std()
        
        if std_resid == 0:
            std_resid = 1e-6  # Avoid division by zero
            
        # Store parameters
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
        
        anomaly_count_train = train_fe[f'{col}_anomaly'].sum()
        anomaly_count_test = test_fe[f'{col}_anomaly'].sum() if f'{col}_anomaly' in test_fe.columns else 0
        
        print(f"   {col}: {anomaly_count_train} anomalies in training, {anomaly_count_test} in test")
    
    # ==========================================
    # 3. ACF FEATURES (TRAINING DATA ONLY)
    # ==========================================
    print("\n3. ACF Features (Training Data Only)...")
    
    acf_params = {}
    
    def compute_acf_features(series, max_lags=10):
        """Compute all ACF-based features for a series"""
        if len(series.dropna()) < max_lags + 5:
            return None
            
        try:
            clean_series = series.fillna(method='ffill').fillna(method='bfill').dropna()
            
            if len(clean_series) < max_lags + 5:
                return None
            
            # Original series ACF
            acf_original = acf(clean_series, nlags=max_lags, fft=True)
            first_acf_original = acf_original[1] if len(acf_original) > 1 else 0
            sumsq_acf_original = np.sum(acf_original[1:] ** 2) if len(acf_original) > 1 else 0
            
            # First differenced series
            diff1_series = clean_series.diff().dropna()
            if len(diff1_series) >= max_lags + 2:
                acf_diff1 = acf(diff1_series, nlags=max_lags, fft=True)
                first_acf_diff1 = acf_diff1[1] if len(acf_diff1) > 1 else 0
                sumsq_acf_diff1 = np.sum(acf_diff1[1:] ** 2) if len(acf_diff1) > 1 else 0
            else:
                first_acf_diff1 = 0
                sumsq_acf_diff1 = 0
            
            # Second differenced series  
            diff2_series = diff1_series.diff().dropna()
            if len(diff2_series) >= max_lags + 2:
                acf_diff2 = acf(diff2_series, nlags=max_lags, fft=True)
                first_acf_diff2 = acf_diff2[1] if len(acf_diff2) > 1 else 0
                sumsq_acf_diff2 = np.sum(acf_diff2[1:] ** 2) if len(acf_diff2) > 1 else 0
            else:
                first_acf_diff2 = 0
                sumsq_acf_diff2 = 0
                
            # Seasonal ACF (lag 12 for monthly data)
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
            print(f"   ERROR computing ACF: {str(e)}")
            return None
    
    # Compute ACF features for each column using TRAINING DATA ONLY
    for col in acf_columns:
        if col not in train_fe.columns:
            continue
            
        print(f"   Computing ACF features for {col}...")
        
        # Compute ACF features from training data
        train_series = train_fe[col]
        acf_features = compute_acf_features(train_series)
        
        if acf_features is None:
            print(f"   WARNING: Could not compute ACF features for {col}")
            continue
            
        # Store parameters for test application
        acf_params[col] = acf_features
        
        # Add ACF features to training data
        for feature_name, feature_value in acf_features.items():
            train_fe[f'{col}_{feature_name}'] = feature_value
        
        # For test data: Use the SAME values computed from training
        # This assumes ACF properties are stationary
        for feature_name, feature_value in acf_features.items():
            test_fe[f'{col}_{feature_name}'] = feature_value
    
    # ==========================================
    # 4. SUMMARY AND VALIDATION
    # ==========================================
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    
    # Count new features
    original_train_cols = len(train_df.columns)
    original_test_cols = len(test_df.columns)
    new_train_cols = len(train_fe.columns)
    new_test_cols = len(test_fe.columns)
    
    print(f"Training data: {original_train_cols} → {new_train_cols} columns (+{new_train_cols - original_train_cols})")
    print(f"Test data: {original_test_cols} → {new_test_cols} columns (+{new_test_cols - original_test_cols})")
    
    # Feature breakdown
    stl_features = len([col for col in train_fe.columns if any(suffix in col for suffix in ['_trend', '_seasonal', '_residual'])])
    anomaly_features = len([col for col in train_fe.columns if '_anomaly' in col])
    acf_features = len([col for col in train_fe.columns if any(acf_type in col for acf_type in ['_first_acf', '_sumsq_acf', '_seasonal_acf'])])
    
    print(f"\nFeature breakdown:")
    print(f"  STL features: {stl_features}")
    print(f"  Anomaly features: {anomaly_features}")  
    print(f"  ACF features: {acf_features}")
    
    # Data leakage check
    print(f"\nDATA LEAKAGE PREVENTION:")
    print(f"  ✓ STL parameters computed from training data only")
    print(f"  ✓ Anomaly thresholds computed from training residuals only")
    print(f"  ✓ ACF features computed from training data only")
    print(f"  ✓ Test features derived using training parameters")
    
    # Show sample of new features
    new_feature_cols = [col for col in train_fe.columns if col not in train_df.columns]
    if new_feature_cols:
        print(f"\nSample new features:")
        for i, col in enumerate(new_feature_cols[:10]):
            print(f"  {i+1}. {col}")
        if len(new_feature_cols) > 10:
            print(f"  ... and {len(new_feature_cols) - 10} more")
    
    # Check for missing values
    train_missing = train_fe.isnull().sum().sum()
    test_missing = test_fe.isnull().sum().sum()
    print(f"\nMissing values:")
    print(f"  Training: {train_missing}")
    print(f"  Test: {test_missing}")
    
    print(f"\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    
    return train_fe, test_fe, {
        'stl_params': stl_params,
        'anomaly_params': anomaly_params, 
        'acf_params': acf_params
    }
    
# Apply safe feature engineering
train_engineered, test_engineered, params = safe_feature_engineering_pipeline(train_df, test_df)


def add_fourier_features(train_df, test_df, date_col='date'):
    """
    Add Fourier series features for seasonal patterns to both training and test data.
    Prevents data leakage by using training data start date as reference for both datasets.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataset with date column
    test_df : pd.DataFrame  
        Test dataset with date column
    date_col : str
        Name of the date column (default: 'date')
        
    Returns:
    --------
    tuple: (train_with_fourier, test_with_fourier, fourier_params)
        - train_with_fourier: Training data with added Fourier features
        - test_with_fourier: Test data with added Fourier features  
        - fourier_params: Dictionary containing Fourier parameters for reference
    """
    
    print("=" * 60)
    print("FOURIER SERIES FEATURE ENGINEERING")
    print("=" * 60)
    
    # Make copies to avoid modifying originals
    train_fourier = train_df.copy()
    test_fourier = test_df.copy()
    
    # Ensure date columns are datetime
    train_fourier[date_col] = pd.to_datetime(train_fourier[date_col])
    test_fourier[date_col] = pd.to_datetime(test_fourier[date_col])
    
    # Get reference date from training data (CRITICAL for avoiding data leakage)
    train_start_date = train_fourier[date_col].min()
    
    print(f"Training period: {train_fourier[date_col].min()} to {train_fourier[date_col].max()}")
    print(f"Test period: {test_fourier[date_col].min()} to {test_fourier[date_col].max()}")
    print(f"Using training start date as reference: {train_start_date}")
    
    # Initialize fourier parameters dictionary
    fourier_params = {
        'reference_date': train_start_date,
        'features_added': []
    }
    
    def add_fourier_terms(df, reference_date, K, seasonal_period, prefix):
        """
        Add K pairs of Fourier terms (sin/cos) to dataframe
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe to add features to
        reference_date : pd.Timestamp
            Reference date (t=0 point)
        K : int
            Number of Fourier pairs to generate
        seasonal_period : float
            Seasonal period in the same units as time calculation
        prefix : str
            Prefix for feature names
        """
        
        # Calculate time index relative to reference date
        # Convert to units of seasonal period
        time_numeric = (df[date_col] - reference_date).dt.days / (365.25 / seasonal_period)
        
        features_added = []
        
        # Generate K pairs of Fourier terms
        for k in range(1, K + 1):
            # Frequency for this harmonic
            freq = 2 * np.pi * k / seasonal_period
            
            # Feature names
            sin_name = f'{prefix}_sin_{k}'
            cos_name = f'{prefix}_cos_{k}'
            
            # Generate sin and cos features
            df[sin_name] = np.sin(freq * time_numeric)
            df[cos_name] = np.cos(freq * time_numeric)
            
            features_added.extend([sin_name, cos_name])
        
        return df, features_added
    
    # ==========================================
    # 1. MONTHLY/ANNUAL SEASONALITY
    # ==========================================
    print("\n1. Adding Monthly/Annual Fourier Features...")
    
    K_monthly = 6  # 6 pairs = 12 features
    seasonal_period_monthly = 12  # Monthly data, 12 months = 1 year
    
    train_fourier, monthly_features = add_fourier_terms(
        train_fourier, train_start_date, K_monthly, seasonal_period_monthly, 'fourier_monthly'
    )
    
    test_fourier, _ = add_fourier_terms(
        test_fourier, train_start_date, K_monthly, seasonal_period_monthly, 'fourier_monthly'
    )
    
    fourier_params['monthly'] = {
        'K': K_monthly,
        'seasonal_period': seasonal_period_monthly,
        'features': monthly_features
    }
    fourier_params['features_added'].extend(monthly_features)
    
    print(f"   Added {K_monthly} pairs ({len(monthly_features)} features) for monthly seasonality")
    print(f"   Features: {monthly_features[:4]}..." if len(monthly_features) > 4 else f"   Features: {monthly_features}")
    
    # ==========================================
    # 2. QUARTERLY SEASONALITY
    # ==========================================
    print("\n2. Adding Quarterly Fourier Features...")
    
    K_quarterly = 2  # 2 pairs = 4 features
    seasonal_period_quarterly = 4  # 4 quarters per year
    
    train_fourier, quarterly_features = add_fourier_terms(
        train_fourier, train_start_date, K_quarterly, seasonal_period_quarterly, 'fourier_quarterly'
    )
    
    test_fourier, _ = add_fourier_terms(
        test_fourier, train_start_date, K_quarterly, seasonal_period_quarterly, 'fourier_quarterly'
    )
    
    fourier_params['quarterly'] = {
        'K': K_quarterly,
        'seasonal_period': seasonal_period_quarterly,
        'features': quarterly_features
    }
    fourier_params['features_added'].extend(quarterly_features)
    
    print(f"   Added {K_quarterly} pairs ({len(quarterly_features)} features) for quarterly seasonality")
    print(f"   Features: {quarterly_features}")
    
    # ==========================================
    # 3. BUSINESS CYCLE PATTERNS
    # ==========================================
    print("\n3. Adding Business Cycle Fourier Features...")
    
    K_business = 3  # 3 pairs = 6 features
    seasonal_period_business = 48  # 4-year business cycle (48 months)
    
    train_fourier, business_features = add_fourier_terms(
        train_fourier, train_start_date, K_business, seasonal_period_business, 'fourier_business'
    )
    
    test_fourier, _ = add_fourier_terms(
        test_fourier, train_start_date, K_business, seasonal_period_business, 'fourier_business'
    )
    
    fourier_params['business_cycle'] = {
        'K': K_business,
        'seasonal_period': seasonal_period_business,
        'features': business_features
    }
    fourier_params['features_added'].extend(business_features)
    
    print(f"   Added {K_business} pairs ({len(business_features)} features) for business cycle patterns")
    print(f"   Features: {business_features}")
    
    # ==========================================
    # 4. LONG-TERM ECONOMIC CYCLES (OPTIONAL)
    # ==========================================
    print("\n4. Adding Long-term Economic Cycle Features...")
    
    K_longterm = 2  # 2 pairs = 4 features
    seasonal_period_longterm = 120  # 10-year cycles (120 months)
    
    train_fourier, longterm_features = add_fourier_terms(
        train_fourier, train_start_date, K_longterm, seasonal_period_longterm, 'fourier_longterm'
    )
    
    test_fourier, _ = add_fourier_terms(
        test_fourier, train_start_date, K_longterm, seasonal_period_longterm, 'fourier_longterm'
    )
    
    fourier_params['longterm_cycle'] = {
        'K': K_longterm,
        'seasonal_period': seasonal_period_longterm,
        'features': longterm_features
    }
    fourier_params['features_added'].extend(longterm_features)
    
    print(f"   Added {K_longterm} pairs ({len(longterm_features)} features) for long-term cycles")
    print(f"   Features: {longterm_features}")
    
    # ==========================================
    # 5. SUMMARY AND VALIDATION
    # ==========================================
    print("\n" + "=" * 60)
    print("FOURIER FEATURES SUMMARY")
    print("=" * 60)
    
    original_train_cols = len(train_df.columns)
    original_test_cols = len(test_df.columns)
    new_train_cols = len(train_fourier.columns)
    new_test_cols = len(test_fourier.columns)
    
    total_fourier_features = len(fourier_params['features_added'])
    
    print(f"Training data: {original_train_cols} → {new_train_cols} columns (+{new_train_cols - original_train_cols})")
    print(f"Test data: {original_test_cols} → {new_test_cols} columns (+{new_test_cols - original_test_cols})")
    print(f"Total Fourier features added: {total_fourier_features}")
    
    print(f"\nFeature breakdown:")
    print(f"  Monthly/Annual: {K_monthly} pairs ({K_monthly * 2} features)")
    print(f"  Quarterly: {K_quarterly} pairs ({K_quarterly * 2} features)")
    print(f"  Business Cycle: {K_business} pairs ({K_business * 2} features)")
    print(f"  Long-term Cycle: {K_longterm} pairs ({K_longterm * 2} features)")
    
    print(f"\nMathematical basis:")
    print(f"  Monthly: sin/cos(2πkt/12) for k=1,2,...,{K_monthly}")
    print(f"  Quarterly: sin/cos(2πkt/4) for k=1,2,...,{K_quarterly}")
    print(f"  Business: sin/cos(2πkt/48) for k=1,2,...,{K_business}")
    print(f"  Long-term: sin/cos(2πkt/120) for k=1,2,...,{K_longterm}")
    
    print(f"\nDATA LEAKAGE PREVENTION:")
    print(f"  ✓ All features use training start date as reference (t=0)")
    print(f"  ✓ Fourier terms are deterministic functions of time only")
    print(f"  ✓ No test data values used in feature computation")
    print(f"  ✓ Test features are continuous extensions of training features")
    
    # Sample feature values for verification
    print(f"\nSample feature verification:")
    sample_train_features = train_fourier[fourier_params['features_added'][:4]].iloc[0].round(4)
    sample_test_features = test_fourier[fourier_params['features_added'][:4]].iloc[0].round(4)
    
    print(f"  First training sample: {dict(sample_train_features)}")
    print(f"  First test sample: {dict(sample_test_features)}")
    
    # Check for missing values
    train_missing = train_fourier.isnull().sum().sum()
    test_missing = test_fourier.isnull().sum().sum()
    print(f"\nMissing values after Fourier features:")
    print(f"  Training: {train_missing}")
    print(f"  Test: {test_missing}")
    
    print(f"\n" + "=" * 60)
    print("FOURIER FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    
    return train_fourier, test_fourier, fourier_params

# Example usage and testing function
def test_fourier_features():
    """Test the Fourier features function with sample data"""
    
    print("Testing Fourier features function...")
    
    # Create sample data
    np.random.seed(42)
    
    # Training data (2 years)
    train_dates = pd.date_range('2020-01-01', '2021-12-31', freq='M')
    train_data = {
        'date': train_dates,
        'recession_probability': np.random.random(len(train_dates)),
        'CPI': 100 + np.random.randn(len(train_dates)) * 2
    }
    train_df = pd.DataFrame(train_data)
    
    # Test data (6 months)
    test_dates = pd.date_range('2022-01-01', '2022-06-30', freq='M')
    test_data = {
        'date': test_dates,
        'recession_probability': np.random.random(len(test_dates)),
        'CPI': 100 + np.random.randn(len(test_dates)) * 2
    }
    test_df = pd.DataFrame(test_data)
    
    # Apply Fourier features
    train_with_fourier, test_with_fourier, params = add_fourier_features(train_engineered, test_engineered)
    
    print(f"\nTest completed successfully!")
    print(f"Original columns: {list(train_df.columns)}")
    print(f"New columns sample: {list(train_with_fourier.columns)[-8:]}")
    
    return train_with_fourier, test_with_fourier, params
    
train_final, test_final, fourier_params = test_fourier_features()

# Columns that are in train_final but not in test_final
train_only = set(train_final.columns) - set(test_final.columns)

# Columns that are in test_final but not in train_final
test_only = set(test_final.columns) - set(train_final.columns)

print("🔹 Columns only in train_final:")
print(train_only if train_only else "None")

print("\n🔹 Columns only in test_final:")
print(test_only if test_only else "None")

test_final.to_csv("data/fix/feature_engineered_recession_test.csv", index=False)
train_final.to_csv("data/fix/feature_engineered_recession_train.csv", index=False)