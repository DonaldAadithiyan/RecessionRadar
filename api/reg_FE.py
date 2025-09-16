from statsmodels.tsa.seasonal import STL
import pandas as pd
import numpy as np

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
print("Dropped 'dataset' column. Current shape:", full_df.shape)



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

# Show stats for each column
for col, stats in anomaly_stats.items():
    print(f"{col:20s} | mean: {stats['mean']:.4f} | std: {stats['std']:.4f} | "
          f"lower: {stats['lower_bound']:.4f} | upper: {stats['upper_bound']:.4f}")

import pickle
import os

# Create folder if it doesn't exist
os.makedirs("anomaly_models", exist_ok=True)

# Save dictionary
with open("anomaly_models/anomaly_stats.pkl", "wb") as f:
    pickle.dump(anomaly_stats, f)

print("Anomaly stats dictionary saved to 'anomaly_models/anomaly_stats.pkl'")

with open("anomaly_models/anomaly_stats.pkl", "rb") as f:
    loaded_stats = pickle.load(f)
    
print(loaded_stats)


train_df = full_df[full_df['date'] < '2020-01-01'].reset_index(drop=True)
test_df = full_df[full_df['date'] >= '2020-01-01'].reset_index(drop=True)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

# -----------------------------
# Key indicators to keep
# -----------------------------
key_indicators = [
    '1_year_rate','3_months_rate','6_months_rate','CPI','INDPRO',
    '10_year_rate','share_price','unemployment_rate','PPI',
    'OECD_CLI_index','CSI_index','gdp_per_capita'
]

# -----------------------------
# Targets
# -----------------------------
recession_targets = [
    "recession_probability",
    "1_month_recession_probability",
    "3_month_recession_probability",
    "6_month_recession_probability"
]

X_train = train_df.drop(columns=recession_targets + ["date"])
X_test  = test_df.drop(columns=recession_targets + ["date"])
y_train = train_df[recession_targets]
y_test  = test_df[recession_targets]

# -----------------------------
# Clean data
# -----------------------------
def clean_data(X_or_y):
    X_or_y = X_or_y.replace([np.inf, -np.inf], np.nan)
    X_or_y = X_or_y.ffill().bfill()
    X_or_y = X_or_y.fillna(0)
    return X_or_y

X_train = clean_data(X_train)
X_test = clean_data(X_test)
y_train = clean_data(y_train)
y_test = clean_data(y_test)

# -----------------------------
# Feature reduction per target
# -----------------------------
top_features_dict = {}

for target in recession_targets:
    print(f"\nSelecting features for: {target}")
    
    # Train preliminary Random Forest
    rf = RandomForestRegressor(n_estimators=500, random_state=42)
    rf.fit(X_train, y_train[target])
    
    # Use SelectFromModel to pick important features automatically
    selector = SelectFromModel(rf, prefit=True, max_features=30, threshold=-np.inf)  # keep top 40
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    # Ensure key indicators are included
    for f in key_indicators:
        if f not in selected_features and f in X_train.columns:
            selected_features.append(f)
    
    top_features_dict[target] = selected_features
    print(f"Selected {len(selected_features)} features for {target}")

# -----------------------------
# Reduced datasets
# -----------------------------
X_train_reduced = {}
X_test_reduced = {}

for target in recession_targets:
    features = top_features_dict[target]
    X_train_reduced[target] = X_train[features].copy()
    X_test_reduced[target]  = X_test[features].copy()


# -----------------------------
# Combine all selected features across targets
# -----------------------------
all_selected_features = set()
for target, features in top_features_dict.items():
    all_selected_features.update(features)

# Convert to list
all_selected_features = list(all_selected_features)

# Optional: if too many features (>40), keep only top 40 most frequent features across targets
if len(all_selected_features) > 40:
    # Count frequency across targets
    feature_freq = {}
    for features in top_features_dict.values():
        for f in features:
            feature_freq[f] = feature_freq.get(f, 0) + 1
    # Sort by frequency and pick top 40
    all_selected_features = sorted(feature_freq, key=feature_freq.get, reverse=True)[:40]

print(f"✅ Final selected features ({len(all_selected_features)}):")
print(all_selected_features)

# -----------------------------
# Create reduced dataframe
# -----------------------------
df_reduced = full_df[all_selected_features + recession_targets + ["date"]].copy()

# List of anomaly columns to add
anomaly_cols = [
     'CPI_anomaly', 'unemployment_rate_anomaly', 
     'share_price_anomaly', '3_months_rate_anomaly', '6_months_rate_anomaly', '10_year_rate_anomaly'
]

# Ensure all columns exist in df_3
anomaly_cols = [col for col in anomaly_cols if col in full_df.columns]

# Add these columns to df_reduced
df_reduced = pd.concat([df_reduced, full_df[anomaly_cols]], axis=1)

df_reduced.to_csv('data/fix/feature_selected_reg_full.csv', index=False)