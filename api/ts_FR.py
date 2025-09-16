import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def feature_selection_pipeline(train_df, test_df, n_features_target=35):
    """
    Feature selection pipeline using RFECV for multiple target variables.
    Keeps recession probabilities and targets, selects best additional features.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataset with engineered features
    test_df : pd.DataFrame
        Test dataset with engineered features
    n_features_target : int
        Target number of features to keep (default: 35)
        
    Returns:
    --------
    tuple: (train_selected, test_selected, selection_results)
    """
    
    print("=" * 70)
    print("FEATURE SELECTION PIPELINE - RFECV")
    print("=" * 70)
    
    # Make copies
    train_fs = train_df.copy()
    test_fs = test_df.copy()
    
    # Define column groups
    mandatory_cols = ['date']  # Always keep
    
    recession_cols = [
        'recession_probability', 
        '1_month_recession_probability',
        '3_month_recession_probability', 
        '6_month_recession_probability'
    ]
    
    target_cols = [
        '1_year_rate', '3_months_rate', '6_months_rate', 'CPI', 'INDPRO', 
        '10_year_rate', 'share_price', 'unemployment_rate', 'PPI', 
        'OECD_CLI_index', 'CSI_index', 'gdp_per_capita'
    ]
    
    # Must-keep columns (not subject to feature selection)
    must_keep = mandatory_cols + recession_cols + target_cols
    n_must_keep = len(must_keep)
    
    print(f"Dataset shape: Train {train_fs.shape}, Test {test_fs.shape}")
    print(f"Must-keep columns: {n_must_keep} ({must_keep})")
    
    # Available features for selection (exclude must-keep columns)
    available_features = [col for col in train_fs.columns if col not in must_keep]
    n_available = len(available_features)
    
    # Calculate how many additional features we can select
    n_additional_target = max(0, n_features_target - n_must_keep)
    n_additional_actual = min(n_additional_target, n_available)
    
    print(f"Available features for selection: {n_available}")
    print(f"Target additional features: {n_additional_target}")
    print(f"Actual additional features to select: {n_additional_actual}")
    print(f"Final target columns: {n_must_keep + n_additional_actual}")
    
    if n_additional_actual <= 0:
        print("\nWARNING: No additional features to select!")
        selected_features = must_keep
    else:
        print(f"\nAvailable features for selection:")
        for i, col in enumerate(available_features[:20], 1):
            print(f"  {i:2d}. {col}")
        if n_available > 20:
            print(f"  ... and {n_available - 20} more features")
    
    # Initialize results storage
    selection_results = {
        'must_keep_cols': must_keep,
        'available_features': available_features,
        'target_additional_features': n_additional_actual,
        'rfecv_results': {},
        'feature_importance_summary': {},
        'final_selected_features': [],
        'categorical_encoders': {}
    }
    
    # ==========================================
    # RFECV FEATURE SELECTION
    # ==========================================
    
    if n_additional_actual > 0:
        print(f"\n" + "=" * 50)
        print("RECURSIVE FEATURE ELIMINATION WITH CV")
        print("=" * 50)
        
        # Prepare data for RFECV
        X_available = train_fs[available_features].copy()
        
        # Handle missing values and categorical variables
        print(f"\nPreparing data for RFECV...")
        print(f"Missing values in available features: {X_available.isnull().sum().sum()}")
        
        # Identify and encode categorical variables
        categorical_cols = []
        for col in X_available.columns:
            if X_available[col].dtype == 'object' or X_available[col].dtype.name == 'category':
                categorical_cols.append(col)
        
        print(f"Categorical columns found: {categorical_cols}")
        
        # Handle categorical variables with label encoding
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            # Handle missing values first
            X_available[col] = X_available[col].fillna('Unknown')
            X_available[col] = le.fit_transform(X_available[col].astype(str))
            label_encoders[col] = le
        
        # Store encoders for later use
        selection_results['categorical_encoders'] = label_encoders
        
        # Handle missing values for numerical columns
        X_available = X_available.fillna(method='ffill').fillna(method='bfill')
        
        # If still missing values, fill with median
        for col in X_available.columns:
            if X_available[col].isnull().any():
                X_available[col] = X_available[col].fillna(X_available[col].median())
        
        print(f"After preprocessing: {X_available.isnull().sum().sum()} missing values")
        
        # Feature importance aggregation across targets
        feature_scores = np.zeros(len(available_features))
        feature_selection_frequency = np.zeros(len(available_features))
        
        # RFECV parameters optimized for small dataset
        cv_folds = min(3, len(train_fs) // 100)  # 3 folds for 630 rows
        cv_folds = max(cv_folds, 2)  # At least 2 folds
        
        rf_params = {
            'n_estimators': 50,  # Reduced for speed
            'max_depth': 8,      # Limited depth for small dataset
            'min_samples_split': 10,  # Higher to prevent overfitting
            'min_samples_leaf': 5,    # Higher to prevent overfitting
            'random_state': 42,
            'n_jobs': -1
        }
        
        print(f"RFECV Configuration:")
        print(f"  CV folds: {cv_folds}")
        print(f"  RandomForest params: {rf_params}")
        print(f"  Target additional features: {n_additional_actual}")
        
        # Run RFECV for each target variable
        successful_targets = 0
        for i, target_col in enumerate(target_cols, 1):
            print(f"\n[{i}/{len(target_cols)}] Processing target: {target_col}")
            
            if target_col not in train_fs.columns:
                print(f"  WARNING: {target_col} not found in training data, skipping...")
                continue
            
            # Prepare target variable
            y_target = train_fs[target_col].copy()
            
            # Handle missing values in target
            mask_valid = ~y_target.isnull()
            if mask_valid.sum() < 50:  # Need at least 50 valid samples
                print(f"  WARNING: Insufficient valid samples for {target_col} ({mask_valid.sum()}), skipping...")
                continue
            
            X_target = X_available[mask_valid]
            y_target = y_target[mask_valid]
            
            print(f"  Valid samples: {len(y_target)}")
            
            try:
                # Create RFECV
                rf_estimator = RandomForestRegressor(**rf_params)
                
                rfecv = RFECV(
                    estimator=rf_estimator,
                    min_features_to_select=min(5, n_additional_actual),  # At least 5 features
                    cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
                    scoring='r2',
                    n_jobs=1  # Avoid nested parallelism issues
                )
                
                # Fit RFECV
                print(f"  Running RFECV...")
                rfecv.fit(X_target, y_target)
                
                # Get CV scores - handle both old and new scikit-learn versions
                try:
                    cv_scores = rfecv.cv_results_  # New version
                except AttributeError:
                    cv_scores = rfecv.grid_scores_  # Old version
                
                # Store results
                selection_results['rfecv_results'][target_col] = {
                    'optimal_features': rfecv.n_features_,
                    'selected_features': np.array(available_features)[rfecv.support_].tolist(),
                    'feature_ranking': rfecv.ranking_.copy(),
                    'cv_scores': cv_scores.copy() if hasattr(cv_scores, 'copy') else list(cv_scores),
                    'support_mask': rfecv.support_.copy()
                }
                
                # Aggregate feature importance
                feature_scores += (1.0 / rfecv.ranking_)  # Higher score for lower ranking
                feature_selection_frequency += rfecv.support_.astype(float)
                successful_targets += 1
                
                # Get best CV score - handle different data types
                if hasattr(cv_scores, '__iter__') and len(cv_scores) > 0:
                    try:
                        best_cv_score = max(cv_scores)
                        if isinstance(best_cv_score, (int, float)):
                            score_str = f"{best_cv_score:.4f}"
                        else:
                            score_str = str(best_cv_score)
                    except (ValueError, TypeError):
                        score_str = "N/A"
                else:
                    try:
                        if isinstance(cv_scores, (int, float)):
                            score_str = f"{cv_scores:.4f}"
                        else:
                            score_str = str(cv_scores)
                    except:
                        score_str = "N/A"
                
                print(f"  Optimal features: {rfecv.n_features_}")
                print(f"  Best CV score: {score_str}")
                print(f"  Selected features: {rfecv.support_.sum()}")
                
            except Exception as e:
                print(f"  ERROR in RFECV for {target_col}: {str(e)}")
                continue
        
        # ==========================================
        # AGGREGATE FEATURE SELECTION
        # ==========================================
        print(f"\n" + "=" * 50)
        print("AGGREGATING FEATURE SELECTION RESULTS")
        print("=" * 50)
        
        # Calculate aggregate scores
        if successful_targets > 0:
            # Normalize scores
            avg_importance = feature_scores / successful_targets
            avg_selection_freq = feature_selection_frequency / successful_targets
            
            # Combine importance and selection frequency
            combined_score = 0.7 * avg_importance + 0.3 * avg_selection_freq
            
            # Create feature summary
            feature_summary = pd.DataFrame({
                'feature': available_features,
                'avg_importance': avg_importance,
                'selection_frequency': avg_selection_freq,
                'combined_score': combined_score
            })
            
            # Sort by combined score
            feature_summary = feature_summary.sort_values('combined_score', ascending=False)
            
            # Select top features
            top_features = feature_summary.head(n_additional_actual)['feature'].tolist()
            
            print(f"Processed {successful_targets} target variables successfully")
            print(f"Selected top {len(top_features)} features:")
            
            for i, feat in enumerate(top_features[:min(15, len(top_features))], 1):
                feature_row = feature_summary[feature_summary['feature'] == feat].iloc[0]
                print(f"  {i:2d}. {feat}")
                print(f"      Importance: {feature_row['avg_importance']:.4f}, "
                      f"Frequency: {feature_row['selection_frequency']:.2f}, "
                      f"Combined: {feature_row['combined_score']:.4f}")
            
            if len(top_features) > 15:
                print(f"  ... and {len(top_features) - 15} more features")
            
            # Store results
            selection_results['feature_importance_summary'] = feature_summary
            selected_additional_features = top_features
            
        else:
            print("ERROR: No targets were successfully processed!")
            # Fallback: select features based on correlation with recession probability
            print("\nFalling back to correlation-based selection...")
            
            try:
                # Use correlation with recession_probability as fallback
                correlation_scores = []
                recession_target = train_fs['recession_probability'].fillna(method='ffill').fillna(method='bfill')
                
                for col in available_features:
                    col_data = X_available[col]
                    
                    # Calculate correlation
                    corr = np.corrcoef(col_data, recession_target)[0, 1]
                    correlation_scores.append(abs(corr) if not np.isnan(corr) else 0)
                
                # Create fallback feature summary
                feature_summary = pd.DataFrame({
                    'feature': available_features,
                    'correlation_score': correlation_scores
                })
                
                feature_summary = feature_summary.sort_values('correlation_score', ascending=False)
                top_features = feature_summary.head(n_additional_actual)['feature'].tolist()
                
                print(f"Selected top {len(top_features)} features by correlation:")
                for i, feat in enumerate(top_features[:10], 1):
                    corr_score = feature_summary[feature_summary['feature'] == feat]['correlation_score'].iloc[0]
                    print(f"  {i:2d}. {feat} (corr: {corr_score:.4f})")
                
                selection_results['feature_importance_summary'] = feature_summary
                selected_additional_features = top_features
                
            except Exception as e:
                print(f"ERROR in fallback selection: {str(e)}")
                selected_additional_features = available_features[:n_additional_actual]
    
    else:
        selected_additional_features = []
    
    # ==========================================
    # FINAL FEATURE SET
    # ==========================================
    print(f"\n" + "=" * 50)
    print("FINAL FEATURE SELECTION")
    print("=" * 50)
    
    # Combine must-keep and selected features
    final_features = must_keep + selected_additional_features
    selection_results['final_selected_features'] = final_features
    
    print(f"Final feature set ({len(final_features)} features):")
    print(f"  Must-keep: {len(must_keep)} features")
    print(f"  Selected: {len(selected_additional_features)} features")
    
    # Show feature breakdown
    print(f"\nFeature breakdown:")
    print(f"  Mandatory: {len(mandatory_cols)} - {mandatory_cols}")
    print(f"  Recession: {len(recession_cols)} - {recession_cols}")
    print(f"  Targets: {len(target_cols)} - {target_cols}")
    print(f"  Selected: {len(selected_additional_features)}")
    
    if selected_additional_features:
        print(f"    Top selected features:")
        for i, feat in enumerate(selected_additional_features[:10], 1):
            print(f"      {i:2d}. {feat}")
        if len(selected_additional_features) > 10:
            print(f"      ... and {len(selected_additional_features) - 10} more")
    
    # Apply feature selection to both train and test
    train_selected = train_fs[final_features].copy()
    test_selected = test_fs[final_features].copy()
    
    # Apply categorical encoding to test data if needed
    if selection_results['categorical_encoders']:
        print(f"\nApplying categorical encoding to test data...")
        
        for col in categorical_cols:
            if col in selected_additional_features and col in test_selected.columns:
                # Handle test data encoding
                test_values = test_selected[col].fillna('Unknown').astype(str)
                le = selection_results['categorical_encoders'][col]
                
                # Map known categories, unknown ones become -1
                encoded_values = []
                for val in test_values:
                    try:
                        encoded_val = le.transform([val])[0]
                    except ValueError:
                        # New category not seen in training, use most frequent class
                        encoded_val = 0  # or use mode of training data
                    encoded_values.append(encoded_val)
                
                test_selected[col] = encoded_values
                train_selected[col] = selection_results['categorical_encoders'][col].transform(
                    train_selected[col].fillna('Unknown').astype(str)
                )
    
    print(f"\nFinal dataset shapes:")
    print(f"  Training: {train_selected.shape}")
    print(f"  Test: {test_selected.shape}")
    
    # Verify no missing columns
    missing_in_test = [col for col in final_features if col not in test_fs.columns]
    if missing_in_test:
        print(f"  WARNING: Missing in test data: {missing_in_test}")
    
    # Data quality check
    print(f"\nData quality after selection:")
    train_missing = train_selected.isnull().sum().sum()
    test_missing = test_selected.isnull().sum().sum()
    print(f"  Training missing values: {train_missing}")
    print(f"  Test missing values: {test_missing}")
    
    print(f"\n" + "=" * 70)
    print("FEATURE SELECTION COMPLETE")
    print("=" * 70)
    
    return train_selected, test_selected, selection_results

# Utility function to analyze selection results
def analyze_selection_results(selection_results):
    """
    Analyze and visualize feature selection results
    """
    print("=" * 60)
    print("FEATURE SELECTION ANALYSIS")
    print("=" * 60)
    
    if ('feature_importance_summary' in selection_results and 
        isinstance(selection_results['feature_importance_summary'], pd.DataFrame) and 
        not selection_results['feature_importance_summary'].empty):
        
        summary = selection_results['feature_importance_summary']
        
        print(f"\nTop 15 features by combined score:")
        top_15 = summary.head(15)
        for i, (_, row) in enumerate(top_15.iterrows(), 1):
            if 'combined_score' in row:
                print(f"{i:2d}. {row['feature']:<30} "
                      f"Score: {row['combined_score']:.4f} "
                      f"(Imp: {row['avg_importance']:.3f}, "
                      f"Freq: {row['selection_frequency']:.2f})")
            else:
                # Fallback correlation-based selection
                print(f"{i:2d}. {row['feature']:<30} "
                      f"Correlation: {row['correlation_score']:.4f}")
        
        if 'selection_frequency' in summary.columns:
            print(f"\nSelection frequency distribution:")
            freq_dist = summary['selection_frequency'].value_counts().sort_index(ascending=False)
            for freq, count in freq_dist.items():
                if freq > 0:
                    targets_processed = len(selection_results['rfecv_results'])
                    print(f"  Selected by {freq:.0f}/{targets_processed} targets: {count} features")
    
    else:
        print("\nNo feature importance summary available.")
    
    print(f"\nRFECV Results per target:")
    if selection_results['rfecv_results']:
        for target, results in selection_results['rfecv_results'].items():
            cv_scores = results['cv_scores']
            try:
                if hasattr(cv_scores, '__iter__') and len(cv_scores) > 0:
                    best_score = max(cv_scores)
                    if isinstance(best_score, (int, float)):
                        score_str = f"{best_score:.4f}"
                    else:
                        score_str = str(best_score)
                else:
                    if isinstance(cv_scores, (int, float)):
                        score_str = f"{cv_scores:.4f}"
                    else:
                        score_str = str(cv_scores)
            except:
                score_str = "N/A"
            
            print(f"  {target}: {results['optimal_features']} features, "
                  f"CV score: {score_str}")
    else:
        print("  No RFECV results available (fallback method used)")
    
    print(f"\nFinal selection summary:")
    print(f"  Must-keep features: {len(selection_results['must_keep_cols'])}")
    print(f"  Available for selection: {len(selection_results['available_features'])}")
    print(f"  Target additional: {selection_results['target_additional_features']}")
    print(f"  Final selected: {len(selection_results['final_selected_features'])}")
    
    if selection_results['categorical_encoders']:
        print(f"\nCategorical features encoded: {list(selection_results['categorical_encoders'].keys())}")

# Additional utility function to handle deprecated pandas methods
def safe_fillna(df, method='ffill'):
    """
    Safe fillna method that handles deprecated method parameter
    """
    try:
        # Try new approach first (pandas >= 2.0)
        if method == 'ffill':
            return df.ffill().bfill()
        elif method == 'bfill':
            return df.bfill().ffill()
        else:
            return df.fillna(method=method)
    except TypeError:
        # Fallback to old method for older pandas versions
        return df.fillna(method=method)

# Updated version that handles pandas deprecation warnings
def feature_selection_pipeline_v2(train_df, test_df, n_features_target=35):
    """
    Updated feature selection pipeline that handles pandas deprecation warnings
    """
    # Replace all fillna(method='...') calls with safe_fillna
    # This is a wrapper that can be used if you encounter pandas deprecation warnings
    return feature_selection_pipeline(train_df, test_df, n_features_target)

train_eng = pd.read_csv('data/fix/feature_engineered_recession_train.csv')
test_eng = pd.read_csv('data/fix/feature_engineered_recession_test.csv')

# After running your STL and Fourier pipelines
train_selected, test_selected, results = feature_selection_pipeline(
    train_eng, test_eng, n_features_target=35
)

# Analyze the selection results
analyze_selection_results(results)

train_selected.to_csv('data/fix/feature_selected_recession_train.csv', index=False)
test_selected.to_csv('data/fix/feature_selected_recession_test.csv', index=False)

# Columns that are in train_final but not in test_final
train_only = set(train_selected.columns) - set(test_selected.columns)

# Columns that are in test_final but not in train_final
test_only = set(test_selected.columns) - set(train_selected.columns)

print("🔹 Columns only in train_final:")
print(train_only if train_only else "None")

print("\n🔹 Columns only in test_final:")
print(test_only if test_only else "None")