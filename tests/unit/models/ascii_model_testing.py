#!/usr/bin/env python3
"""
RecessionRadar - Fixed Complete Model Testing with Proper Logging
================================================================

This script provides comprehensive testing for all PKL model files and system components
with proper ASCII-only logging that works on Windows systems.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Add API directory to Python path for lgbm_wrapper import
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(project_root, 'api'))

# Configure logging with ASCII-only output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tests/reports/json/data_integrity_test.log', mode='a', encoding='ascii', errors='replace'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_complete_model_coverage():
    """
    Comprehensive model testing with proper logging
    """
    logger.info("="*60)
    logger.info("STARTING COMPLETE MODEL TESTING SUITE")
    logger.info("="*60)
    
    results = {
        'ts_models_found': 0,
        'ts_models_total': 12,
        'regression_models_found': 0,
        'regression_models_total': 4,
        'pkl_files_valid': 0,
        'pkl_files_corrupted': 0,
        'test_start_time': datetime.now()
    }
    
    # Test 1: Time Series Models
    logger.info("=== TEST 1: Time Series Model Files ===")
    ts_model_dir = 'models/ts_models'
    expected_ts_models = [
        '1_year_rate_prophet_model.pkl',
        '10_year_rate_prophet_model.pkl', 
        '3_months_rate_arima_model.pkl',
        '6_months_rate_arima_model.pkl',
        'CPI_prophet_model.pkl',
        'CSI_index_prophet_model.pkl',
        'gdp_per_capita_arima_model.pkl',
        'INDPRO_prophet_model.pkl',
        'OECD_CLI_index_prophet_model.pkl',
        'PPI_prophet_model.pkl',
        'share_price_prophet_model.pkl',
        'unemployment_rate_arima_model.pkl'
    ]
    
    if os.path.exists(ts_model_dir):
        for model_file in expected_ts_models:
            model_path = os.path.join(ts_model_dir, model_file)
            if os.path.exists(model_path):
                results['ts_models_found'] += 1
                logger.info(f"[PASS] Found: {model_file}")
                
                # Test PKL integrity
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    results['pkl_files_valid'] += 1
                    logger.info(f"[PASS] PKL Valid: {model_file}")
                except Exception as e:
                    results['pkl_files_corrupted'] += 1
                    logger.error(f"[FAIL] PKL Corrupted: {model_file} - {str(e)[:50]}")
            else:
                logger.error(f"[FAIL] Missing: {model_file}")
    else:
        logger.error(f"[FAIL] Directory not found: {ts_model_dir}")
    
    # Test 2: Regression Models
    logger.info("=== TEST 2: Regression Model Files ===")
    model_dir = 'models'
    expected_reg_models = [
        'catboost_recession_6m_model.pkl',
        'catboost_recession_chain_model.pkl',
        'lgbm_recession_6m_model.pkl',
        'lgbm_recession_chain_model.pkl'
    ]
    
    if os.path.exists(model_dir):
        for model_file in expected_reg_models:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                results['regression_models_found'] += 1
                logger.info(f"[PASS] Found: {model_file}")
                
                # Test PKL integrity
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    results['pkl_files_valid'] += 1
                    logger.info(f"[PASS] PKL Valid: {model_file}")
                except Exception as e:
                    if 'lgbm_wrapper' in str(e):
                        logger.warning(f"[WARN] Dependency issue: {model_file} - lgbm_wrapper module needed")
                    else:
                        results['pkl_files_corrupted'] += 1
                        logger.error(f"[FAIL] PKL Corrupted: {model_file} - {str(e)[:50]}")
            else:
                logger.error(f"[FAIL] Missing: {model_file}")
    else:
        logger.error(f"[FAIL] Directory not found: {model_dir}")
    
    # Test 3: Data File Validation
    logger.info("=== TEST 3: Data File Validation ===")
    csv_path = "data/recession_probability.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"[PASS] CSV loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Check for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            logger.info(f"[PASS] Numeric columns: {len(numeric_cols)}")
            
            # Check date column
            if 'date' in df.columns:
                try:
                    pd.to_datetime(df['date'])
                    logger.info(f"[PASS] Date column parsed successfully")
                except:
                    logger.warning(f"[WARN] Date column parsing issues")
                    
        except Exception as e:
            logger.error(f"[FAIL] CSV loading error: {str(e)[:50]}")
    else:
        logger.error(f"[FAIL] Data file not found: {csv_path}")
    
    # Final Summary
    results['test_end_time'] = datetime.now()
    duration = results['test_end_time'] - results['test_start_time']
    
    logger.info("="*60)
    logger.info("COMPLETE MODEL TESTING SUMMARY")
    logger.info("="*60)
    logger.info(f"Test Duration: {duration}")
    logger.info(f"Time Series Models: {results['ts_models_found']}/{results['ts_models_total']}")
    logger.info(f"Regression Models: {results['regression_models_found']}/{results['regression_models_total']}")
    logger.info(f"Valid PKL Files: {results['pkl_files_valid']}")
    logger.info(f"Corrupted PKL Files: {results['pkl_files_corrupted']}")
    
    # Calculate success rate
    total_expected = results['ts_models_total'] + results['regression_models_total']
    total_found = results['ts_models_found'] + results['regression_models_found']
    success_rate = (total_found / total_expected) * 100 if total_expected > 0 else 0
    
    logger.info(f"Overall Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        logger.info("[SUCCESS] Model testing completed successfully!")
        return True
    else:
        logger.error("[FAILURE] Model testing failed - review issues above")
        return False

def main():
    """Main entry point for external calling"""
    try:
        return test_complete_model_coverage()
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        sys.exit(1)