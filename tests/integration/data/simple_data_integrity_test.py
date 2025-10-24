# Simple Data Integrity Test with Proper Logging
import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

def setup_logging():
    """Setup comprehensive logging"""
    log_file = 'tests/reports/json/data_integrity_test.log'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logger = logging.getLogger('DataIntegrityTest')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler - use ASCII encoding to avoid Windows issues
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def run_simple_data_integrity_test():
    """Run basic data integrity tests and return success status"""
    logger = setup_logging()
    
    test_results = {
        'csv_exists': 'NOT_RUN',
        'data_validation': 'NOT_RUN', 
        'imports': 'NOT_RUN',
        'models': 'NOT_RUN',
        'env_vars': 'NOT_RUN'
    }

    # Test 1: CSV File Existence and Readability
    logger.info("[TEST 1] CSV File Existence and Readability")
    try:
        csv_path = "data/recession_probability.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            logger.info(f"[PASS] CSV file found with {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"[PASS] Columns: {list(df.columns)}")
            test_results['csv_exists'] = 'PASSED'
        else:
            logger.warning("[WARNING] recession_probability.csv not found - may be expected for new setup")
            test_results['csv_exists'] = 'SKIPPED'
    except Exception as e:
        logger.error(f"[FAIL] CSV file test failed: {e}")
        test_results['csv_exists'] = 'FAILED'
    
    # Test 2: Basic data validation if CSV exists
    if os.path.exists("data/recession_probability.csv"):
        logger.info("\n[TEST 2] Basic Data Validation")
        try:
            df = pd.read_csv("data/recession_probability.csv")
            
            # Check for reasonable data ranges
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            logger.info(f"[PASS] Found {len(numeric_cols)} numeric columns: {list(numeric_cols)}")
            
            for col in numeric_cols:
                if df[col].notna().any():
                    col_stats = {
                        'min': df[col].min(),
                        'max': df[col].max(), 
                        'mean': df[col].mean(),
                        'null_count': df[col].isnull().sum()
                    }
                    logger.info(f"[PASS] {col}: min={col_stats['min']:.4f}, max={col_stats['max']:.4f}, mean={col_stats['mean']:.4f}, nulls={col_stats['null_count']}")
                    
                    # Check probability columns are in reasonable range
                    if 'probability' in col.lower():
                        if col_stats['min'] >= 0 and col_stats['max'] <= 1:
                            logger.info(f"[PASS] {col}: Probability values in valid range [0,1]")
                        else:
                            logger.warning(f"[WARNING] {col}: Probability values outside [0,1] range")
            
            test_results['data_validation'] = 'PASSED'
        except Exception as e:
            logger.error(f"[FAIL] Data validation failed: {e}")
            test_results['data_validation'] = 'FAILED'
    
    # Test 3: API modules import test
    logger.info("\n[TEST 3] API Module Import Test")
    try:
        from api.data_collection import fetch_latest_fred_value
        logger.info("[PASS] Successfully imported fetch_latest_fred_value")
        
        # from api.lgbm_wrapper import LGBMWrapper
        # logger.info("[PASS] Successfully imported LGBMWrapper")
        
        test_results['imports'] = 'PASSED'
        
    except Exception as e:
        logger.error(f"[FAIL] Import test failed: {e}")
        test_results['imports'] = 'FAILED'
    
    # Test 4: Model files existence
    logger.info("\n[TEST 4] Model Files Existence")
    try:
        model_dir = "models"
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            logger.info(f"[PASS] Found {len(model_files)} model files in {model_dir}/")
            for model_file in model_files:
                logger.info(f"  - {model_file}")
            test_results['models'] = 'PASSED'
        else:
            logger.warning("[WARNING] Models directory not found")
            test_results['models'] = 'SKIPPED'
    except Exception as e:
        logger.error(f"[FAIL] Model files test failed: {e}")
        test_results['models'] = 'FAILED'
    
    # Test 5: Environment variables
    logger.info("\n[TEST 5] Environment Variables")
    try:
        fred_key = os.getenv('FRED_API_KEY')
        if fred_key:
            logger.info(f"[PASS] FRED_API_KEY found (length: {len(fred_key)} characters)")
            test_results['env_vars'] = 'PASSED'
        else:
            logger.warning("[WARNING] FRED_API_KEY not found in environment variables")
            test_results['env_vars'] = 'SKIPPED'
    except Exception as e:
        logger.error(f"[FAIL] Environment variables test failed: {e}")
        test_results['env_vars'] = 'FAILED'
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("[SUMMARY] TEST EXECUTION SUMMARY")
    logger.info("="*60)
    
    total_tests = 5
    passed_tests = sum(1 for result in test_results.values() if result == 'PASSED')
    failed_tests = sum(1 for result in test_results.values() if result == 'FAILED')
    skipped_tests = sum(1 for result in test_results.values() if result == 'SKIPPED')
    
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests} [PASS]")
    logger.info(f"Failed: {failed_tests} [FAIL]")
    logger.info(f"Skipped: {skipped_tests} [WARNING]")
    logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    logger.info("\nDetailed Results:")
    status_mapping = {'PASSED': '[PASS]', 'FAILED': '[FAIL]', 'SKIPPED': '[WARNING]', 'NOT_RUN': '[NOT_RUN]'}
    test_names = ['csv_exists', 'data_validation', 'imports', 'models', 'env_vars']
    
    for test_name in test_names:
        result = test_results[test_name]
        status_icon = status_mapping[result]
        logger.info(f"{status_icon} {test_name.replace('_', ' ').title()}: {result}")
    
    # Success criteria: all tests should either pass or be skipped (no failures)
    success = all(result in ['PASSED', 'SKIPPED'] for result in test_results.values())
    
    # Save detailed JSON report
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': total_tests,
        'passed': passed_tests,
        'failed': failed_tests,
        'skipped': skipped_tests,
        'success_rate': (passed_tests/total_tests)*100,
        'overall_success': success,
        'test_results': test_results
    }
    
    with open('data_integrity_test_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    logger.info(f"\n[REPORT] Detailed report saved to: data_integrity_test_report.json")
    logger.info(f"[REPORT] Full log saved to: tests/reports/json/data_integrity_test.log")
    
    if not success:
        logger.warning("[WARNING] Some tests failed. Please review the detailed results above.")
    
    return success

if __name__ == "__main__":
    success = run_simple_data_integrity_test()
    exit(0 if success else 1)