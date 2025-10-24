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
    
    # Configure logging with both file and console output (APPEND mode)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),  # Append mode - don't overwrite!
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_simple_data_integrity_test():
    """Run a simple but comprehensive data integrity test"""
    logger = setup_logging()
    
    logger.info("🚀 Starting RecessionRadar Data Integrity Testing")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test 1: Check if CSV file exists and is readable
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
        logger.error(f"✗ CSV file test failed: {e}")
        test_results['csv_exists'] = 'FAILED'
    
    # Test 2: Basic data validation if CSV exists
    if os.path.exists("data/recession_probability.csv"):
        logger.info("\n📊 Test 2: Basic Data Validation")
        try:
            df = pd.read_csv("data/recession_probability.csv")
            
            # Check for reasonable data ranges
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            logger.info(f"✓ Found {len(numeric_cols)} numeric columns: {list(numeric_cols)}")
            
            for col in numeric_cols:
                col_stats = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'null_count': int(df[col].isnull().sum())
                }
                logger.info(f"✓ {col}: min={col_stats['min']:.4f}, max={col_stats['max']:.4f}, mean={col_stats['mean']:.4f}, nulls={col_stats['null_count']}")
                
                # Check if probability-like columns are in [0,1] range
                if 'prob' in col.lower():
                    if col_stats['min'] >= 0 and col_stats['max'] <= 1:
                        logger.info(f"✓ {col}: Probability values in valid range [0,1]")
                    else:
                        logger.warning(f"⚠️ {col}: Probability values outside [0,1] range")
            
            test_results['data_validation'] = 'PASSED'
            
        except Exception as e:
            logger.error(f"✗ Data validation failed: {e}")
            test_results['data_validation'] = 'FAILED'
    
    # Test 3: API modules import test
    logger.info("\n🔧 Test 3: API Module Import Test")
    try:
        from api.data_collection import fetch_latest_fred_value
        logger.info("✓ Successfully imported fetch_latest_fred_value")
        
        from api.ML_pipe import load_model
        logger.info("✓ Successfully imported load_model")
        
        test_results['imports'] = 'PASSED'
        
    except Exception as e:
        logger.error(f"✗ Import test failed: {e}")
        test_results['imports'] = 'FAILED'
    
    # Test 4: Model files existence
    logger.info("\n🤖 Test 4: Model Files Existence")
    try:
        model_dir = 'models'
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            logger.info(f"✓ Found {len(model_files)} model files in {model_dir}/")
            for model_file in model_files[:5]:  # Log first 5 files
                logger.info(f"  - {model_file}")
            if len(model_files) > 5:
                logger.info(f"  ... and {len(model_files) - 5} more files")
            test_results['models'] = 'PASSED'
        else:
            logger.warning("⚠️ Models directory not found")
            test_results['models'] = 'SKIPPED'
            
    except Exception as e:
        logger.error(f"✗ Model files test failed: {e}")
        test_results['models'] = 'FAILED'
    
    # Test 5: Environment variables
    logger.info("\n🔑 Test 5: Environment Variables")
    try:
        fred_key = os.getenv('FRED_API_KEY')
        if fred_key:
            logger.info(f"✓ FRED_API_KEY found (length: {len(fred_key)} characters)")
            test_results['env_vars'] = 'PASSED'
        else:
            logger.warning("⚠️ FRED_API_KEY not found in environment variables")
            test_results['env_vars'] = 'SKIPPED'
            
    except Exception as e:
        logger.error(f"✗ Environment variables test failed: {e}")
        test_results['env_vars'] = 'FAILED'
    
    # Generate summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST EXECUTION SUMMARY")
    logger.info("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result == 'PASSED')
    failed_tests = sum(1 for result in test_results.values() if result == 'FAILED')
    skipped_tests = sum(1 for result in test_results.values() if result == 'SKIPPED')
    
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests} ✓")
    logger.info(f"Failed: {failed_tests} ✗")
    logger.info(f"Skipped: {skipped_tests} ⚠️")
    
    if total_tests > 0:
        success_rate = (passed_tests / total_tests) * 100
        logger.info(f"Success Rate: {success_rate:.1f}%")
    
    # Log detailed results
    logger.info("\nDetailed Results:")
    for test_name, result in test_results.items():
        status_icon = "✓" if result == 'PASSED' else "✗" if result == 'FAILED' else "⚠️"
        logger.info(f"{status_icon} {test_name.replace('_', ' ').title()}: {result}")
    
    # Save detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'skipped': skipped_tests,
            'success_rate': (passed_tests/total_tests)*100 if total_tests > 0 else 0
        },
        'detailed_results': test_results
    }
    
    with open('data_integrity_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n📋 Detailed report saved to: data_integrity_test_report.json")
    logger.info(f"📋 Full log saved to: tests/reports/json/data_integrity_test.log")
    
    if failed_tests > 0:
        logger.warning("⚠️ Some tests failed. Please review the detailed results above.")
        return False
    else:
        logger.info("🎉 All tests completed successfully!")
        return True

if __name__ == "__main__":
    success = run_simple_data_integrity_test()
    sys.exit(0 if success else 1)