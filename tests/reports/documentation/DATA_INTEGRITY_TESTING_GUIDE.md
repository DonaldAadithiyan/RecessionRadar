# RecessionRadar Data Integrity Testing Guide

## Overview

This guide provides step-by-step instructions for implementing comprehensive data integrity testing for the RecessionRadar economic analysis system. The testing framework validates data accuracy, pipeline integrity, and system reliability.

## Testing Architecture

```
tests/
├── test_data_integrity.py          # Pytest-based unit tests
├── run_data_integrity_tests.py     # Comprehensive test runner
├── data_validation_config.py       # Validation rules and configurations
└── data_integrity_test_report.json # 
```

## Step-by-Step Implementation

### Step 1: Environment Setup

1. **Install Required Dependencies**
   ```bash
   pip install pytest pandas numpy great-expectations requests-mock
   ```

2. **Set Environment Variables**
   ```bash
   # Create .env file with FRED API key
   FRED_API_KEY=your_fred_api_key_here
   ```

3. **Create Test Directory Structure**
   ```bash
   mkdir tests
   cd tests
   ```

### Step 2: Execute Data Integrity Tests

#### Option A: Run Comprehensive Test Suite
```bash
# Run the complete test suite with detailed reporting
python tests/run_data_integrity_tests.py
```

**Expected Output:**
```
🚀 Starting RecessionRadar Data Integrity Testing
============================================================

📋 Test 1: Schema Validation
------------------------------
✓ UNRATE: Valid pandas Series
✓ GDP: Valid pandas Series
...

📊 Test 2: Data Range Validation
------------------------------
✓ UNRATE values within expected range
✓ GDP values within expected range
...

🎉 All tests passed successfully!
```

#### Option B: Run Pytest-Based Tests
```bash
# Run individual test components
python -m pytest tests/test_data_integrity.py -v -s
```

### Step 3: Interpret Test Results

#### Test Categories and Success Criteria

1. **Schema Validation Tests**
   - ✅ **Pass Criteria**: All data columns conform to expected pandas data types
   - ❌ **Fail Indicators**: Non-numeric values in economic indicators, invalid datetime indices
   - **Action on Failure**: Check FRED API response format, validate data transformation functions

2. **Data Range Validation Tests**
   - ✅ **Pass Criteria**: Economic indicators within historical reasonable ranges
   - ❌ **Fail Indicators**: Unemployment rates >25%, interest rates <-2% or >20%
   - **Action on Failure**: Investigate API data quality, check for data corruption

3. **Data Completeness Tests**
   - ✅ **Pass Criteria**: No missing required columns, minimal null values
   - ❌ **Fail Indicators**: Missing critical economic indicators, excessive null values
   - **Action on Failure**: Verify API connectivity, check data fetching logic

4. **Temporal Consistency Tests**
   - ✅ **Pass Criteria**: Chronological date ordering, no future dates
   - ❌ **Fail Indicators**: Out-of-order dates, unrealistic date ranges
   - **Action on Failure**: Review date parsing logic, validate API response handling

5. **Error Handling Tests**
   - ✅ **Pass Criteria**: Graceful handling of API failures, malformed data
   - ❌ **Fail Indicators**: Unhandled exceptions, system crashes on bad data
   - **Action on Failure**: Implement robust error handling, add fallback mechanisms

6. **CSV File Integrity Tests**
   - ✅ **Pass Criteria**: Valid file structure, probability values [0,1]
   - ❌ **Fail Indicators**: Corrupted files, invalid probability ranges
   - **Action on Failure**: Check file write permissions, validate prediction logic

### Step 4: Advanced Testing Scenarios

#### Testing with Live Data
```python
# Test with current FRED API data
python -c "
from api.data_collection import fetch_and_combine_fred_series
from tests.data_validation_config import DataValidationRules

# Fetch live data
data = fetch_and_combine_fred_series()
validator = DataValidationRules()

# Validate each indicator
for col in data.columns:
    if col in validator.INDICATOR_RANGES:
        is_valid, msg = validator.validate_indicator_range(data[col], col)
        print(f'{col}: {msg}')
"
```

#### Testing with Corrupted Data
```python
# Generate and test corrupted data scenarios
python -c "
from tests.data_validation_config import TestDataGenerator, DataValidationRules

generator = TestDataGenerator()
validator = DataValidationRules()

# Test missing data handling
corrupted = generator.generate_corrupted_data(['UNRATE', 'GDP'], 'missing')
for indicator, series in corrupted.items():
    is_valid, msg = validator.validate_indicator_range(series, indicator)
    print(f'{indicator} (missing data): {msg}')
"
```

### Step 5: Automated Testing Integration

#### CI/CD Pipeline Integration
```yaml
# GitHub Actions example (.github/workflows/data-integrity.yml)
name: Data Integrity Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pandas numpy
    - name: Run Data Integrity Tests
      env:
        FRED_API_KEY: ${{ secrets.FRED_API_KEY }}
      run: |
        python tests/run_data_integrity_tests.py
        python -m pytest tests/test_data_integrity.py -v
```

#### Scheduled Testing
```bash
# Cron job for daily data integrity validation
# Add to crontab: crontab -e
0 6 * * * cd /path/to/RecessionRadar && python tests/run_data_integrity_tests.py >> data_integrity.log 2>&1
```

### Step 6: Monitoring and Alerting

#### Test Report Analysis
```python
# Analyze test report programmatically
import json

with open('data_integrity_test_report.json', 'r') as f:
    report = json.load(f)

# Check success rate
success_rate = report['summary']['success_rate']
if success_rate < 100:
    print(f"⚠️ Test success rate: {success_rate}%")
    
    # Identify failed tests
    for test_name, result in report['detailed_results'].items():
        if result['status'] == 'FAILED':
            print(f"❌ {test_name}: {result.get('error', 'Unknown error')}")
```

#### Email Alerting (Optional)
```python
# Send email alerts for test failures
import smtplib
from email.mime.text import MIMEText

def send_alert(test_results):
    failed_tests = [name for name, result in test_results.items() if result['status'] == 'FAILED']
    
    if failed_tests:
        msg = MIMEText(f"Data integrity tests failed: {', '.join(failed_tests)}")
        msg['Subject'] = 'RecessionRadar Data Integrity Alert'
        msg['From'] = 'alerts@yourcompany.com'
        msg['To'] = 'admin@yourcompany.com'
        
        # Configure SMTP server
        server = smtplib.SMTP('smtp.yourcompany.com', 587)
        server.send_message(msg)
        server.quit()
```

## Troubleshooting Common Issues

### Issue: Import Errors
**Problem**: `ImportError: cannot import name 'function_name'`
**Solution**: 
1. Check function names in source files: `grep -n "def " api/*.py`
2. Verify Python path includes project root
3. Check for typos in function names

### Issue: API Key Errors
**Problem**: `ValueError: FRED API key not found`
**Solution**:
1. Create `.env` file with `FRED_API_KEY=your_key`
2. Verify environment variable loading: `python -c "import os; print(os.getenv('FRED_API_KEY'))"`
3. Request API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)

### Issue: Data Range Violations
**Problem**: Economic indicators outside expected ranges
**Solution**:
1. Check recent economic conditions for unusual values
2. Update validation ranges in `data_validation_config.py`
3. Investigate API data quality issues

### Issue: Performance Degradation
**Problem**: Tests running slowly or timing out
**Solution**:
1. Reduce test dataset size for faster execution
2. Use cached data samples instead of live API calls
3. Implement parallel test execution
4. Add timeout limits to API calls

## Best Practices

1. **Regular Execution**: Run tests daily or after each data update
2. **Version Control**: Track test configurations and validation rules
3. **Documentation**: Maintain clear test result interpretation guides
4. **Monitoring**: Set up alerts for test failures
5. **Continuous Improvement**: Update validation rules based on changing economic conditions

## Performance Benchmarks

| Test Category | Expected Duration | Success Threshold |
|---------------|------------------|-------------------|
| Schema Validation | <5 seconds | 100% |
| Data Range Validation | <10 seconds | >95% |
| API Error Handling | <15 seconds | 100% |
| CSV Integrity | <5 seconds | 100% |
| Complete Suite | <60 seconds | >90% |

## Next Steps

1. **Extend Coverage**: Add tests for ML model predictions
2. **Integration Testing**: Test frontend-backend data flow
3. **Load Testing**: Validate performance under concurrent access
4. **Security Testing**: Test input validation and API security

This comprehensive testing framework ensures the RecessionRadar system maintains high data quality and reliability for accurate economic analysis and recession probability predictions.