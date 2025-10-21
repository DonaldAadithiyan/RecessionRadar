# RecessionRadar Functional Testing Suite

## Section 3.1.2 Function Testing Implementation

This directory contains comprehensive functional testing for the RecessionRadar application, focusing on use cases, business functions, and business rules validation.

## Testing Objectives

**Technique Objective:** Exercise target-of-test functionality, including navigation, data entry, processing, and retrieval to observe and log target behavior.

**Technique:** Execute each use-case scenario's individual use-case flows or functions and features, using valid and invalid data, to verify that:
- Expected results occur when valid data is used
- Appropriate error or warning messages are displayed when invalid data is used  
- Each business rule is properly applied

## Test Structure

### 📁 Test Files

1. **`test_api_endpoints.py`** - API Endpoint Functional Testing
   - Tests all FastAPI endpoints
   - Validates Pydantic model compliance
   - Tests error handling and status codes
   - Business rule validation

2. **`test_ml_pipeline.py`** - ML Pipeline Functional Testing
   - Tests ML pipeline components
   - Feature engineering validation
   - Model loading and prediction testing
   - Data collection function testing

3. **`master_functional_test_runner.py`** - Master Test Runner
   - Orchestrates all functional tests
   - Generates professional reports
   - Provides comprehensive logging

## Running Tests

### Option 1: Run Master Test Suite (Recommended)
```bash
python functional_tests/master_functional_test_runner.py
```

### Option 2: Run Individual Test Suites
```bash
# API endpoint tests
python -m pytest functional_tests/test_api_endpoints.py -v

# ML pipeline tests
python -m pytest functional_tests/test_ml_pipeline.py -v
```

### Option 3: Run with Coverage
```bash
python -m pytest functional_tests/ --cov=api --cov-report=html -v
```

## Test Coverage

### API Endpoints Tested
- ✅ `GET /` - Root endpoint
- ✅ `GET /api/treasury-yields` - Treasury yields data
- ✅ `GET /api/economic-indicators` - Economic indicators
- ✅ `GET /api/recession-probabilities` - Recession probabilities
- ✅ `GET /api/historical-economic-data` - Historical data
- ✅ `GET /api/current-prediction` - Current predictions
- ✅ `POST /api/custom-prediction` - Custom predictions

### Business Rules Validated
- ✅ Recession probabilities within 0-100% range
- ✅ Economic indicators positive value validation
- ✅ Treasury yield reasonableness checks
- ✅ Pydantic schema compliance
- ✅ HTTP status code correctness
- ✅ Error message appropriateness

### ML Pipeline Components
- ✅ Time series feature engineering
- ✅ Time series prediction
- ✅ Regression feature engineering
- ✅ Regression prediction
- ✅ Model loading functionality
- ✅ Data collection functions

## Oracles (Validation Strategies)

**Expected Outputs:** Defined by FastAPI schemas (TreasuryYields, EconomicIndicators, RecessionPrediction, etc.)

**Validation Method:** Test results compared against Pydantic data models for schema compliance

**Automated Assessment:** Self-verifying tests with automated pass/fail determination

## Required Tools

- ✅ **pytest** - Test execution framework
- ✅ **FastAPI TestClient** - API endpoint testing
- ✅ **unittest.mock** - Mocking external dependencies
- ✅ **requests** - HTTP request testing
- ✅ **Pydantic** - Schema validation

## Success Criteria

### Primary Success Criteria
- ✅ All API endpoints return HTTP 200 for valid requests
- ✅ Appropriate error codes for invalid requests
- ✅ All responses match expected Pydantic schemas
- ✅ Business rules properly enforced
- ✅ ML pipeline components function correctly

### Key Use Cases Tested
- ✅ Valid data processing and retrieval
- ✅ Invalid data error handling
- ✅ API endpoint navigation
- ✅ Business rule enforcement
- ✅ Schema compliance validation

## Special Considerations

### Mocking Strategy
- **External APIs:** FRED API calls mocked to avoid external dependencies
- **ML Models:** Model predictions mocked for isolated testing
- **File I/O:** Model loading operations mocked where appropriate

### Test Environment
- Tests designed to run without external API keys
- Mock data provides consistent test scenarios
- Isolated testing prevents external service dependencies

## Report Generation

### Automated Reports
- **`FUNCTIONAL_TEST_REPORT.log`** - Detailed execution log
- **`FUNCTIONAL_TEST_REPORT.json`** - Machine-readable test results

### Report Contents
- Test execution summary
- Individual test results
- Business rule validation status
- Performance metrics
- Success/failure analysis

## Integration with CI/CD

This functional test suite is designed to integrate with CI/CD pipelines:

```yaml
# Example GitHub Actions integration
- name: Run Functional Tests
  run: |
    python functional_tests/master_functional_test_runner.py
    
- name: Upload Test Reports
  uses: actions/upload-artifact@v3
  with:
    name: functional-test-reports
    path: |
      FUNCTIONAL_TEST_REPORT.log
      FUNCTIONAL_TEST_REPORT.json
```

## Maintenance

### Adding New Tests
1. Add test methods to appropriate test classes
2. Follow naming convention: `test_<functionality>_<scenario>`
3. Include both positive and negative test cases
4. Update business rule validation as needed

### Updating Mock Data
- Update mock data in test class `setup_class` methods
- Ensure mock data reflects current API schemas
- Maintain consistency across test scenarios

---

**Note:** This functional testing suite implements Section 3.1.2 Function Testing requirements with focus on black box testing techniques, business rule validation, and comprehensive API endpoint coverage.