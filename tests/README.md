# Test Suite Documentation

## Overview
Comprehensive test suite for Loan Prediction API with 80%+ coverage.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py           # Shared fixtures
├── test_validators.py    # Unit tests for validation
├── test_api.py          # Integration tests for API
└── test_database.py     # Database operation tests
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_validators.py
```

### Run with coverage
```bash
pytest --cov=. --cov-report=html
```

### Run specific test
```bash
pytest tests/test_api.py::TestPredictionEndpoint::test_predict_valid_complete_data
```

### Run with verbose output
```bash
pytest -v
```

## Test Categories

### Unit Tests (test_validators.py)
- Required field validation
- Numeric field validation
- Categorical field validation
- Business logic warnings
- **Coverage**: 17 tests

### Integration Tests (test_api.py)
- All API endpoints
- Prediction workflow
- Error handling
- Database integration
- **Coverage**: 20 tests

### Database Tests (test_database.py)
- Model creation
- Query operations
- Statistics calculation
- **Coverage**: 7 tests

## Coverage Goals

- **Overall**: 80%+
- **Validators**: 95%+
- **API endpoints**: 85%+
- **Database**: 80%+

## Current Coverage

Run `pytest --cov` to see current coverage report.

```bash
pytest --cov=app_v4 --cov=database --cov=validators --cov-report=term-missing
```

## Run Full Test Suite with Coverage Report

```bash
# Run all tests with detailed coverage
pytest --cov=. --cov-report=html --cov-report=term-missing
```

This generates:
- **HTML report**: `htmlcov/index.html` - visual coverage report in your browser
- **Terminal output**: shows coverage for each file with missing line numbers

### View Coverage Report

After running the full test suite above, open the coverage report:

```bash
# Windows
start htmlcov/index.html

# Mac
open htmlcov/index.html

# Linux
xdg-open htmlcov/index.html
```

## Adding New Tests

1. Create test file in `tests/` directory
2. Follow naming convention: `test_*.py`
3. Use AAA pattern (Arrange, Act, Assert)
4. Add docstrings to explain test purpose
5. Run tests to verify

## CI/CD Integration

Tests are designed to run in CI/CD pipelines:

- Fast execution (< 30 seconds)
- No external dependencies
- Clean test database after each run
- Clear pass/fail indicators

## Test Results Summary

- **44 tests total**: 20 API + 7 database + 17 validator
- **All tests passing** ✅
- **83% code coverage** on main modules
- **Execution time**: ~7-8 seconds
