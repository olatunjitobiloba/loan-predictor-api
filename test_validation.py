import requests
import json
import pytest

BASE_URL = "http://localhost:5000"


def validation_case(name, data, expected_status=200):
    """Helper function to test a validation case"""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    print(f"Input: {json.dumps(data, indent=2)}")

    response = requests.post(f"{BASE_URL}/predict", json=data)

    print(f"\nStatus: {response.status_code} (Expected: {expected_status})")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == expected_status, \
        f"Expected {expected_status}, got {response.status_code}: {response.json()}"
    print("✅ PASS")
    
    return response


def test_valid_complete_data():
    """Test 1: Valid complete data"""
    validation_case("Valid Complete Data", {
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1500,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "Property_Area": "Urban"
    }, 200)


def test_minimal_valid_data():
    """Test 2: Minimal valid data"""
    validation_case("Minimal valid Data", {
        "ApplicantIncome": 5000
    }, 200)


def test_missing_required_field():
    """Test 3: Missing required field"""
    validation_case("Missing Required Field", {
        "LoanAmount": 150
    }, 400)


def test_invalid_data_type():
    """Test 4: Invalid data type"""
    validation_case("Invalid Data Type", {
        "ApplicantIncome": "not a number"
    }, 400)


def test_negative_income():
    """Test 5: Negative income"""
    validation_case("Negative Income", {
        "ApplicantIncome": -5000
    }, 400)


def test_income_out_of_range():
    """Test 6: Income too high"""
    validation_case("Income Out of Range", {
        "ApplicantIncome": 200000
    }, 400)


def test_invalid_gender():
    """Test 7: Invalid gender"""
    validation_case("Invalid Gender", {
        "ApplicantIncome": 5000,
        "Gender": "Other"
    }, 400)


def test_invalid_education():
    """Test 8: Invalid education"""
    validation_case("Invalid Education", {
        "ApplicantIncome": 5000,
        "Education": "PhD"
    }, 400)


def test_valid_with_warnings():
    """Test 9: Valid but with warnings (low income, no credit)"""
    validation_case("Valid with Warnings", {
        "ApplicantIncome": 800,
        "LoanAmount": 300,
        "Credit_History": 0
    }, 200)


def test_high_loan_to_income_ratio():
    """Test 10: Valid but with warnings (high loan-to-income)"""
    validation_case("High Loan-to-Income Ratio", {
        "ApplicantIncome": 2000,
        "LoanAmount": 500
    }, 200)


def test_empty_request():
    """Test 11: Empty request"""
    validation_case("Empty Request", {}, 400)


def test_multiple_validation_errors():
    """Test 12: Multiple errors"""
    validation_case("Multiple Validation Errors", {
        "ApplicantIncome": -1000,
        "Gender": "Unknown",
        "LoanAmount": 50000
    }, 400)


def test_validation_rules_endpoint():
    """Test validation rules endpoint"""
    print("\n\nFetching validation rules...")
    response = requests.get(f"{BASE_URL}/validation-rules")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    print(json.dumps(response.json(), indent=2))