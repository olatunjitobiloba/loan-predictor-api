import requests
import json

BASE_URL = "http://localhost:5000"

def test_case(name, data, expected_status=200):
    """Test a validation case"""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    print(f"Input: {json.dumps(data, indent=2)}")

    response = requests.post(f"{BASE_URL}/predict", json=data)

    print(f"\nStatus: {response.status_code} (Expected: {expected_status})")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    if response.status_code == expected_status:
        print("✅ PASS")
    else:
        print("❌ FAIL")
    
    return response

# Test cases
print("="*70)
print("VALIDATION TEST SUITE")
print("="*70)

# Test 1: Valid complete data
test_case("Valid Complete Data", {
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
}), 200

# Test 2: Minimal valid data
test_case("Minimal valid Data", {
    "ApplicantIncome": 5000
}, 200)

# Test 3: Missing required field
test_case("Missing Required Field", {
    "LoanAmount": 150
}, 400)

# Test 4: Invalid data type
test_case("Invalid Data Type", {
    "ApplicantIncome": "not a number"
}, 400)

# Test 5: Negative income
test_case("Negative Income", {
    "ApplicantIncome": -5000
}, 400)

# Test 6: Income too high
test_case("Income Out of Range", {
    "ApplicantIncome": 200000
}, 400)

# Test 7: Invalid gender
test_case("Invalid Gender", {
    "ApplicantIncome": 5000,
    "Gender": "Other"
}, 400)

# Test 8: Invalid education
test_case("Invalid Education",{
    "ApplicantIncome": 5000,
    "Education": "PhD"
}, 400)

# Test 9: Valid but with warnings (low income, no credit)
test_case("Valid with Warnings", {
    "ApplicantIncome": 800,
    "LoanAmount": 300,
    "Credit_History": 0
}, 200)

# Test 10: Valid but with warnings (high loan-to-income)
test_case("High Loan-to-Income Ratio", {
    "ApplicantIncome": 2000,
    "LoanAmount": 500
}, 200)

# Test 11: Empty request
test_case("Empty Request", {}, 400)

# Test 12: Multiple errors
test_case("Multiple Validation Errors", {
    "ApplicantIncome": -1000,
    "Gender": "Unknown",
    "LoanAmount": 50000
}, 400)

print(f"\n{'='*70}")
print("TEST SUITE COMPLETE")
print(f"{'='*70}")

# Test validation rules endpoint
print("\n\nFetching validation rules...")
response = requests.get(f"{BASE_URL}/validation-rules")
print(json.dumps(response.json(), indent=2))