import json

import requests

# API URL
BASE_URL = "http://localhost:5000"


def predict_endpoint(data):
    """Helper function to test the prediction endpoint"""
    url = f"{BASE_URL}/predict"
    headers = {"Content-Type": "application/json"}

    print(f"\n{'='*60}")
    print("TESTING PREDICTION")
    print(f"{'='*60}")
    print(f"Input: {json.dumps(data, indent=2)}")

    response = requests.post(url, json=data, headers=headers)

    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print(f"{'='*60}\n")

    return response.json()


def get_stats_endpoint():
    """Helper function to test the stats endpoint"""
    url = f"{BASE_URL}/stats"
    response = requests.get(url)
    print(f"\n{'='*60}")
    print("API STATISTICS")
    print(f"{'='*60}")
    print(json.dumps(response.json(), indent=2))
    print(f"{'='*60}\n")


# Test cases
if __name__ == "__main__":
    # Test 1: High income, good credit (minimal fields)
    predict_endpoint(
        {
            "ApplicantIncome": 10000,
            "CoapplicantIncome": 3000,
            "LoanAmount": 150,
            "Credit_History": 1,
            "Education": "Graduate",
        }
    )

    # Test 2: Low income, bad credit
    predict_endpoint({"ApplicantIncome": 2000, "LoanAmount": 300, "Credit_History": 0})

    # Test 3: Medium income, good credit
    predict_endpoint(
        {
            "ApplicantIncome": 5000,
            "CoapplicantIncome": 2000,
            "LoanAmount": 200,
            "Credit_History": 1,
        }
    )

    # Test 4: Complete data - all fields provided
    predict_endpoint(
        {
            "ApplicantIncome": 10000,
            "CoapplicantIncome": 3000,
            "LoanAmount": 150,
            "Loan_Amount_Term": 360,
            "Credit_History": 1,
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "Property_Area": "Urban",
        }
    )
    # Test 5: Very high income, small loan
    predict_endpoint(
        {
            "ApplicantIncome": 15000,
            "CoapplicantIncome": 5000,
            "LoanAmount": 100,
            "Loan_Amount_Term": 360,
            "Credit_History": 1,
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "Property_Area": "Urban",
        }
    )

    # Test 6: Moderate income, tiny loan
    predict_endpoint(
        {
            "ApplicantIncome": 6000,
            "CoapplicantIncome": 2000,
            "LoanAmount": 50,
            "Loan_Amount_Term": 360,
            "Credit_History": 1,
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "1",
            "Education": "Graduate",
            "Self_Employed": "No",
            "Property_Area": "Semiurban",
        }
    )

    # Test 7: Try different property area
    predict_endpoint(
        {
            "ApplicantIncome": 8000,
            "CoapplicantIncome": 4000,
            "LoanAmount": 120,
            "Loan_Amount_Term": 360,
            "Credit_History": 1,
            "Gender": "Female",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "Property_Area": "Rural",
        }
    )

    # Add this test - using typical approved profile from loan datasets
    predict_endpoint(
        {
            "ApplicantIncome": 4583,
            "CoapplicantIncome": 1508,
            "LoanAmount": 128,
            "Loan_Amount_Term": 360,
            "Credit_History": 1,
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "Property_Area": "Rural",
        }
    )

    # 🎯 Call stats AFTER all predictions
    get_stats_endpoint()
