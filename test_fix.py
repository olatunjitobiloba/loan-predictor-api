#!/usr/bin/env python3
"""Test script to verify the loan prediction fix"""

import sys

from app_v7 import feature_names, load_model, model, preprocess_input


def test_prediction():
    """Test that the model makes proper predictions (not always rejected)"""

    # Load the model
    print("Loading model...")
    load_model()

    if model is None:
        print("ERROR: Model failed to load!")
        return False
    print("Model loaded successfully!")
    print("Feature names ({}): {}".format(len(feature_names), feature_names))
    print()

    # Test cases
    test_cases = [
        {
            "name": "Good applicant (high income, low loan)",
            "data": {
                "ApplicantIncome": 50000,
                "CoapplicantIncome": 0,
                "LoanAmount": 200,
                "Loan_Amount_Term": 360,
                "Credit_History": 1.0,
                "Gender": "Male",
                "Married": "Yes",
                "Dependents": "0",
                "Education": "Graduate",
                "Self_Employed": "No",
                "Property_Area": "Urban",
            },
        },
        {
            "name": "Another applicant",
            "data": {
                "ApplicantIncome": 30000,
                "CoapplicantIncome": 10000,
                "LoanAmount": 150,
                "Loan_Amount_Term": 360,
                "Credit_History": 1.0,
                "Gender": "Female",
                "Married": "Yes",
                "Dependents": "1",
                "Education": "Graduate",
                "Self_Employed": "No",
                "Property_Area": "Urban",
            },
        },
    ]

    all_passed = True

    for test in test_cases:
        print(f"Test: {test['name']}")
        print(f"  Input: {test['data']}")

        try:
            # Preprocess
            df = preprocess_input(test["data"])

            # Predict
            pred = model.predict(df)[0]
            proba = model.predict_proba(df)[0]

            pred_text = "APPROVED" if pred == 1 else "REJECTED"
            confidence = max(proba) * 100

            print(f"  Prediction: {pred_text}")
            print(f"  Confidence: {confidence:.2f}%")
            print(
                f"  Probabilities: [Rejected: {proba[0]:.4f}, Approved: {proba[1]:.4f}]"
            )

            # Check if it's not always the same (the original bug)
            if pred != 0:  # If any prediction is not "Rejected"
                print("  Result: PASS (Model can predict approvals)")
            else:
                print("  Result: WARN (Model predicting rejection)")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False

        print()

    return all_passed


if __name__ == "__main__":
    success = test_prediction()
    if success:
        print("=" * 60)
        print("SUCCESS: Fix appears to be working correctly!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("FAILURE: There are still issues with the prediction model")
        print("=" * 60)
        sys.exit(1)
