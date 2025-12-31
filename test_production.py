"""
Test production API deployment
"""

import requests
import json

# CHANGE THIS to your actual Render URL
PRODUCTION_URL = "https://loan-predictor-api-91xu.onrender.com"

def test_production_api():
    """Comprehensive production API test"""
    
    print("="*70)
    print("PRODUCTION API TEST SUITE")
    print(f"Testing: {PRODUCTION_URL}")
    print("="*70)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{PRODUCTION_URL}/health", timeout=10)
        if response.status_code == 200:
            print("   ✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False
    
    # Test 2: API info
    print("\n2. Testing API info endpoint...")
    try:
        response = requests.get(f"{PRODUCTION_URL}/api", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("   ✅ API info retrieved")
            print(f"   Version: {data.get('version')}")
            print(f"   Model accuracy: {data.get('model_accuracy')}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Test 3: Model info
    print("\n3. Testing model info endpoint...")
    try:
        response = requests.get(f"{PRODUCTION_URL}/model-info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Model info retrieved")
            print(f"   Accuracy: {data.get('accuracy')}")
            print(f"   Features: {data.get('feature_count')}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Test 4: Make prediction
    print("\n4. Testing prediction endpoint...")
    test_data = {
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
    }
    
    try:
        response = requests.post(
            f"{PRODUCTION_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Prediction successful")
            print(f"   Result: {data.get('prediction')}")
            print(f"   Confidence: {data.get('confidence'):.2%}")
            print(f"   Prediction ID: {data.get('prediction_id')}")
        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Test 5: Validation (should fail)
    print("\n5. Testing validation (negative test)...")
    invalid_data = {"ApplicantIncome": -5000}
    
    try:
        response = requests.post(
            f"{PRODUCTION_URL}/predict",
            json=invalid_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 400:
            print("   ✅ Validation correctly rejected invalid data")
            print(f"   Error: {response.json().get('validation_errors')}")
        else:
            print(f"   ❌ Validation failed to catch error")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Test 6: Statistics
    print("\n6. Testing statistics endpoint...")
    try:
        response = requests.get(f"{PRODUCTION_URL}/statistics", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Statistics retrieved")
            print(f"   Total predictions: {data.get('total_predictions')}")
            print(f"   Approval rate: {data.get('approval_rate')}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Test 7: History
    print("\n7. Testing history endpoint...")
    try:
        response = requests.get(f"{PRODUCTION_URL}/history?limit=5", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("   ✅ History retrieved")
            print(f"   Count: {data.get('count')}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    print("\n" + "="*70)
    print("✅ PRODUCTION API TEST COMPLETE")
    print("="*70)
    print(f"\n🎉 Your API is LIVE at: {PRODUCTION_URL}")
    print("\nShare this URL to demonstrate your project!")
    
    return True

if __name__ == '__main__':
    test_production_api()
