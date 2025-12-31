import requests
import json
import time
import pytest

BASE_URL = "http://localhost:5000"

# Store prediction IDs from first test
_prediction_ids = []


def test_make_predictions_and_populate_database():
    """Test 1: Make some predictions to populate database"""
    global _prediction_ids
    
    print("="*70)
    print("TEST 1: Making predictions to populate database...")
    print("="*70)
    
    test_cases = [
        {
            "ApplicantIncome": 5000,
            "LoanAmount": 150,
            "Credit_History": 1,
            "Education": "Graduate"
        },
        {
            "ApplicantIncome": 3000,
            "LoanAmount": 200,
            "Credit_History": 0,
        },
        {
            "ApplicantIncome": 8000,
            "CoapplicantIncome": 2000,
            "LoanAmount": 100,
            "Credit_History": 1
        }
    ]

    for i, test_data in enumerate(test_cases, 1):
        print(f"\n Prediction {i}...")
        response = requests.post(f"{BASE_URL}/predict", json=test_data)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.json()}"
        result = response.json()
        pred_id = result.get('prediction_id')
        _prediction_ids.append(pred_id)
        print(f" ✅ Stored with ID: {pred_id}")
        print(f"  Result: {result['prediction']}({result['confidence']:.2%})")
        time.sleep(0.5)

    assert len(_prediction_ids) == 3, f"Expected 3 prediction IDs, got {len(_prediction_ids)}"


def test_fetch_recent_history():
    """Test 2: Get recent history"""
    print("\n" + "="*70)
    print("TEST 2: Fetching recent history...")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/history?limit=5")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    history = response.json()
    print(f"✅ Found {history['count']} predictions")
    for pred in history['predictions']:
        print(f"\n ID: {pred['id']}")
        print(f" Time: {pred['timestamp']}")
        print(f" Income: ${pred['input']['applicant_income']}")
        print(f" Result: {pred['prediction']}({pred['confidence']:.2%})")


def test_fetch_specific_prediction():
    """Test 3: Get specific prediction"""
    global _prediction_ids
    
    if not _prediction_ids:
        pytest.skip("No predictions available")
    
    print("\n" + "="*70)
    print(f"TEST 3: Fetching specific prediction (ID: {_prediction_ids[0]})...")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/history/{_prediction_ids[0]}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    pred = response.json()
    print("✅ Prediction details:")
    print(json.dumps(pred, indent=2))


def test_fetch_statistics():
    """Test 4: Get statistics"""
    print("\n" + "="*70)
    print("TEST 4: Fetching statistics...")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/statistics")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    stats = response.json()
    print("✅ Statistics:")
    print(f" Total Predictions: {stats['total_predictions']}")
    print(f" Approved: {stats['approved']}")
    print(f" Rejected: {stats['rejected']}")
    print(f" Approval rate: {stats['approval_rate']}")
    print(f" Avg confidence: {stats['average_confidence']}")


def test_fetch_analytics():
    """Test 5: Get analytics"""
    print("\n" + "="*70)
    print("TEST 5: Fetching analytics...")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/analytics")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    analytics = response.json()
    print("✅ Analytics:")
    print(json.dumps(analytics, indent=2))


def test_check_api_endpoint():
    """Test 6: Check API endpoint shows stats"""
    print("\n" + "="*70)
    print("TEST 6: Checking API endpoint...")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/api")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    api_info = response.json()
    print("✅ API Info:")
    print(f" Status: {api_info['status']}")
    print(f" Version: {api_info['version']}")
    print(f" Model Accuracy: {api_info['model_accuracy']}")
    print(f" Statistics: {api_info['statistics']}")


if __name__ == "__main__":
    print("="*70)
    print("DATABASE INTEGRATION TEST SUITE")
    print("="*70)
    print("\nRun with: pytest test_database.py -v -s")
