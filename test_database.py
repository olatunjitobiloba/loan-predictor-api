import requests
import json
import time

BASE_URL = "http://localhost:5000"

print("="*70)
print("DATABASE INTEGRATION TEST SUITE")
print("="*70)

# Test 1: Make some predictions
print("\n1. Making predictions to populate database...")
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

prediction_ids = []

for i, test_data in enumerate(test_cases, 1):
    print(f"\n Prediction {i}...")
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    if response.status_code == 200:
        result = response.json()
        pred_id = result.get('prediction_id')
        prediction_ids.append(pred_id)
        print(f" ✅ Stored with ID: {pred_id}")
        print(f"  Result: {result['prediction']}({result['confidence']:.2%})")
    else:
        print(f" ❌ Failed: {response.json()}")
    time.sleep(0.5)

# Test 2: Get recent history
print("\n" + "="*70)
print("2. Fetching recent history...")
print("="*70)
response = requests.get(f"{BASE_URL}/history?limit=5")
if response.status_code == 200:
    history = response.json()
    print(f"✅ Found {history['count']} predictions")
    for pred in history['predictions']:
        print(f"\n ID: {pred['id']}")
        print(f" Time: {pred['timestamp']}")
        print(f" Income: ${pred['input']['applicant_income']}")
        print(f" Result: {pred['prediction']}({pred['confidence']:.2%})")
else:
    print(f"❌ Failed: {response.json()}")

# Test 3: Get specific prediction
if prediction_ids:
    print("\n" + "="*70)
    print(f"3. Fetching specific prediction (ID: {prediction_ids[0]})...")
    print("="*70)
    response = requests.get(f"{BASE_URL}/history/{prediction_ids[0]}")
    if response.status_code == 200:
        pred = response.json()
        print("✅ Prediction details:")
        print(json.dumps(pred, indent=2))
    else:
        print(f"❌ Failed: {response.json()}")

# Test 4: Get statistics
print("\n" + "="*70)
print("4. Fetching statistics...")
print("="*70)
response = requests.get(f"{BASE_URL}/statistics")
if response.status_code == 200:
    stats = response.json()
    print("✅ Statistics:")
    print(f" Total Predictions: {stats['total_predictions']}")
    print(f" Approved: {stats['approved']}")
    print(f" Rejected: {stats['rejected']}")
    print(f" Approval rate: {stats['approval_rate']}")
    print(f" Avg confidence: {stats['average_confidence']}")
else:
    print(f"❌ Failed: {response.json()}")

# Test 5: Get analyticws
print("\n" + "="*70)
print("5. Fetching analyticss...")
print("="*70)
response = requests.get(f"{BASE_URL}/analytics")
if response.status_code == 200:
    analytics = response.json()
    print("✅ Analytics:")
    print(json.dumps(analytics, indent=2))
else:
    print(f"❌ Failed: {response.json()}")

# Test 6: Check API endpoint shows stats
print("\n" + "="*70)
print("6. Checking API endpoint...")
print("="*70)
response = requests.get(f"{BASE_URL}/api")
if response.status_code == 200:
    api_info = response.json()
    print("✅ API Info:")
    print(f" Status: {api_info['status']}")
    print(f" Version: {api_info['version']}")
    print(f" Model Accuracy: {api_info['model_accuracy']}")
    print(f" Statistics: {api_info['statistics']}")
else:
    print(f"❌ Failed: {response.json()}")

print("\n" + "="*70)
print("DATABASE INTEGRATION TESTS COMPLETE")
print("="*70)
print(f"\nDatabase file created: predictions.db")
print("You can inspect it with: sqlite3 predictions.db")
