import requests

BASE_URL = "https://loan-predictor-api-91xu.onrender.com"

# Make prediction
data = {"ApplicantIncome": 5000, "LoanAmount": 150, "Credit_History": 1}

response = requests.post(f"{BASE_URL}/predict", json=data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
