import json
import urllib.request

payload = {
    "ApplicantIncome": 10000,
    "CoapplicantIncome": 0,
    "LoanAmount": 9998,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "Property_Area": "Urban",
}
req = urllib.request.Request(
    "http://127.0.0.1:5000/predict",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
try:
    resp = urllib.request.urlopen(req, timeout=10).read().decode()
    print(resp)
except Exception as e:
    print("ERROR", e)
