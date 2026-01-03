import json
import pytest

from app_v7 import app


@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


def test_predict_endpoint_returns_expected_structure(client):
    payload = {
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 0,
        "LoanAmount": 100,
        "Credit_History": 1,
        "Education": "Graduate"
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, dict)
    # basic fields
    assert "prediction" in data
    assert "probability" in data or "probabilities" in data
    # Ensure the response contains input_data or model_info or ensemble summary
    assert any(k in data for k in ("input_data", "model_info", "ensemble", "results"))


def test_benchmark_endpoint_runs(client):
    resp = client.post("/models/benchmark", json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, dict)
    # should include results or consensus
    assert "results" in data or "consensus" in data
