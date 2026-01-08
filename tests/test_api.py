"""
Integration tests for Flask API
"""

import json
import os

import pytest

from app_v4 import Prediction, app, db


@pytest.fixture
def client():
    """Create test client"""
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///test_predictions.db"

    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client
        with app.app_context():
            db.drop_all()

    # Clean up test database
    if os.path.exists("test_predictions.db"):
        os.remove("test_predictions.db")


@pytest.fixture
def valid_production_data():
    """Sample valid prediction data"""
    return {
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
        "Property_Area": "Urban",
    }


@pytest.fixture
def valid_prediction_data(valid_production_data):
    """Alias for valid_production_data"""
    return valid_production_data


class TESTAPIEndpoints:
    """Test suite for API endpoints"""

    def test_home_endpoint(self, client):
        """Test GET /api returns API information"""
        response = client.get("/api")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["version"] == "4.0"
        assert "endpoints" in data

    def test_health_endpoint(self, client):
        """Test GET /health returns healthy status"""
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)

        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_model_info_endpoint(self, client):
        """Test GET /model-info returns model details"""
        response = client.get("/model-info")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "accuracy" in data
        assert "features" in data
        assert data["feature_count"] > 0

    def test_validation_rules_endpoint(self, client):
        """Test GET /validation-rules returns validation info"""
        response = client.get("/validation-rules")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "required_fields" in data
        assert "ApplicantIncome" in data["required_fields"]

    def test_statistics_endpoint(self, client):
        """Test GET / staticstics returns stats"""
        response = client.get("/statistics")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "total_predictions" in data
        assert "approved" in data
        assert "rejected" in data

    def test_analytics_endpoint(self, client):
        """Test GET /analytics returns analytics"""
        response = client.get("/analytics")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "overall" in data
        assert "last_24_hours" in data


class TestPredictionEndpoint:
    """Test suite for /predict endpoint"""

    def test_predict_valid_complete_data(self, client, valid_prediction_data):
        response = client.post(
            "/predict",
            data=json.dumps(valid_prediction_data),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["prediction"] in ["Approved", "Rejected"]
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1
        assert "prediction_id" in data

    def test_predict_minimal_valid_data(self, client):
        minimal_data = {"ApplicantIncome": 5000}
        response = client.post(
            "/predict", data=json.dumps(minimal_data), content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["prediction"] in ["Approved", "Rejected"]

    def test_predict_no_data(self, client):
        response = client.post(
            "/predict", data=json.dumps({}), content_type="application/json"
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_predict_missing_required_field(self, client):
        """Test prediction without required field returns 400"""
        invalid_data = {"LoanAmount": 150}
        response = client.post(
            "/predict", data=json.dumps(invalid_data), content_type="application/json"
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_predict_invalid_data_type(self, client):
        """Test prediction with invalid data type returns 400"""
        invalid_data = {"ApplicantIncome": "not a number"}
        response = client.post(
            "/predict", data=json.dumps(invalid_data), content_type="application/json"
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "validation_errors" in data

    def test_predict_negative_income(self, client):
        """Test prediction with negative income returns 400"""
        invalid_data = {"ApplicantIncome": -5000}
        response = client.post(
            "/predict", data=json.dumps(invalid_data), content_type="application/json"
        )
        assert response.status_code == 400

    def test_predict_invalid_categorical(self, client):
        """Test prediction with invalid categorical value returns 400"""
        invalid_data = {"ApplicantIncome": 5000, "Gender": "Other"}
        response = client.post(
            "/predict", data=json.dumps(invalid_data), content_type="application/json"
        )
        assert response.status_code == 400

    def test_predict_with_warnings(self, client):
        """Test prediction that should generate warnings"""
        risky_data = {"ApplicantIncome": 800, "Credit_History": 0}
        response = client.post(
            "/predict", data=json.dumps(risky_data), content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "warnings" in data
        assert len(data["warnings"]) > 0

    def test_predict_stores_in_database(self, client, valid_prediction_data):
        """Test that prediction is stored in database"""
        response = client.post(
            "/predict",
            data=json.dumps(valid_prediction_data),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        prediction_id = data["prediction_id"]

        # Verify it's in database
        with app.app_context():
            prediction = db.session.get(Prediction, prediction_id)
            assert prediction is not None
            assert (
                prediction.applicant_income == valid_prediction_data["ApplicantIncome"]
            )


class TestHistoryEndpoints:
    """Test suite for history endpoints"""

    def test_history_empty(self, client):
        """Test /history with no predictions"""
        response = client.get("/history")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["count"] == 0
        assert data["predictions"] == []

    def test_history_with_predictions(self, client, valid_prediction_data):
        """Test /history after making predictions"""
        # Make a prediction first
        client.post(
            "/predict",
            data=json.dumps(valid_prediction_data),
            content_type="application/json",
        )

        # Get history
        response = client.get("/history")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["count"] == 1
        assert len(data["predictions"]) == 1

    def test_history_limit(self, client, valid_prediction_data):
        """Test /history with limit parameter"""
        # Make multiple predictions
        for _ in range(5):
            client.post(
                "/predict",
                data=json.dumps(valid_prediction_data),
                content_type="application/json",
            )

        # Get with limit
        response = client.get("/history?limit=3")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["count"] == 3

    def test_get_specific_prediction(self, client, valid_prediction_data):
        """Test GET /history/<id> for specific prediction"""
        # Make a prediction
        response = client.post(
            "/predict",
            data=json.dumps(valid_prediction_data),
            content_type="application/json",
        )
        prediction_id = json.loads(response.data)["prediction_id"]

        # Get specific prediction
        response = client.get(f"/history/{prediction_id}")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["id"] == prediction_id
        assert "prediction" in data

    def test_get_get_nonexistent_prediction(self, client):
        """Test GET /history/<id> for non-existent prediction"""
        response = client.get("/history/99999")
        assert response.status_code == 404
        data = json.loads(response.data)
        assert "error" in data


# Run tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
