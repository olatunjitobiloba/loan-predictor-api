"""
Tests for database operations
"""

import pytest
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0,
               os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app_v4 import app, db, Prediction
from database import get_recent_predictions, get_statistics, update_daily_stats

@pytest.fixture
def test_app():
    """Create test app with database"""
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test_db.db'

    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()

    if os.path.exists('test_db.db'):
        os.remove('test_db.db')

class TestPredictModel:
    """Test Prediction model"""

    def test_create_prediction(self, test_app):
        """Test creating a prediction record"""
        with test_app.app_context():
            prediction=Prediction(
                applicant_income=5000,
                loan_amount=150,
                prediction='Approved',
                prediction_code=1,
                confidence=0.85,
                probability_approved=0.85,
                probability_rejected=0.15
            )
            db.session.add(prediction)
            db.session.commit()

            assert prediction.id is not None
            assert prediction.applicant_income == 5000
            assert prediction.prediction == 'Approved'
    
    def test_prediction_to_dict(self, test_app):
        """Test converting prediction to dictionary"""
        with test_app.app_context():
            prediction = Prediction(
                applicant_income=5000,
                prediction='Approved',
                prediction_code=1,
                confidence=0.85,
                probability_approved=0.85,
                probability_rejected=0.15
            )
            db.session.add(prediction)
            db.session.commit()

            pred_dict = prediction.to_dict()
            assert pred_dict['id'] == prediction.id
            assert pred_dict['prediction'] == 'Approved'
            assert 'input' in pred_dict
            assert 'probability' in pred_dict
    
    def test_prediction_from_request(self, test_app):
        """Test creating prediction from request data"""
        with test_app.app_context():
            request_data = {
                'ApplicantIncome': 5000,
                'LoanAmount': 150,
                'Credit_History': 1
            }

            prediction_result = {
                'prediction': 'Approved',
                'prediction_code': 1,
                'confidence': 0.85,
                'probability': {
                    'approved': 0.85,
                    'rejected': 0.15
                }
            }

            prediction = Prediction.from_request(
                request_data,
                prediction_result,
                warnings=['Test warning'],
                ip_address='127.0.0.1'
            )

            assert prediction.applicant_income == 5000
            assert prediction.prediction == 'Approved'
            assert prediction.had_warnings == True
            assert prediction.ip_address == '127.0.0.1'

class TestDatabaseOperations:
    """Test database helper functions"""

    def test_get_recent_predictions_empty(self, test_app):
        """Test getting recent predictions when database is empty"""
        with test_app.app_context():
            predictions = get_recent_predictions(10)
            assert len(predictions) == 0
    
    def test_get_recent_predictions(self, test_app):
        """Test getting recent predictions"""
        with test_app.app_context():
            # Create test predictions
            for i in range(5):
                prediction = Prediction(
                    applicant_income=5000 + i * 1000,
                    prediction = 'Approved',
                    prediction_code=1,
                    confidence=0.85,
                    probability_approved=0.85,
                    probability_rejected=0.15
                )
                db.session.add(prediction)
            db.session.commit()

            predictions = get_recent_predictions(3)
            assert len(predictions) == 3
    
    def test_get_statistics_empty(self, test_app):
        """Test statistics with empty database"""
        with test_app.app_context():
            stats = get_statistics()
            assert stats['total_predictions'] == 0
            assert stats['approved'] == 0
            assert stats['rejected'] == 0

    def test_get_statistics(self, test_app):
        """Test statistics calculation"""
        with test_app.app_context():
            # Create test predictions
            for i in range(10):
                prediction = Prediction(
                    applicant_income=5000,
                    prediction='Approved' if i < 7 else 'Rejected',
                    prediction_code=1 if i < 7 else 0,
                    confidence=0.85,
                    probability_approved=0.85 if i < 7 else 0.15,
                    probability_rejected=0.15 if i < 7 else 0.85
                )
                db.session.add(prediction)
            db.session.commit()

            stats = get_statistics()
            assert stats['total_predictions'] == 10
            assert stats['approved'] == 7
            assert stats['rejected'] == 3
            assert '70.00%' in stats['approval_rate']

# Run tests if this file is executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
