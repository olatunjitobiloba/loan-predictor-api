"""
Database models and operationns for Loan Prediction API
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
import json

db = SQLAlchemy()

class Prediction(db.Model):
    """Model for storing prediction history"""

    __tablename__ = 'predictions'

    # Primary key
    id = db.Column(db.Integer, primary_key=True)

    # Timestamp
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                          nullable=False)
    
    # Input features (numeric)
    applicant_income = db.Column(db.Float, nullable=False)
    coapplicant_income = db.Column(db.Float, default=0)
    loan_amount = db.Column(db.Float)
    loan_amount_term = db.Column(db.Float)
    credit_history = db.Column(db.Float)

    # Input features (categorical)
    gender = db.Column(db.String(10))
    married = db.Column(db.String(5))
    dependents = db.Column(db.String(5))
    education = db.Column(db.String(20))
    self_employed = db.Column(db.String(5))
    property_area = db.Column(db.String(20))

    # Prediction results
    prediction = db.Column(db.String(20), nullable=False) # Approved/Rejected
    prediction_code = db.Column(db.Integer, nullable=False) # 1/0
    confidence = db.Column(db.Float, nullable=False)
    probability_approved = db.Column(db.Float, nullable=False)
    probability_rejected = db.Column(db.Float, nullable=False)

    # Metadata
    model_version = db.Column(db.String(10), default='2.0')
    had_warnings = db.Column(db.Boolean, default=False)
    warnings = db.Column(db.Text) # JSON string of warnings

    # Request metadata
    ip_address = db.Column(db.String(50))

    def __repr__(self):
        return f'<Prediction {self.id}: {self.prediction}({self.confidence:.2%})>'
    
    def to_dict(self):
        """Convert prediction to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'input': {
                'applicant_income': self.applicant_income,
                'coapplicant_income': self.coapplicant_income,
                'loan_amount': self.loan_amount,
                'loan_amount_term': self.loan_amount_term,
                'credit_history': self.credit_history,
                'gender': self.gender,
                'married': self.married,
                'dependents': self.dependents,
                'education': self.education,
                'self_employed': self.self_employed,
                'property_area': self.property_area
            },
            'prediction': self.prediction,
            'confidence': self.confidence,
            'probability': {
                'approved': self.probability_approved,
                'rejected': self.probability_rejected
            },
            'model_version': self.model_version,
            'had_warnings': self.had_warnings,
            'warnings': json.loads(self.warnings) if self.warnings else []
        }
    
    @staticmethod
    def from_request(data, prediction_result, warnings=None, 
                     ip_address=None):
        """Create Prediction object from request data and prediction
        result"""
        return Prediction(
        # Input features
        applicant_income=data.get('ApplicantIncome'),
        coapplicant_income=data.get('CoapplicantIncome', 0),
        loan_amount=data.get('LoanAmount'),
        loan_amount_term=data.get('Loan_Amount_Term'),
        credit_history=data.get('Credit_History'),
        gender=data.get('Gender'),
        married=data.get('Married'),
        dependents=data.get('Dependents'),
        education=data.get('Education'),
        self_employed=data.get('Self_Employed'),
        property_area=data.get('Property_Area'),

        # Prediction results
        prediction=prediction_result['prediction'],
        prediction_code=prediction_result['prediction_code'],
        confidence=prediction_result['confidence'],
        probability_approved=prediction_result['probability']['approved'],
        probability_rejected=prediction_result['probability']['rejected'],

        # Metadata
        had_warnings=bool(warnings),
        warnings=json.dumps(warnings) if warnings else None,
        ip_address=ip_address
    )

class APIStats(db.Model):
    """Model for tracking API statisics"""

    __tablename__ = 'api_stats'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, default=lambda: datetime.now().date(),
                     unique=True, nullable=False)
    total_requests = db.Column(db.Integer, default=0)
    approved_count = db.Column(db.Integer, default=0)
    rejected_count = db.Column(db.Integer, default=0)
    avg_confidence = db.Column(db.Float, default=0.0)

    def __repr__(self):
        return f'<APIStats {self.date}: {self.total_requests} requests>'
    
    def to_dict(self):
        return {
            'date': self.date.isoformat(),
            'total_requests': self.total_requests,
            'approved_count': self.approved_count,
            'rejected_count': self.rejected_count,
            'approval_rate': f"{(self.approved_count / self.total_requests * 100):.2f}%" if self.total_requests > 0 else "0%",
            'avg_confidence': f"{self.avg_confidence:.2%}" 
        }
    
def init_db(app):
    """Initialize database with Flask app"""
    db.init_app(app)

    with app.app_context():
        # Create all tables
        db.create_all()
        # Avoid non-ASCII characters here to prevent Windows console encoding errors
        print("Database initialized")

def get_recent_predictions(limit=10):
    """Get recent predictions"""
    return Prediction.query.order_by(Prediction.timestamp.desc()).limit(limit).all()

def get_prediction_by_id(prediction_id):
    """Get specific prediction by ID"""
    return db.session.get(Prediction, prediction_id)

def get_predictions_by_date(start_date, end_date):
    """Get predictions within date range"""
    return Prediction.query.filter(
        Prediction.timestamp >= start_date,
        Prediction.timestamp <= end_date
    ).all()

def get_statistics():
    """Get overall statistics"""
    total = Prediction.query.count()
    approved = Prediction.query.filter_by(prediction='Approved').count()
    rejected = Prediction.query.filter_by(prediction='Rejected').count()

    avg_confidence = db.session.query(db.func.avg(Prediction.confidence)).scalar() or 0

    return {
        'total_predictions': total,
        'approved': approved,
        'rejected': rejected,
        'approval_rate': f"{(approved / total * 100):.2f}%" if total > 0
else "0%",
        'average_confidence': f"{avg_confidence:.2%}"
    }

def update_daily_stats(prediction_result):
    """Update daily statistics"""
    from datetime import date

    today = date.today()
    stats = APIStats.query.filter_by(date=today).first()

    if not stats:
        stats = APIStats(
            date=today,
            total_requests=0,
            approved_count=0,
            rejected_count=0,
            avg_confidence=0.0
        )
        db.session.add(stats)
    
    # Ensure fields are initialized
    if stats.total_requests is None:
        stats.total_requests = 0
    if stats.approved_count is None:
        stats.approved_count = 0
    if stats.rejected_count is None:
        stats.rejected_count = 0
    if stats.avg_confidence is None:
        stats.avg_confidence = 0.0
    
    stats.total_requests += 1

    if prediction_result['prediction'] == 'Approved':
        stats.approved_count += 1
    else:
        stats.rejected_count += 1

    # Update average confidence
    total_confidence = stats.avg_confidence * (stats.total_requests - 1) + prediction_result['confidence']
    stats.avg_confidence = total_confidence / stats.total_requests

    db.session.commit()

# Test database operations
if __name__ == '__main__':
    from flask import Flask

    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test_predictions.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    init_db(app)

    with app.app_context():
        # Test creating a prediction
        test_prediction = Prediction(
            applicant_income=5000,
            loan_amount=150,
            prediction='Approved',
            prediction_code=1,
            confidence=0.85,
            probability_approved=0.85,
            probability_rejected=0.15
        )

        db.session.add(test_prediction)
        db.session.commit()

        print(f"Created: {test_prediction}")

        # Test querying
        all_predictions = Prediction.query.all()
        print(f"\nTotal predictions: {len(all_predictions)}")

        # Test statistics
        stats = get_statistics()
        print(f"\nStatistics: {stats}")

        print("\n✅ Database tests passed!")