import json
import logging
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from database import (
    Prediction,
    db,
    get_recent_predictions,
    get_statistics,
    init_db,
    update_daily_stats,
)
from validators import LoanApplicationValidator

load_dotenv()

app = Flask(__name__)

# Configure database (use environment variable or default to SQLite)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:///predictions.db"
)

# Handle PostgreSQL URL format (Render uses postgresql://, SQLAlchemy needs postgresql+psycopg2://)
if app.config["SQLALCHEMY_DATABASE_URI"].startswith("postgres://"):
    app.config["SQLALCHEMY_DATABASE_URI"] = app.config[
        "SQLALCHEMY_DATABASE_URI"
    ].replace("postgres://", "postgresql+psycopg2://", 1)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "dev-secret-key-change-in-production"
)
# Initialize database
init_db(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("api.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Global variables
model = None
feature_names = []
model_info = {}
validator = LoanApplicationValidator()


def load_model():
    """Load the trained model and metadata on startup"""
    global model, feature_names, model_info

    try:
        model_path = "models/loan_model_v2.pkl"
        model = joblib.load(model_path)
        logger.info(f"[OK] Model loaded from {model_path}")

        with open("models/feature_names.txt", "r") as f:
            feature_names = [line.strip() for line in f.readlines()]
        logger.info(f"[OK] Loaded {len(feature_names)} feature names")

        with open("models/model_info.json", "r") as f:
            model_info = json.load(f)
        logger.info(f"[OK] Model accuracy: {model_info['accuracy']:.2%}")

        return True
    except Exception as e:
        logger.error(f"[ERROR] Error loading model: {str(e)}")
        return False


def preprocess_input(data):
    """
    Preprocess input data to match training data format

    Parameters:
    - data: Dictionary with loan application details

    Returns:
    - DataFrame ready for prediction
    """
    # Create DataFrame from input
    df = pd.DataFrame([data])

    # Handle missing values (use same logic as training)
    # Numeric columns - fill with median (use training medians)
    if "LoanAmount" not in df.columns or pd.isna(df["LoanAmount"].iloc[0]):
        df["LoanAmount"] = 128.0  # Training median

    if "Loan_Amount_Term" not in df.columns or pd.isna(df["Loan_Amount_Term"].iloc[0]):
        df["Loan_Amount_Term"] = 360.0  # Training median

    if "Credit_History" not in df.columns or pd.isna(df["Credit_History"].iloc[0]):
        df["Credit_History"] = 1.0  # Training mode

    # CoapplicantIncome - default to 0
    if "CoapplicantIncome" not in df.columns:
        df["CoapplicantIncome"] = 0

    # Categorical defaults
    if "Gender" not in df.columns:
        df["Gender"] = "Male"
    if "Married" not in df.columns:
        df["Married"] = "Yes"
    if "Dependents" not in df.columns:
        df["Dependents"] = "0"
    if "Education" not in df.columns:
        df["Education"] = "Graduate"
    if "Self_Employed" not in df.columns:
        df["Self_Employed"] = "No"
    if "Property_Area" not in df.columns:
        df["Property_Area"] = "Urban"

    # ============================================================================
    # FEATURE ENGINEERING (MUST MATCH TRAINING)
    # ============================================================================

    # Total Income
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

    # Income to Loan Ratio
    df["Income_Loan_Ratio"] = df["TotalIncome"] / (df["LoanAmount"] + 1)

    # Loan Amount per Term
    df["Loan_Amount_Per_Term"] = df["LoanAmount"] / (df["Loan_Amount_Term"] + 1)

    # EMI (Estimated Monthly Installment)
    df["EMI"] = df["LoanAmount"] / (df["Loan_Amount_Term"] / 12 + 1)

    # Balance Income (Income after EMI)
    df["Balance_Income"] = df["TotalIncome"] - (df["EMI"] * 1000)

    # Log transformations for skewed features
    df["Log_ApplicantIncome"] = np.log1p(df["ApplicantIncome"])
    df["Log_CoapplicantIncome"] = np.log1p(df["CoapplicantIncome"])
    df["Log_LoanAmount"] = np.log1p(df["LoanAmount"])
    df["Log_TotalIncome"] = np.log1p(df["TotalIncome"])

    # ============================================================================
    # ENCODE CATEGORICAL VARIABLES
    # ============================================================================

    # Map Gender
    gender_map = {"Male": 1, "Female": 0}
    df["Gender_Encoded"] = df["Gender"].map(gender_map).fillna(1)

    # Map Married
    married_map = {"Yes": 1, "No": 0}
    df["Married_Encoded"] = df["Married"].map(married_map).fillna(1)

    # Map Dependents
    df["Dependents"] = df["Dependents"].replace("3+", "3")
    df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce").fillna(0)
    df["Dependents_Encoded"] = df["Dependents"]

    # Map Education
    education_map = {"Graduate": 1, "Not Graduate": 0}
    df["Education_Encoded"] = df["Education"].map(education_map).fillna(1)

    # Map Self_Employed
    self_employed_map = {"Yes": 1, "No": 0}
    df["Self_Employed_Encoded"] = df["Self_Employed"].map(self_employed_map).fillna(0)

    # Map Property_Area
    property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    df["Property_Area_Encoded"] = df["Property_Area"].map(property_map).fillna(2)

    # ============================================================================
    # SELECT ONLY REQUIRED FEATURES IN CORRECT ORDER
    # ============================================================================
    # Create additional alias columns expected by older feature lists
    # Some versions expect underscored names and one-hot property flags
    try:
        df["Total_Income"] = df["TotalIncome"]
    except Exception:
        df["Total_Income"] = df.get(
            "TotalIncome", df["ApplicantIncome"] + df["CoapplicantIncome"]
        )

    # Loan to income / income-loan ratio naming variants
    if "Income_Loan_Ratio" in df.columns:
        df["Loan_to_Income_Ratio"] = df["Income_Loan_Ratio"]
    else:
        df["Loan_to_Income_Ratio"] = df["Total_Income"] / (df["LoanAmount"] + 1)

    # Income per dependent (safeguard)
    if "Income_per_Dependent" not in df.columns:
        df["Income_per_Dependent"] = df["Total_Income"] / (df["Dependents"] + 1)

    # One-hot property flags (older feature sets)
    df["Property_Semiurban"] = (df["Property_Area_Encoded"] == 1).astype(int)
    df["Property_Urban"] = (df["Property_Area_Encoded"] == 2).astype(int)

    df = df[feature_names]

    return df


@app.route("/")
@app.route("/home")
@app.route("/app")
def frontend():
    """Serve frontend application"""
    return render_template("index.html")


@app.route("/api")
def api_info():
    """API information endpoint"""
    stats = get_statistics()

    return jsonify(
        {
            "name": "Loan Predictor API",
            "version": "4.0",
            "status": "running",
            "model_loaded": model is not None,
            "model_accuracy": f"{model_info.get('accuracy', 0):.2%}",
            "database": "connected",
            "statistics": stats,
            "features": {
                "validation": "Comprehensive input validation",
                "database": "Prediction history storage",
                "analytics": "Usage statistics and trends",
            },
            "endpoints": {
                "/": "Home page",
                "/home": "Home page",
                "/about": "About page",
                "/api": "API information",
                "/health": "Health check",
                "/predict": "Make loan prediction (POST)",
                "/validate-loan": "Validate loan data (POST)",
                "/history": "Recent predictions",
                "/history/<id>": "Specific prediction",
                "/statistics": "Overall statistics",
                "/analytics": "Detailed analytics",
                "/model-info": "Model details",
                "/validation-rules": "Input validation rules",
            },
        }
    )


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": model is not None,
            "database": "connected",
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Predict loan approval and store in database"""

    try:
        logger.info(f"Prediction request from {request.remote_addr}")

        if not model:
            logger.error("Model not loaded")
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()

        if not data:
            return (
                jsonify(
                    {
                        "error": "No data provided",
                        "message": "Please send JSON data in request body",
                    }
                ),
                400,
            )

        # Validate input
        is_valid, errors, warnings = validator.validate_loan_application(data)

        if not is_valid:
            logger.warning(f"Validation failed: {errors}")
            return (
                jsonify({"error": "Validation failed", "validation_errors": errors}),
                400,
            )

        if warnings:
            logger.info(f"Validation warnings: {warnings}")

        # Preprocess and prepare model input
        df_processed = preprocess_input(data)

        # Ensure columns match the model's expected features. If the model exposes
        # `feature_names_in_`, reindex to that order and fill missing with 0.
        try:
            if hasattr(model, "feature_names_in_"):
                expected = list(model.feature_names_in_)
                model_input = df_processed.reindex(columns=expected, fill_value=0)
            else:
                model_input = df_processed

            prediction = model.predict(model_input)[0]
            prediction_proba = model.predict_proba(model_input)[0]
        except Exception:
            # Fallback: try passing raw processed frame values
            prediction = model.predict(df_processed.values)[0]
            prediction_proba = model.predict_proba(df_processed.values)[0]

        # Prepare result
        result = {
            "success": True,
            "prediction": "Approved" if prediction == 1 else "Rejected",
            "prediction_code": int(prediction),
            "confidence": float(max(prediction_proba)),
            "probability": {
                "rejected": float(prediction_proba[0]),
                "approved": float(prediction_proba[1]),
            },
            "input_data": data,
            "model_info": {"accuracy": model_info.get("accuracy"), "version": "2.0"},
            "timestamp": datetime.now().isoformat(),
        }

        if warnings:
            result["warnings"] = warnings

        # Store in database
        try:
            prediction_record = Prediction.from_request(
                data, result, warnings, request.remote_addr
            )
            db.session.add(prediction_record)
            db.session.commit()

            result["prediction_id"] = prediction_record.id

            # Update daily statistics
            update_daily_stats(result)

            logger.info(
                f"Prediction stored: ID={prediction_record.id},Result={result['prediction']}"
            )
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            # Continue even if database fails
            result["database_warning"] = "Prediction not stored in database"

        logger.info(
            f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.2%}"
        )

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error", "message": str(e)}), 500


@app.route("/validate-loan", methods=["POST"])
def validate_loan():
    """Validate loan application data before prediction"""
    logger.info(f"Validation request received from {request.remote_addr}")

    try:
        # Get JSON data
        try:
            data = request.get_json(force=True)
        except Exception as e:
            return (
                jsonify(
                    {
                        "error": "Invalid JSON in request body",
                        "message": "Please ensure your request body contains valid JSON",
                        "details": str(e),
                    }
                ),
                400,
            )

        if not data:
            return (
                jsonify(
                    {
                        "error": "No data provided",
                        "message": "Please send JSON data in request body",
                    }
                ),
                400,
            )

        # Validate using the validator
        is_valid, errors, warnings = validator.validate_loan_application(data)

        # Build response
        response = {
            "valid": is_valid,
            "timestamp": datetime.now().isoformat(),
            "input_data": data,
        }

        if errors:
            response["errors"] = errors
            response["message"] = "Validation failed - please correct the errors"

        if warnings:
            response["warnings"] = warnings

        if is_valid and not warnings:
            response["message"] = "Data is valid and ready for prediction"
        elif is_valid and warnings:
            response["message"] = "Data is valid but has warnings"

        # Log result
        if is_valid:
            logger.info(f"Validation successful with {len(warnings)} warnings")
        else:
            logger.warning(f"Validation failed: {errors}")

        status_code = 200 if is_valid else 400
        return jsonify(response), status_code

    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return (
            jsonify({"error": "Server error during validation", "message": str(e)}),
            500,
        )


@app.route("/history")
def history():
    """Get recent prediction history"""
    try:
        limit = request.args.get("limit", 10, type=int)
        limit = min(limit, 100)  # Max 100 records

        predictions = get_recent_predictions(limit)

        return (
            jsonify(
                {
                    "count": len(predictions),
                    "predictions": [p.to_dict() for p in predictions],
                }
            ),
            200,
        )
    except Exception as e:
        logger.error(f"History error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/history/<int:prediction_id>")
def get_prediction(prediction_id):
    """Get specific prediction by ID"""
    try:
        prediction = Prediction.query.get(prediction_id)

        if not prediction:
            return jsonify({"error": "Prediction not found"}), 404

        return jsonify(prediction.to_dict()), 200
    except Exception as e:
        logger.error(f"Get prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/statistics")
def statistics():
    """Get overall statistics"""
    try:
        stats = get_statistics()
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"Statistics error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/analytics")
def analytics():
    """Get detailed analytics"""
    try:
        # Overall stats
        stats = get_statistics()

        # Recent trends (last 24 hours)
        yesterday = datetime.now() - timedelta(days=1)
        recent = Prediction.query.filter(Prediction.timestamp >= yesterday).all()

        recent_approved = sum(1 for p in recent if p.prediction == "Approved")
        recent_rejected = len(recent) - recent_approved

        # Average confidence by prediction type
        approved_predictions = Prediction.query.filter_by(prediction="Approved").all()
        rejected_predictions = Prediction.query.filter_by(prediction="Rejected").all()

        avg_confidence_approved = (
            np.mean([p.confidence for p in approved_predictions])
            if approved_predictions
            else 0
        )
        avg_confidence_rejected = (
            np.mean([p.confidence for p in rejected_predictions])
            if rejected_predictions
            else 0
        )

        return (
            jsonify(
                {
                    "overall": stats,
                    "last_24_hours": {
                        "total": len(recent),
                        "approved": recent_approved,
                        "rejected": recent_rejected,
                        "approval_rate": (
                            f"{(recent_approved / len(recent) * 100):.2f}%"
                            if recent
                            else "0%"
                        ),
                    },
                    "confidence_analysis": {
                        "avg_confidence_approved": f"{avg_confidence_approved:.2%}",
                        "avg_confidence_rejected": f"{avg_confidence_rejected:.2%}",
                    },
                }
            ),
            200,
        )
    except Exception as e:
        logger.error(f"Analytics error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/model-info")
def model_info_endpoint():
    """Return mode information"""
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    return jsonify(
        {
            "accuracy": model_info.get("accuracy"),
            "precision": model_info.get("precision"),
            "recall": model_info.get("recall"),
            "f1_score": model_info.get("f1_score"),
            "features": feature_names,
            "feature_count": len(feature_names),
        }
    )


@app.route("/validation-rules")
def validation_rules():
    """Return validation rules"""
    return jsonify(
        {
            "required_fields": ["ApplicantIncome"],
            "optional_fields": [
                "CoapplicantIncome",
                "LoanAmount",
                "Loan_Amount_Term",
                "Credit_History",
                "Gender",
                "Married",
                "Dependents",
                "Education",
                "Self_Employed",
                "Property_Area",
            ],
            "valid_values": {
                "Gender": validator.VALID_GENDER,
                "Married": validator.VALID_MARRIED,
                "Dependents": validator.VALID_DEPENDENTS,
                "Education": validator.VALID_EDUCATION,
                "Self_Employed": validator.VALID_SELF_EMPLOYED,
                "Property_Area": validator.VALID_PROPERTY_AREA,
            },
            "ranges": {
                "ApplicantIncome": f"{validator.MIN_INCOME} - {validator.MAX_INCOME}",
                "CoapplicantIncome": f"{validator.MIN_INCOME} - {validator.MAX_INCOME}",
                "LoanAmount": f"{validator.MIN_LOAN_AMOUNT} - {validator.MAX_LOAN_AMOUNT}",
                "Credit_History": "0 or 1",
                "Loan_Amount_Term": validator.VALID_LOAN_TERMS,
            },
        }
    )


@app.route("/search", methods=["GET"])
def search():
    """Example endpoint for query parameters"""
    query = request.args.get("q")
    limit = request.args.get("limit", default=10, type=int)
    page = request.args.get("page", 1, type=int)

    return jsonify(
        {
            "query": query,
            "limit": limit,
            "page": page,
            "message": f"Searching for '{query}' with limit {limit} on page {page}",
        }
    )


@app.route("/headers", methods=["GET", "POST"])
def show_headers():
    """Show request headers (for debugging)"""
    user_agent = request.headers.get("User-Agent")
    content_type = request.headers.get("Content-Type")
    all_headers = dict(request.headers)

    return jsonify(
        {
            "user_agent": user_agent,
            "content_type": content_type,
            "all_headers": all_headers,
        }
    )


@app.route("/request-info", methods=["GET", "POST"])
def request_info():
    """Show request metadata (for debugging)"""
    return jsonify(
        {
            "method": request.method,
            "url": request.url,
            "path": request.path,
            "remote_addr": request.remote_addr,
            "is_json": request.is_json,
            "content_type": request.content_type,
            "content_length": request.content_length,
        }
    )


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return (
        jsonify(
            {
                "error": "Endpoint not found",
                "message": "The requested endpoint does not exist",
                "available_endpoints": [
                    "/",
                    "/home",
                    "/about",
                    "/api",
                    "/health",
                    "/predict",
                    "/validate-loan",
                    "/history",
                    "/statistics",
                    "/analytics",
                    "/model-info",
                    "/validation-rules",
                ],
            }
        ),
        404,
    )


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return (
        jsonify(
            {
                "error": "Method not allowed",
                "message": "This endpoint does not support the requested HTTP method",
            }
        ),
        405,
    )


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return (
        jsonify(
            {
                "error": "Internal server error",
                "message": "An unexpected error occurred",
            }
        ),
        500,
    )


# Load model on startup
with app.app_context():
    if not load_model():
        logger.warning("[WARNING] Model failed to load")

if __name__ == "__main__":
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get("PORT", 5000))

    # Run in production mode
    app.run(host="0.0.0.0", port=port, debug=False)
