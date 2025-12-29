from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
model = None
feature_names = []
model_info = {}
prediction_count = 0
prediction_history = []

# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model():
    """Load the trained model and metadata on startup"""
    global model, feature_names, model_info

    try:
        # Load model
        model_path = 'models/loan_model_v2.pkl'
        model = joblib.load(model_path)
        print(f"✅ Model loaded from {model_path}")

        # Load feature names
        with open('models/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        print(f"✅ Loaded {len(feature_names)} feature names")
        
        # Load model info
        with open('models/model_info.json', 'r') as f:
            model_info = json.load(f)
        print(f"✅ Model accuracy: {model_info['accuracy']:.2%}")

        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False


# ============================================================================
# PREPROCESSING
# ============================================================================
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
    if 'LoanAmount' not in df.columns or pd.isna(df['LoanAmount'].iloc[0]):
        df['LoanAmount'] = 128.0  # Training median
    
    if 'Loan_Amount_Term' not in df.columns or pd.isna(df['Loan_Amount_Term'].iloc[0]):
        df['Loan_Amount_Term'] = 360.0  # Training median

    if 'Credit_History' not in df.columns or pd.isna(df['Credit_History'].iloc[0]):
        df['Credit_History'] = 1.0  # Training mode
    
    # CoapplicantIncome - default to 0
    if 'CoapplicantIncome' not in df.columns:
        df['CoapplicantIncome'] = 0

    # Categorical defaults
    if 'Gender' not in df.columns:
        df['Gender'] = 'Male'
    if 'Married' not in df.columns:
        df['Married'] = 'Yes'
    if 'Dependents' not in df.columns:
        df['Dependents'] = '0'
    if 'Education' not in df.columns:
        df['Education'] = 'Graduate'
    if 'Self_Employed' not in df.columns:
        df['Self_Employed'] = 'No'
    if 'Property_Area' not in df.columns:
        df['Property_Area'] = 'Urban'
    
    # ============================================================================
    # FEATURE ENGINEERING (MUST MATCH TRAINING)
    # ============================================================================
    
    # Total Income
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

    # Income to Loan Ratio
    df['Income_Loan_Ratio'] = df['TotalIncome'] / (df['LoanAmount'] + 1)

    # Loan Amount per Term
    df['Loan_Amount_Per_Term'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + 1)

    # EMI (Estimated Monthly Installment)
    df['EMI'] = df['LoanAmount'] / (df['Loan_Amount_Term'] / 12 + 1)

    # Balance Income (Income after EMI)
    df['Balance_Income'] = df['TotalIncome'] - (df['EMI'] * 1000)

    # Log transformations for skewed features
    df['Log_ApplicantIncome'] = np.log1p(df['ApplicantIncome'])
    df['Log_CoapplicantIncome'] = np.log1p(df['CoapplicantIncome'])
    df['Log_LoanAmount'] = np.log1p(df['LoanAmount'])
    df['Log_TotalIncome'] = np.log1p(df['TotalIncome'])

    # ============================================================================
    # ENCODE CATEGORICAL VARIABLES
    # ============================================================================
    
    # Map Gender
    gender_map = {'Male': 1, 'Female': 0}
    df['Gender_Encoded'] = df['Gender'].map(gender_map).fillna(1)

    # Map Married
    married_map = {'Yes': 1, 'No': 0}
    df['Married_Encoded'] = df['Married'].map(married_map).fillna(1)

    # Map Dependents
    df['Dependents'] = df['Dependents'].replace('3+', '3')
    df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').fillna(0)
    df['Dependents_Encoded'] = df['Dependents']

    # Map Education
    education_map = {'Graduate': 1, 'Not Graduate': 0}
    df['Education_Encoded'] = df['Education'].map(education_map).fillna(1)

    # Map Self_Employed
    self_employed_map = {'Yes': 1, 'No': 0}
    df['Self_Employed_Encoded'] = df['Self_Employed'].map(self_employed_map).fillna(0)

    # Map Property_Area
    property_map = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
    df['Property_Area_Encoded'] = df['Property_Area'].map(property_map).fillna(2)

    # ============================================================================
    # SELECT ONLY REQUIRED FEATURES IN CORRECT ORDER
    # ============================================================================
    
    df = df[feature_names]

    return df


# ============================================================================
# WEB ROUTES (HTML PAGES)
# ============================================================================
@app.route("/")
@app.route("/home")
def home():
    """Home page with API documentation"""
    return render_template('home.html', title='Home')


@app.route("/about")
def about():
    """About page"""
    return render_template('about.html', title='About')


# ============================================================================
# API INFORMATION ENDPOINTS
# ============================================================================
@app.route("/api")
def api_info():
    """API information endpoint"""
    return jsonify({
        "name": "Loan Predictor API",
        "version": "2.0",
        "status": "running",
        "model_loaded": model is not None,
        "model_accuracy": model_info.get('accuracy', 'N/A'),
        "endpoints": {
            "/": "Home page",
            "/api": "API information",
            "/health": "Health check",
            "/predict": "Make loan prediction (POST)",
            "/validate-loan": "Validate loan data (POST)",
            "/model-info": "Model details",
            "/about": "About page"
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


@app.route('/model-info')
def model_info_endpoint():
    """Return model information"""
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "accuracy": model_info.get('accuracy'),
        "precision": model_info.get('precision'),
        "recall": model_info.get('recall'),
        "f1_score": model_info.get('f1_score'),
        "features": feature_names,
        "feature_count": len(feature_names)
    })

@app.route('/stats')
def stats():
    """API usage statistics"""
    return jsonify({
        "total_predictions": prediction_count,
        "model_accuracy": model_info.get('accuracy'),
        "uptime": "Running",
        "recent_predictions": len(prediction_history)
    })

# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================
@app.route('/predict', methods=['POST'])      
def predict():
    """
    Predict loan approval

    Expected JSON format:
    {
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
    """
    logger.info(f"Prediction request received from {request.remote_addr}")
    global prediction_count, prediction_history
    try:
        # Check if model is loaded
        if not model:
            return jsonify({
                "error": "Model not loaded",
                "message": "Server error - please contact administrator"
            }), 500
        
        # Get JSON data
        try:
            data = request.get_json(force=True)
        except Exception as e:
            return jsonify({
                "error": "Invalid JSON in request body",
                "message": "Please ensure your request body contains valid JSON",
                "details": str(e)
            }), 400

        # Validate input
        if not data:
            return jsonify({
                "error": "No data provided",
                "message": "Please send JSON data in request body",
                "example": {
                    "ApplicantIncome": 5000,
                    "LoanAmount": 150,
                    "Credit_History": 1
                }
            }), 400
        
        # Check required fields
        required_fields = ['ApplicantIncome']
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields,
                "received_fields": list(data.keys()),
                "message": "ApplicantIncome is required"
            }), 400
        
        # Preprocess input
        try:
            df_processed = preprocess_input(data)
        except Exception as e:
            return jsonify({
                "error": "Preprocessing error",
                "message": str(e)
            }), 400
        
        # Make prediction
        prediction = model.predict(df_processed)[0]
        prediction_proba = model.predict_proba(df_processed)[0]

        # Prepare response
        result = {
            "prediction": "Approved" if prediction == 1 else "Rejected",
            "prediction_code": int(prediction),
            "confidence": float(max(prediction_proba)),
            "probability": {
                "rejected": float(prediction_proba[0]),
                "approved": float(prediction_proba[1])
            },
            "input_data": data,
            "model_accuracy": model_info.get('accuracy'),
            "message": "Prediction successful"
        }

        prediction_count += 1
        prediction_history.append({
            "timestamp": datetime.now().isoformat(),
            "prediction": result['prediction'],
            "confidence": result['confidence']
        })

        # Keep only last 100 predictions
        if len(prediction_history) > 100:
            prediction_history.pop(0)
        logger.info(f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.2%}")

        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route("/validate-loan", methods=["POST"])
def validate_loan():
    """Validate loan application data before prediction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract and validate fields
        age = data.get("age")
        income = data.get("income")
        loan_amount = data.get("loan_amount")
        
        # Validation checks
        errors = []
        
        if age is not None:
            if not isinstance(age, int) or age < 18 or age > 100:
                errors.append("age must be an integer between 18 and 100")
        
        if income is not None:
            if not isinstance(income, (int, float)) or income < 0:
                errors.append("income must be a positive number")
        
        if loan_amount is not None:
            if not isinstance(loan_amount, (int, float)) or loan_amount < 0:
                errors.append("loan_amount must be a positive number")
        
        # Return errors if validation failed
        if errors:
            return jsonify({"error": "Validation failed", "errors": errors}), 400
        
        # If validation passes
        return jsonify({
            "status": "valid",
            "message": "Loan application validated successfully",
            "data": data
        })
        
    except Exception as e:
        return jsonify({"error": "Server error", "message": str(e)}), 500


# ============================================================================
# UTILITY ENDPOINTS (FOR TESTING/DEBUGGING)
# ============================================================================
@app.route("/search", methods=["GET"])
def search():
    """Example endpoint for query parameters"""
    query = request.args.get("q")
    limit = request.args.get("limit", default=10, type=int)
    page = request.args.get("page", 1, type=int)
    
    return jsonify({
        "query": query,
        "limit": limit,
        "page": page,
        "message": f"Searching for '{query}' with limit {limit} on page {page}"
    })


@app.route("/headers", methods=["GET", "POST"])
def show_headers():
    """Show request headers (for debugging)"""
    user_agent = request.headers.get("User-Agent")
    content_type = request.headers.get("Content-Type")
    all_headers = dict(request.headers)
    
    return jsonify({
        "user_agent": user_agent,
        "content_type": content_type,
        "all_headers": all_headers
    })


@app.route("/request-info", methods=["GET", "POST"])
def request_info():
    """Show request metadata (for debugging)"""
    return jsonify({
        "method": request.method,
        "url": request.url,
        "path": request.path,
        "remote_addr": request.remote_addr,
        "is_json": request.is_json,
        "content_type": request.content_type,
        "content_length": request.content_length,
    })


# ============================================================================
# APPLICATION STARTUP
# ============================================================================
# Load model when app starts
with app.app_context():
    if not load_model():
        print("⚠️ WARNING: Model failed to load. API will not work properly.")


if __name__ == '__main__':
    app.run(debug=True, port=5000)