# Loan Predictor API

A production-ready machine learning API for predicting loan approval using Random Forest classification. Built with Flask, featuring comprehensive data validation, feature engineering, and a web interface.

## 🎯 Project Overview

This project implements an end-to-end machine learning solution for loan approval prediction. It includes:
- A trained Random Forest model with **88.62% accuracy**
- RESTful API for making predictions
- Comprehensive input validation
- Web interface for easy interaction
- Complete data preprocessing and feature engineering pipeline
- Exploratory data analysis notebooks
- Unit tests and validation test suite

**Current Model Performance: 88.62% Accuracy | 89% Precision | 95.3% Recall | 92% F1-Score**

## ✨ Key Features

- ✅ **RESTful API** - Flask-based API with proper error handling
- ✅ **Input Validation** - Comprehensive validation with detailed error messages
- ✅ **Feature Engineering** - 20 engineered features including income ratios, EMI calculations, and log transformations
- ✅ **Web Interface** - Bootstrap 5 responsive web UI
- ✅ **Data Preprocessing** - Full preprocessing pipeline with encoding and normalization
- ✅ **Model Serving** - Random Forest model with optimized hyperparameters
- ✅ **Health Checks** - Endpoint monitoring and API status
- ✅ **EDA & Notebooks** - Exploratory analysis and model training notebooks
- ✅ **Test Suite** - Comprehensive validation and integration tests

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Input Validation](#input-validation)
- [Model Details](#model-details)
- [Feature Engineering](#feature-engineering)
- [Usage Examples](#usage-examples)
- [Testing](#testing)
- [Data Files](#data-files)
- [Technical Stack](#technical-stack)
- [Development](#development)

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/olatunjitobiloba1/loan-predictor-api
cd loan-predictor-api

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
The project uses the following dependencies:
- Flask 3.1.2 - Web framework
- scikit-learn 1.8.0 - Machine learning
- pandas 2.3.3 - Data manipulation
- numpy 2.4.0 - Numerical computing
- joblib 1.5.3 - Model serialization

See [requirements.txt](requirements.txt) for the complete list.

## 🚀 Quick Start

```bash
# Run the application
python app.py

# Access the web interface
# Open browser to http://localhost:5000
```

The API will be available at `http://localhost:5000`

## 📁 Project Structure

```
loan-predictor-api/
├── app.py                    # Main Flask application (v1)
├── app_v2.py                 # Flask app with enhanced validation (v2)
├── app_v2_improved.py        # Improved version with better error handling (v2+)
├── app_v3.py                 # Latest version with additional features (v3)
├── preprocess.py             # Data preprocessing pipeline
├── validators.py             # Input validation logic
├── train_model_v3.py         # Model training script
├── test_api.py               # API integration tests
├── test_validation.py        # Validation test suite
├── requirements.txt          # Project dependencies
├── README.md                 # This file
│
├── models/
│   ├── model_info.json       # Model metadata and performance metrics
│   ├── feature_names.txt     # Feature names used by the model
│   └── submission.csv        # Model predictions on test set
│
├── data/
│   ├── train_u6lujuX_CVtuZ9i.csv    # Training dataset
│   ├── test_Y3wMUE5_7gLdaTN.csv     # Test dataset
│   ├── test_predictions.csv          # Model predictions on test data
│   └── data_summary.txt              # Data summary statistics
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA and data analysis
│   ├── 02_model_training.ipynb       # Model training and evaluation
│   ├── 03_feature_engineering.ipynb  # Feature engineering techniques
│   └── explore_data.ipynb            # Additional data exploration
│
├── templates/
│   ├── home.html             # Home page template
│   ├── about.html            # About page template
│   └── layout.html           # Base layout template
│
├── static/
│   └── main.css              # Stylesheet
│
├── visualizations/
│   └── eda_plots/            # Generated EDA visualizations
│
└── screenshots/              # Screenshots of API usage
    ├── API_GET_RESPONSE_IN_POSTMAN/
    └── API_POST_RESPONSE IN POSTMAN/
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 88.62% |
| **Precision** | 89.01% |
| **Recall** | 95.29% |
| **F1-Score** | 92.05% |
| **Cross-Validation Mean** | 78.41% |
| **Cross-Validation Std** | 2.40% |

### Model Configuration
- **Algorithm**: Random Forest Classifier
- **Number of Estimators**: 200
- **Max Depth**: 15
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2
- **Training Samples**: 491
- **Validation Samples**: 123

## 🔌 API Endpoints

### GET /
Home page with web interface.

**Response**: HTML page with prediction form

---

### GET /api
API information endpoint.

**Response:**
```json
{
  "message": "Loan Predictor API",
  "version": "1.0",
  "status": "running"
}
```

---

### GET /health
Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy"
}
```

---

### GET /about
About page with project information.

**Response**: HTML page with project details

---

### POST /predict
Make a loan approval prediction.

**Required Fields:**
- `ApplicantIncome` (number) - Applicant's income

**Optional Fields:**
- `CoapplicantIncome` (number) - Co-applicant's income
- `LoanAmount` (number) - Loan amount requested
- `Loan_Amount_Term` (number) - Term of the loan (in months)
- `Credit_History` (0 or 1) - Credit history availability
- `Gender` (string) - "Male" or "Female"
- `Married` (string) - "Yes" or "No"
- `Dependents` (string) - "0", "1", "2", or "3+"
- `Education` (string) - "Graduate" or "Not Graduate"
- `Self_Employed` (string) - "Yes" or "No"
- `Property_Area` (string) - "Urban", "Semiurban", or "Rural"

**Request Example:**
```json
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
```

**Success Response (200):**
```json
{
  "prediction": "Approved",
  "confidence": 0.89,
  "received_data": {...},
  "message": "Prediction successful"
}
```

**Error Response (400):**
```json
{
  "error": "Validation Error",
  "message": "ApplicantIncome must be positive",
  "received_data": {...}
}
```

## ✅ Input Validation

The API includes comprehensive input validation with the following rules:

### Numeric Fields
- **ApplicantIncome**: Required, range 0-100,000
- **CoapplicantIncome**: Optional, range 0-100,000
- **LoanAmount**: Optional, range 0-10,000
- **Loan_Amount_Term**: Optional, valid values [12, 36, 60, 84, 120, 180, 240, 300, 360, 480]

### Categorical Fields
- **Gender**: "Male" or "Female"
- **Married**: "Yes" or "No"
- **Dependents**: "0", "1", "2", or "3+"
- **Education**: "Graduate" or "Not Graduate"
- **Self_Employed**: "Yes" or "No"
- **Property_Area**: "Urban", "Semiurban", or "Rural"

### Validation Features
- Type checking and conversion
- Range validation
- Missing field detection
- Categorical value validation
- Detailed error messages
- Field-level error reporting

## 🔧 Feature Engineering

The model uses 20 engineered features:

### Financial Features
1. **ApplicantIncome** - Applicant's income
2. **CoapplicantIncome** - Co-applicant's income
3. **LoanAmount** - Loan amount
4. **Loan_Amount_Term** - Loan term
5. **TotalIncome** - Sum of applicant and co-applicant income
6. **Income_Loan_Ratio** - Ratio of total income to loan amount
7. **Loan_Amount_Per_Term** - Loan amount divided by term length
8. **EMI** - Equated Monthly Installment (loan payment)
9. **Balance_Income** - Remaining income after loan payment

### Log-Transformed Features
10. **Log_ApplicantIncome** - Log of applicant income
11. **Log_CoapplicantIncome** - Log of co-applicant income
12. **Log_LoanAmount** - Log of loan amount
13. **Log_TotalIncome** - Log of total income

### Encoded Categorical Features
14. **Gender_Encoded** - Encoded gender (0/1)
15. **Married_Encoded** - Encoded marital status (0/1)
16. **Dependents_Encoded** - Encoded number of dependents (0-3)
17. **Education_Encoded** - Encoded education level (0/1)
18. **Self_Employed_Encoded** - Encoded employment status (0/1)
19. **Property_Area_Encoded** - Encoded property area (0/1/2)
20. **Credit_History** - Credit history availability (0/1)

## 💻 Usage Examples

### Using cURL

```bash
# Make a prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Gender": "Male",
    "Married": "Yes",
    "Education": "Graduate"
  }'

# Check API status
curl http://localhost:5000/api

# Health check
curl http://localhost:5000/health
```

### Using Python requests

```python
import requests
import json

url = "http://localhost:5000/predict"
data = {
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1
}

response = requests.post(url, json=data)
print(json.dumps(response.json(), indent=2))
```

### Using Postman

1. Create a new POST request to `http://localhost:5000/predict`
2. Set Content-Type to `application/json`
3. Enter JSON data in the request body
4. Send the request

Screenshots of API responses are available in the `screenshots/` directory.

## 🧪 Testing

### Run Validation Tests

```bash
# Start the API first
python app.py

# In another terminal, run tests
python test_validation.py
```

The test suite includes:
- ✅ Valid complete data
- ✅ Minimal valid data
- ✅ Missing required fields
- ✅ Invalid data types
- ✅ Negative values
- ✅ Out of range values
- ✅ Invalid categorical values
- ✅ Edge cases

### Run API Integration Tests

```bash
python test_api.py
```

## 📊 Data Files

### Training Data
- **File**: `data/train_u6lujuX_CVtuZ9i.csv`
- **Size**: 614 samples
- **Features**: 12 (before engineering)
- **Target**: Loan_Status

### Test Data
- **File**: `data/test_Y3wMUE5_7gLdaTN.csv`
- **Size**: 367 samples
- **Features**: 11 (no target variable)

### Model Predictions
- **File**: `data/test_predictions.csv`
- **Contains**: Test set predictions from the trained model

## 🛠️ Technical Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Web Framework** | Flask | 3.1.2 |
| **ML Library** | scikit-learn | 1.8.0 |
| **Data Processing** | pandas | 2.3.3 |
| **Numerical Computing** | numpy | 2.4.0 |
| **Model Serialization** | joblib | 1.5.3 |
| **Frontend** | Bootstrap 5 | HTML/CSS |
| **Python Version** | 3.8+ | |

## 📓 Notebooks

### 01_data_exploration.ipynb
Comprehensive exploratory data analysis including:
- Data summary and statistics
- Missing value analysis
- Distribution analysis
- Correlation analysis
- Visualization of features

### 02_model_training.ipynb
Model development and evaluation:
- Data preprocessing
- Model training
- Hyperparameter tuning
- Cross-validation
- Performance metrics
- Model evaluation

### 03_feature_engineering.ipynb
Feature engineering techniques:
- Financial ratio calculations
- Log transformations
- Categorical encoding
- Feature scaling
- Feature importance analysis

## 🔄 Data Preprocessing Pipeline

The preprocessing pipeline handles:

1. **Missing Value Imputation**
   - Numeric fields: Filled with median
   - Categorical fields: Filled with mode

2. **Categorical Encoding**
   - Label encoding for categorical variables
   - One-hot encoding support

3. **Feature Engineering**
   - Financial ratios
   - Log transformations
   - Income calculations
   - EMI calculations

4. **Data Normalization**
   - Standardization of numeric features

## 📈 Development

### Running Different API Versions

```bash
# Version 1 (Original)
python app.py

# Version 2 (Enhanced validation)
python app_v2.py

# Version 2+ (Improved error handling)
python app_v2_improved.py

# Version 3 (Latest with additional features)
python app_v3.py
```

### Training the Model

To retrain the model:

```bash
python train_model_v3.py
```

This will:
1. Load the training data
2. Preprocess and engineer features
3. Train the Random Forest model
4. Evaluate performance
5. Save the model and metadata

## 🐛 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Use a different port
python -c "from app import app; app.run(port=5001)"
```

**Missing Dependencies**
```bash
pip install -r requirements.txt
```

**Model File Not Found**
Ensure the model file exists in the `models/` directory. Retrain if needed:
```bash
python train_model_v3.py
```

## 📝 License

This project is provided as-is for educational and demonstration purposes.

## 👤 Author

Created as a comprehensive machine learning project demonstrating:
- End-to-end ML pipeline
- REST API development
- Input validation
- Model deployment
- Web interface development

## 🔗 Related Resources

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Pandas Documentation](https://pandas.pydata.org/)

---

**Last Updated**: December 2025
**Model Version**: 3
**API Version**: 1.0