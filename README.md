# Loan Predictor API

Machine Learning API for predicting loan approval using Random Forest classifier.

## 🎯 Project Overview

This project predicts whether a loan application will be approved based on applicant information such as income, credit history, and loan amount. Built as a RESTful API with Flask, it demonstrates end-to-end ML deployment from data preprocessing to model serving.

**Current Model Accuracy: 88.62%**

## ✨ Features

- ✅ RESTful API with Flask
- ✅ JSON request/response handling
- ✅ Comprehensive error handling with detailed validation
- ✅ Health check endpoint
- ✅ Web interface with Bootstrap 5
- ✅ Random Forest ML model (88.62% accuracy)
- ✅ Data preprocessing pipeline
- ✅ Exploratory Data Analysis (EDA) with visualizations
- ✅ Feature engineering notebooks
- ⏳ Database integration (coming soon)
- ⏳ Production deployment (coming soon)

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/olatunjitobiloba1/loan-predictor-api
cd loan-predictor-api

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Visit the web interface at `http://localhost:5000`

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 88.62% |
| Precision | 89%   |
| Recall | 95% |
| F1-Score | 92% |

## 🔌 API Endpoints

### GET /
Returns the home page with API documentation.

### GET /api
Returns API information in JSON format.

**Response:**
```json
{
  "message": "Loan Predictor API",
  "version": "1.0",
  "status": "running"
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### POST /predict
Predict loan approval based on applicant data.

**Request:**
```json
{
  "age": 35,
  "income": 50000,
  "loan_amount": 20000,
  "credit_history": 1,
  "employment_years": 5
}
```

**Response:**
```json
{
  "received_data": {
    "age": 35,
    "income": 50000,
    "loan_amount": 20000,
    "credit_history": 1,
    "employment_years": 5
  },
  "prediction": "approved",
  "confidence": 0.85,
  "message": "Loan application processed successfully"
}
```

**Error Response (Missing Fields):**
```json
{
  "error": "Missing required fields",
  "missing_fields": ["age"],
  "received_fields": ["income", "loan_amount"],
  "message": "The following required fields are missing: age"
}
```

### POST /validate-loan
Validate loan application data before prediction.

**Request:**
```json
{
  "age": 25,
  "income": 40000,
  "loan_amount": 15000
}
```

**Response:**
```json
{
  "status": "valid",
  "message": "Loan application validated successfully",
  "data": {
    "age": 25,
    "income": 40000,
    "loan_amount": 15000
  }
}
```

## 🛠️ Tech Stack

- **Backend:** Flask, Python 3.9+
- **ML:** scikit-learn, pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Frontend:** Bootstrap 5, Jinja2
- **Development:** Jupyter Notebooks
- **Testing:** pytest (coming soon)
- **Deployment:** Render (coming soon)

## 📁 Project Structure

```
loan-predictor-api/
├── __pycache__/              # Python cache files
├── cursor/                   # Cursor IDE files
├── .venv/                    # Virtual environment
├── data/                     # Datasets
│   ├── data_summary.txt
│   ├── test_predictions.csv
│   ├── test_Y3wMUE5_7gLdaTN.csv
│   └── train_u6lujuX_CVtuZ9i.csv
├── models/                   # Saved models & preprocessors
│   ├── feature_names.txt
│   ├── loan_model_v1.pkl
│   ├── loan_model_v2.pkl
│   ├── model_info.json
│   ├── preprocessor.pkl
│   └── submission.csv
├── notebooks/                # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── explore_data.ipynb
├── screenshots/              # API testing screenshots
│   ├── API_GET_RESPONSE_IN_POSTMAN.png
│   ├── API_POST_RESPONSE_IN_POSTMAN.png
│   ├── confusion_matrix.png
│   └── feature_importance_chart.png
├── static/                   # Static files (CSS, JS)
│   └── main.css
├── templates/                # HTML templates
│   ├── about.html
│   ├── home.html
│   └── layout.html
├── visualizations/           # EDA plots
│   └── eda_plots/
│       ├── 01_missing_values_analysis.png
│       ├── 02_target_variable_distribution.png
│       ├── 03_applicant_income_distribution.png
│       ├── 04_loan_amount_distribution.png
│       ├── 05_income_distribution_combined.png
│       ├── 06_credit_history_impact.png
│       ├── 07_property_area_approval.png
│       ├── 08_education_income_boxplot.png
│       ├── 09_loan_vs_income_scatter.png
│       ├── 10_categorical_vs_loan_status.png
│       ├── 11_correlation_heatmap.png
│       ├── 12_outlier_detection.png
│       └── 13_loan_income_ratio.png
├── .gitattributes
├── .gitignore
├── app.py                    # Main Flask application
├── preprocess.py             # Data preprocessing pipeline
├── train_model_v3.py         # Model training script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 📈 Development Progress

- [x] **Day 1:** Flask setup + basic routes
- [x] **Day 2:** POST endpoint + error handling
- [x] **Day 3:** Data loading & exploration
- [x] **Day 4:** Data visualization & EDA (13 plots)
- [x] **Day 5:** ML model training (Random Forest)
- [x] **Day 6:** Model optimization (79.83% → 88.62% accuracy)
- [ ] **Week 2:** Integration, testing, deployment

## 🧪 Testing

Test the API using the screenshots in the `screenshots/` folder as reference.

**Quick test with curl:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "income": 50000, "loan_amount": 20000, "credit_history": 1, "employment_years": 5}'
```

**Test with Postman:**
- Import the endpoints from the API documentation
- See screenshots in `screenshots/` folder for expected responses

## 📊 Data Analysis

The project includes comprehensive EDA with 13 visualizations:
- Missing values analysis
- Target variable distribution
- Income and loan amount distributions
- Credit history impact on approval
- Property area analysis
- Correlation heatmap
- Outlier detection
- And more...

All visualizations are available in `visualizations/eda_plots/`

## 🎓 What I Learned

1. **Feature engineering** can improve accuracy by 8-10%
2. **Credit history** is the strongest predictor of loan approval
3. Proper **preprocessing** is crucial for model performance
4. Real-world ML is **80% data work, 20% modeling**
5. **API design** matters for usability and maintainability
6. **Iterative improvement**: v1 (79.83%) → v2 (88.62%)

## 🔮 Future Improvements

- [ ] Add database integration (PostgreSQL/MongoDB)
- [ ] Implement user authentication
- [ ] Add model versioning and A/B testing
- [ ] Deploy to production (Render/AWS)
- [ ] Add comprehensive test suite (pytest)
- [ ] Create Docker containerization
- [ ] Add API rate limiting
- [ ] Implement logging and monitoring
- [ ] Create CI/CD pipeline

## 📫 Contact

**Olatunji Oluwatobiloba**
- GitHub: [@olatunjitobiloba](https://github.com/olatunjitobiloba)
- LinkedIn: [@olatunjitobiloba](https://www.linkedin.com/in/olatunji-oluwatobiloba-186659291/)
- Email: [olatunjitobiloba05@gmail.com](mailto:olatunjitobiloba05@gmail.com)

## 📝 License

MIT License

---

**Built with ❤️ in 6 days as part of my journey to become an ML Engineer.**