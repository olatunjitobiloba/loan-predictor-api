# Loan Predictor API

Machine Learning API for predicting loan approval.

## Progress

- [x] Day 1: Flask setup + basic routes
- [x] Day 2: POST endpoint + error handling
- [x] Day 3: Data loading
- [x] Day 4: Data visualization
- [ ] Day 5: ML model training

## Current Features

- ✅ RESTful API with Flask
- ✅ JSON request/response handling
- ✅ Error handling with detailed validation
- ✅ Health check endpoint
- ✅ Web interface with Bootstrap
- ⏳ ML model (coming soon)

## API Endpoints

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
Predict loan approval (dummy predictions for now).

**Request:**
```json
{
  "age": 35,
  "income": 50000,
  "loan_amount": 20000
}
```

**Response:**
```json
{
  "received_data": {
    "age": 35,
    "income": 50000,
    "loan_amount": 20000
  },
  "prediction": "approved",
  "confidence": 0.85,
  "message": "This is a dummy prediction. ML model coming soon!"
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
Validate loan application data.

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

## How to Run

1. Install dependencies:
```bash
pip install flask
```

2. Run the application:
```bash
python app.py
```

3. Visit the web interface:
```
http://localhost:5000
```

4. Test API endpoints:
- Use Postman or curl to test POST endpoints
- See `POSTMAN_TESTING_GUIDE.md` for detailed testing instructions

## Tech Stack

- **Python** 3.9+
- **Flask** - Web framework
- **Bootstrap 5** - Frontend styling
- **Jinja2** - Template engine

(More coming soon...)

## Project Structure

```
loan-predictor-api/
├── app.py                 # Main Flask application
├── templates/             # HTML templates
│   ├── layout.html        # Base template
│   ├── home.html          # Home page
│   └── about.html         # About page
├── static/                # Static files
│   └── main.css           # Custom styles
└── README.md              # This file
```

## Testing

See `POSTMAN_TESTING_GUIDE.md` for comprehensive testing instructions.

Quick test with curl:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "income": 50000, "loan_amount": 20000}'
```
