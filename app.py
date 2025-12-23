from flask import Flask, jsonify, request, render_template, url_for

app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', title='Home')

@app.route("/api")
def api_info():
    return jsonify({
        "message": "Loan Predictor API",
        "version": "1.0",
        "status": "running"
    })

@app.route("/health")
def health():
    return jsonify({"status":"healthy"})

@app.route("/about")
def about():
    return render_template('about.html', title='About')


# ============================================================================
# REQUEST HANDLING TUTORIAL
# ============================================================================

# 1. READING JSON DATA (POST request body)
# Use request.get_json() to read JSON sent in the request body
@app.route("/predict", methods=["POST"])
def predict():
    # Get JSON data from request body
    # force=True allows parsing JSON even if Content-Type is not set correctly
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({
            "error": "Invalid JSON in request body",
            "message": "Please ensure your request body contains valid JSON",
            "details": str(e)
        }), 400
    
    # Check if JSON data was provided or if body is empty
    if not data:
        return jsonify({
            "error": "No JSON data provided",
            "message": "Request body is empty. Please send JSON data in the request body.",
            "example": {
                "age": 35,
                "income": 50000,
                "loan_amount": 20000
            }
        }), 400
    
    # Access specific fields from the JSON
    age = data.get("age")
    income = data.get("income")
    loan_amount = data.get("loan_amount")
    
    # Basic validation - check which fields are actually missing
    missing_fields = []
    if age is None:
        missing_fields.append("age")
    if income is None:
        missing_fields.append("income")
    if loan_amount is None:
        missing_fields.append("loan_amount")
    
    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields,
            "received_fields": list(data.keys()),
            "message": f"The following required fields are missing: {', '.join(missing_fields)}"
        }), 400
    
    # Your prediction logic here (simplified example)
    prediction = "approved" if income > 30000 else "rejected"
    
    # Calculate confidence (dummy calculation - replace with real ML model)
    confidence = 0.85 if prediction == "approved" else 0.75
    
    return jsonify({
        "received_data": data,
        "prediction": prediction,
        "confidence": confidence,
        "message": "This is a dummy prediction. ML model coming soon!"
    })


# 2. READING QUERY PARAMETERS (GET request)
# Query params are in the URL: /search?q=loan&limit=10
@app.route("/search", methods=["GET"])
def search():
    # Get query parameters from URL
    query = request.args.get("q")  # Returns None if not present
    limit = request.args.get("limit", default=10, type=int)  # Default value and type conversion
    page = request.args.get("page", 1, type=int)
    
    # Example: /search?q=loan&limit=20&page=2
    return jsonify({
        "query": query,
        "limit": limit,
        "page": page,
        "message": f"Searching for '{query}' with limit {limit} on page {page}"
    })


# 3. READING FORM DATA (POST with form-urlencoded)
# Used when HTML forms submit data
@app.route("/submit-form", methods=["POST"])
def submit_form():
    # Get form data (from HTML form submission)
    name = request.form.get("name")
    email = request.form.get("email")
    
    if not name or not email:
        return jsonify({"error": "Name and email are required"}), 400
    
    return jsonify({
        "message": "Form submitted successfully",
        "name": name,
        "email": email
    })


# 4. READING REQUEST HEADERS
# Headers contain metadata about the request
@app.route("/headers", methods=["GET", "POST"])
def show_headers():
    # Get specific header
    user_agent = request.headers.get("User-Agent")
    content_type = request.headers.get("Content-Type")
    
    # Get all headers as a dictionary
    all_headers = dict(request.headers)
    
    return jsonify({
        "user_agent": user_agent,
        "content_type": content_type,
        "all_headers": all_headers
    })


# 5. HANDLING MULTIPLE HTTP METHODS
# Same endpoint can handle GET and POST differently
@app.route("/loan", methods=["GET", "POST", "PUT", "DELETE"])
def loan_handler():
    method = request.method
    
    if method == "GET":
        # GET: Retrieve loan information
        loan_id = request.args.get("id")
        return jsonify({"method": "GET", "loan_id": loan_id, "action": "retrieve"})
    
    elif method == "POST":
        # POST: Create new loan
        data = request.get_json()
        return jsonify({"method": "POST", "data": data, "action": "create"})
    
    elif method == "PUT":
        # PUT: Update existing loan
        data = request.get_json()
        loan_id = request.args.get("id")
        return jsonify({"method": "PUT", "loan_id": loan_id, "data": data, "action": "update"})
    
    elif method == "DELETE":
        # DELETE: Remove loan
        loan_id = request.args.get("id")
        return jsonify({"method": "DELETE", "loan_id": loan_id, "action": "delete"})


# 6. ERROR HANDLING AND VALIDATION
@app.route("/validate-loan", methods=["POST"])
def validate_loan():
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
        
        if not age:
            errors.append("age is required")
        elif not isinstance(age, int) or age < 18 or age > 100:
            errors.append("age must be an integer between 18 and 100")
        
        if not income:
            errors.append("income is required")
        elif not isinstance(income, (int, float)) or income < 0:
            errors.append("income must be a positive number")
        
        if not loan_amount:
            errors.append("loan_amount is required")
        elif not isinstance(loan_amount, (int, float)) or loan_amount < 0:
            errors.append("loan_amount must be a positive number")
        
        # Return errors if validation failed
        if errors:
            return jsonify({"error": "Validation failed", "errors": errors}), 400
        
        # If validation passes, process the loan
        return jsonify({
            "status": "valid",
            "message": "Loan application validated successfully",
            "data": data
        })
        
    except Exception as e:
        # Catch any unexpected errors
        return jsonify({"error": "Server error", "message": str(e)}), 500


# 7. ACCESSING REQUEST METADATA
@app.route("/request-info", methods=["GET", "POST"])
def request_info():
    return jsonify({
        "method": request.method,  # GET, POST, etc.
        "url": request.url,  # Full URL
        "path": request.path,  # Path part of URL
        "remote_addr": request.remote_addr,  # Client IP address
        "is_json": request.is_json,  # True if Content-Type is application/json
        "content_type": request.content_type,  # Content-Type header
        "content_length": request.content_length,  # Size of request body
    })
if __name__ == "__main__":
    app.run(debug=True)



