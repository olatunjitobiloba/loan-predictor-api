"""
Swagger/OpenAPI configuration
"""

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/apispec.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs"
}
swagger_config['doc_dir'] = 'docs/swagger'

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Loan Prediction API",
        "description": "AI-powered loan approval prediction system with 88.62% accuracy. Built with Flask, scikit-learn, and PostgreSQL.",
        "contact": {
            "name": "Olatunji Oluwatobiloba",
            "email": "olatunjitobiloba05@example.com",
            "url": "https://github.com/olatunjitobiloba1/loan-predictor-api"
        },
        "version": "5.0.0",
        "license": {
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        }
    },
    # Do not hard-code host here; leaving `host` out lets Swagger UI
    # call the same origin from which the docs are served, avoiding
    # localhost vs 127.0.0.1 mismatches that trigger CORS errors.
    "basePath": "/",
    # Prefer secure scheme when docs are served over HTTPS (hosts like Render).
    # Include both so Swagger UI uses HTTPS on deployed sites and HTTP locally.
    "schemes": [
        "https",
        "http"
    ],
    "tags": [
        {
            "name": "Health",
            "description": "Health check and status endpoints"
        },
        {
            "name": "Prediction",
            "description": "Loan prediction endpoints"
        },
        {
            "name": "History",
            "description": "Prediction history endpoints"
        },
        {
            "name": "Statistics",
            "description": "Analytics and statistics endpoints"
        },
        {
            "name": "Information",
            "description": "API and model information"
        }
    ],
    "securityDefinitions": {},
    "security": []
}
