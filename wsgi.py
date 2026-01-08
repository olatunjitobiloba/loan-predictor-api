"""
WSGI configuration for PythonAnywhere deployment
"""

import os
import sys

# Add your project directory to the sys.path
project_home = "/home/loanpredictorapi/loan-predictor-api"
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

# Set environment variables
os.environ[
    "DATABASE_URL"
] = "sqlite:////home/loanpredictorapi/loan-predictor-api/predictions.db"
os.environ["SECRET_KEY"] = "your-secret-key-here"

# Import Flask app from app_v5.py (import intentionally after env setup)
from app_v5 import app as application  # noqa: F401,E402
