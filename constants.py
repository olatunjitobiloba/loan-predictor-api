"""Application constants for loan-predictor-api"""

import os

# Performance
MAX_REQUEST_TIMES = int(os.environ.get("MAX_REQUEST_TIMES", 1000))

# Model paths and files
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "loan_model_v2.pkl")
FEATURE_NAMES_FILE = os.environ.get("FEATURE_NAMES_FILE", "feature_names.txt")
MODEL_INFO_FILE = os.environ.get("MODEL_INFO_FILE", "model_info.json")

# Cache & rate limit defaults
CACHE_DEFAULT_TIMEOUT = int(os.environ.get("CACHE_DEFAULT_TIMEOUT", 300))
RATELIMIT_DEFAULT = os.environ.get("RATELIMIT_DEFAULT", "200 per hour, 1000 per day")
RATELIMIT_STORAGE_URL = os.environ.get("RATELIMIT_STORAGE_URL", "memory://")
