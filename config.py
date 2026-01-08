"""
Application configuration
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration"""

    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Cache configuration
    CACHE_TYPE = "simple"
    CACHE_DEFAULT_TIMEOUT = 300

    # Rate limiting
    RATELIMIT_STORAGE_URL = "memory://"
    RATELIMIT_DEFAULT = "200 per hour, 1000 per day"

    # Model paths
    MODEL_DIR = "models"
    DEFAULT_MODEL = "loan_model_v2.pkl"
    FEATURE_NAMES_FILE = "feature_names.txt"
    MODEL_INFO_FILE = "model_info.json"

    # Performance
    MAX_REQUEST_TIMES = 1000


class DevelopmentConfig(Config):
    """Development configuration"""

    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DEV_DATABASE_URL", "sqlite:///predictions.db"
    )


class ProductionConfig(Config):
    """Production configuration"""

    # Expect DATABASE_URL to be set in production environment
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", "")


class TestingConfig(Config):
    """Testing configuration"""

    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    CACHE_TYPE = "null"


config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}
