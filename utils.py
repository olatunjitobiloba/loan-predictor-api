"""Utility helpers for loan predictor app"""

from typing import Any, Dict

import pandas as pd

# Performance constant
MAX_REQUEST_TIMES = 1000

DEFAULTS = {
    "LoanAmount": 128.0,
    "Loan_Amount_Term": 360.0,
    "Credit_History": 1.0,
    "CoapplicantIncome": 0,
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "Property_Area": "Urban",
    "ApplicantIncome": 0,
}


def apply_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply default values for missing fields.

    Args:
        data: Input data dictionary

    Returns:
        Data dictionary with defaults applied
    """
    if data is None:
        data = {}

    for key, default_value in DEFAULTS.items():
        # Use pandas isna for robust NA checking when values exist
        if key not in data or pd.isna(data.get(key)):
            data[key] = default_value

    return data
