"""
Evaluate calibration (Brier score) for available models and update
models/model_comparison.json with brier scores (raw and calibrated).
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

MODEL_FILES = {
    "random_forest": "models/random_forest.pkl",
    "logistic_regression": "models/logistic_regression.pkl",
    "gradient_boosting": "models/gradient_boosting.pkl",
    "support_vector_machine": "models/support_vector_machine.pkl",
}

CAL_CSV = "data/train_u6lujuX_CVtuZ9i.csv"


def load_data(path=CAL_CSV):
    df = pd.read_csv(path)
    # label
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0}).fillna(0).astype(int)
    # numeric safety
    df["ApplicantIncome"] = pd.to_numeric(
        df["ApplicantIncome"], errors="coerce"
    ).fillna(0)
    df["CoapplicantIncome"] = pd.to_numeric(
        df.get("CoapplicantIncome", 0), errors="coerce"
    ).fillna(0)
    df["LoanAmount"] = pd.to_numeric(df.get("LoanAmount", 0), errors="coerce").fillna(0)
    df["Loan_Amount_Term"] = pd.to_numeric(
        df.get("Loan_Amount_Term", 360), errors="coerce"
    ).fillna(360)
    df["Credit_History"] = pd.to_numeric(
        df.get("Credit_History", 1), errors="coerce"
    ).fillna(1)

    # defaults
    df["Gender"] = df["Gender"].fillna("Male")
    df["Married"] = df["Married"].fillna("Yes")
    df["Dependents"] = df["Dependents"].fillna("0").replace("3+", "3")
    df["Education"] = df["Education"].fillna("Graduate")
    df["Self_Employed"] = df["Self_Employed"].fillna("No")
    df["Property_Area"] = df["Property_Area"].fillna("Urban")

    # engineered
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["Income_Loan_Ratio"] = df["TotalIncome"] / (df["LoanAmount"] + 1)
    df["Loan_Amount_Per_Term"] = df["LoanAmount"] / (df["Loan_Amount_Term"] + 1)
    df["EMI"] = df["LoanAmount"] / (df["Loan_Amount_Term"] / 12 + 1)
    df["Balance_Income"] = df["TotalIncome"] - (df["EMI"] * 1000)
    df["Log_ApplicantIncome"] = np.log1p(df["ApplicantIncome"])
    df["Log_CoapplicantIncome"] = np.log1p(df["CoapplicantIncome"])
    df["Log_LoanAmount"] = np.log1p(df["LoanAmount"])
    df["Log_TotalIncome"] = np.log1p(df["TotalIncome"])

    # encoded
    gender_map = {"Male": 1, "Female": 0}
    df["Gender_Encoded"] = df["Gender"].map(gender_map).fillna(1)
    married_map = {"Yes": 1, "No": 0}
    df["Married_Encoded"] = df["Married"].map(married_map).fillna(1)
    df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce").fillna(0)
    df["Dependents_Encoded"] = df["Dependents"]
    education_map = {"Graduate": 1, "Not Graduate": 0}
    df["Education_Encoded"] = df["Education"].map(education_map).fillna(1)
    self_employed_map = {"Yes": 1, "No": 0}
    df["Self_Employed_Encoded"] = df["Self_Employed"].map(self_employed_map).fillna(0)
    property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    df["Property_Area_Encoded"] = df["Property_Area"].map(property_map).fillna(2)

    return df


RAW_FIELDS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]


def build_model_input(df, expected, build_func=None):
    # Lightweight mapping similar to app_v7.build_model_input for calibration use
    # We only support the features present in training models (mostly numeric/encoded)
    result = pd.DataFrame(index=df.index)
    for feat in expected:
        if feat in df.columns:
            # prefer numeric encoded if available
            enc = f"{feat}_Encoded"
            if (
                df[feat].dtype == object or df[feat].dtype == "O"
            ) and enc in df.columns:
                result[feat] = df[enc]
            else:
                result[feat] = df[feat]
        else:
            # try mapping candidates
            if feat == "Total_Income" and "TotalIncome" in df.columns:
                result[feat] = df["TotalIncome"]
            elif feat == "Loan_to_Income_Ratio" and "Income_Loan_Ratio" in df.columns:
                result[feat] = df["Income_Loan_Ratio"]
            elif feat in ("Property_Semiurban", "Property_Urban"):
                if feat == "Property_Semiurban":
                    result[feat] = (
                        (df.get("Property_Area") == "Semiurban").astype(int)
                        if "Property_Area" in df.columns
                        else 0
                    )
                else:
                    result[feat] = (
                        (df.get("Property_Area") == "Urban").astype(int)
                        if "Property_Area" in df.columns
                        else 0
                    )
            else:
                result[feat] = 0
    return result


def main():
    df = load_data()
    y = df["Loan_Status"]

    summary = {}

    # Load existing comparison JSON
    comp_path = "models/model_comparison.json"
    try:
        with open(comp_path, "r") as f:
            comp = json.load(f)
    except Exception:
        comp = {"models": {}}

    for name_key, path in MODEL_FILES.items():
        if not os.path.exists(path):
            print("Model not found:", path)
            continue
        print("\nEvaluating", name_key)
        clf = joblib.load(path)
        model_name_display = name_key.replace("_", " ").title()

        fnames = getattr(clf, "feature_names_in_", None)
        model_is_pipeline = hasattr(clf, "steps") and isinstance(
            getattr(clf, "steps"), list
        )

        if model_is_pipeline:
            X = df[RAW_FIELDS].copy()
        else:
            expected = (
                list(fnames)
                if fnames is not None and len(fnames) > 0
                else comp.get("feature_names", []) or []
            )
            X = build_model_input(df, expected)

        # existing model entry (not used further here)

        # raw proba
        if hasattr(clf, "predict_proba"):
            try:
                proba = clf.predict_proba(X)[:, 1]
                brier_raw = float(brier_score_loss(y, proba))
            except Exception as e:
                print("predict_proba failed for", name_key, e)
                proba = None
                brier_raw = None
        else:
            proba = None
            brier_raw = None

        # calibrate
        brier_cal = None
        if proba is not None:
            try:
                # Some sklearn versions don't support CalibratedClassifierCV with cv='prefit'.
                # Fall back to a simple logistic regressor on the predicted probabilities
                if hasattr(clf, "predict_proba"):
                    proba_train = clf.predict_proba(X)[:, 1]
                    cal_lr = LogisticRegression(solver="lbfgs")
                    cal_lr.fit(proba_train.reshape(-1, 1), y)
                    proba_cal = cal_lr.predict_proba(proba_train.reshape(-1, 1))[:, 1]
                    brier_cal = float(brier_score_loss(y, proba_cal))
                else:
                    brier_cal = None
            except Exception as e:
                print("calibration failed for", name_key, e)
                brier_cal = None

        # update comp
        if "models" not in comp:
            comp["models"] = {}
        if model_name_display not in comp["models"]:
            comp["models"][model_name_display] = {}
        if brier_raw is not None:
            comp["models"][model_name_display]["brier_score"] = brier_raw
        if brier_cal is not None:
            comp["models"][model_name_display]["brier_score_calibrated"] = brier_cal

        summary[model_name_display] = {
            "brier_raw": brier_raw,
            "brier_calibrated": brier_cal,
        }

        print("Result for", model_name_display, summary[model_name_display])

    # Save updated comparison
    try:
        with open(comp_path, "w") as f:
            json.dump(comp, f, indent=2)
        print("\nUpdated", comp_path)
    except Exception as e:
        print("Could not write model_comparison.json", e)


if __name__ == "__main__":
    main()
