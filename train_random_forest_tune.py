#!/usr/bin/env python3
"""Train and tune RandomForest to reach target accuracy and save model."""
import json
import os

import joblib
import pandas as pd
# from scipy.stats import randint  # not used
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from preprocess import preprocess_loan_data

TARGET_ACC = 0.8862
RANDOM_STATE = 42


def main():
    os.makedirs("models", exist_ok=True)

    print("Loading data...")
    train_df = pd.read_csv("data/train_u6lujuX_CVtuZ9i.csv")
    test_df = pd.read_csv("data/test_Y3wMUE5_7gLdaTN.csv")

    print("Preprocessing data...")
    train_processed, test_processed, preprocessor = preprocess_loan_data(
        train_df, test_df
    )

    X = train_processed.drop("Loan_Status", axis=1)
    y = train_processed["Loan_Status"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Define parameter distributions
    param_dist = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [8, 10, 12, 15, 18, 20, None],
        "min_samples_split": [2, 4, 5, 8, 10],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt", "log2", None],
        "class_weight": [None, "balanced"],
    }

    base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    print("Starting RandomizedSearchCV (this may take a while)...")
    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=60,
        scoring="accuracy",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=2,
    )

    rs.fit(X_train, y_train)

    print("Best params:", rs.best_params_)
    best_initial = rs.best_estimator_

    # Evaluate on validation for initial best
    y_pred_init = best_initial.predict(X_val)
    acc_init = accuracy_score(y_val, y_pred_init)
    prec_init = precision_score(y_val, y_pred_init)
    rec_init = recall_score(y_val, y_pred_init)
    f1_init = f1_score(y_val, y_pred_init)

    print(f"Initial Validation Accuracy: {acc_init:.4f}")
    print(f"Precision: {prec_init:.4f}, Recall: {rec_init:.4f}, F1: {f1_init:.4f}")

    best_overall = best_initial
    acc_overall = acc_init
    prec_overall = prec_init
    rec_overall = rec_init
    f1_overall = f1_init

    # If target not met (allow tiny epsilon), try refinement
    if acc_init + 1e-8 < TARGET_ACC:
        print(
            f"Target {TARGET_ACC:.4f} not met (got {acc_init:.4f}). Running refinement...\n"
        )
        # Expand n_estimators and try a focused search
        param_grid = {
            "n_estimators": [500, 700, 900],
            "max_depth": [rs.best_params_.get("max_depth")],
            "min_samples_split": [rs.best_params_.get("min_samples_split")],
            "min_samples_leaf": [rs.best_params_.get("min_samples_leaf")],
            "max_features": [rs.best_params_.get("max_features")],
            "class_weight": [rs.best_params_.get("class_weight")],
        }
        print("Refinement grid:", param_grid)
        rs2 = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            param_distributions=param_grid,
            n_iter=6,
            scoring="accuracy",
            cv=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=2,
        )
        rs2.fit(X_train, y_train)
        best_refined = rs2.best_estimator_
        y_pred_ref = best_refined.predict(X_val)
        acc_ref = accuracy_score(y_val, y_pred_ref)
        prec_ref = precision_score(y_val, y_pred_ref)
        rec_ref = recall_score(y_val, y_pred_ref)
        f1_ref = f1_score(y_val, y_pred_ref)
        print(f"Refined Validation Accuracy: {acc_ref:.4f}")

        # Choose the better model between initial and refined
        if acc_ref > acc_overall:
            best_overall = best_refined
            acc_overall = acc_ref
            prec_overall = prec_ref
            rec_overall = rec_ref
            f1_overall = f1_ref
        else:
            print(
                "Refined model did not improve over initial best; keeping initial best."
            )

    # Save model and artifacts
    model_path = "models/random_forest.pkl"
    joblib.dump(best_overall, model_path)
    print(f"Saved best model to {model_path}")

    preproc_path = "models/preprocessor.pkl"
    joblib.dump(preprocessor, preproc_path)
    print(f"Saved preprocessor to {preproc_path}")

    # Save feature names as JSON
    feature_names = X.columns.tolist()
    with open("models/feature_names.json", "w") as f:
        json.dump(feature_names, f)
    print("Saved feature names to models/feature_names.json")

    # Update model_comparison.json
    comp_path = "models/model_comparison.json"
    comp = {}
    if os.path.exists(comp_path):
        try:
            with open(comp_path, "r") as f:
                comp = json.load(f)
        except Exception:
            comp = {}

    comp.setdefault("models", {})
    comp["models"]["Random Forest"] = {
        "accuracy": round(float(acc_overall), 4),
        "precision": round(float(prec_overall), 4),
        "recall": round(float(rec_overall), 4),
        "f1_score": round(float(f1_overall), 4),
        "params": best_overall.get_params(),
    }

    with open(comp_path, "w") as f:
        json.dump(comp, f, indent=2)

    print("Updated models/model_comparison.json")

    print("Done.")


if __name__ == "__main__":
    main()
