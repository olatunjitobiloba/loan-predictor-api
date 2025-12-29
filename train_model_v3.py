import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
import joblib
from preprocess import preprocess_loan_data

print("="*60)
print("IMPROVED MODEL TRAINING")
print("="*60)

# Load data
print("\n1. Loading data...")
train_df = pd.read_csv('data/train_u6lujuX_CVtuZ9i.csv')
test_df = pd.read_csv('data/test_Y3wMUE5_7gLdaTN.csv')
print(f"   ✓ Loaded {len(train_df)} training records")
print(f"   ✓ Loaded {len(test_df)} test records")

# Preprocess data
print("\n2. Preprocessing...")
train_processed, test_processed, preprocessor = preprocess_loan_data(train_df, test_df)

# Separate features and target
X = train_processed.drop('Loan_Status', axis=1)
y = train_processed['Loan_Status']

print(f"\n   ✓ Features: {X.shape[1]}")
print(f"   ✓ Samples: {len(X)}")
print(f"   ✓ Feature names: {X.columns.tolist()}")

# Split data for validation
print("\n3. Splitting data for validation...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Training: {len(X_train)} samples")
print(f"   ✓ Validation: {len(X_val)} samples")

# Train model
print("\n4. Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,  # Fixed typo: was 'min_simples_split'
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("   ✓ Model trained!")

# Cross-validation
print("\n5. Cross-validation (5-fold)...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"   ✓ CV Scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"   ✓ Mean CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")

# Predictions on validation set
print("\n6. Making predictions on validation set...")
y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)

# Evaluate
print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print(f"\n📊 Metrics:")
print(f"   Accuracy:  {accuracy:.2%}")
print(f"   Precision: {precision:.2%}")
print(f"   Recall:    {recall:.2%}")
print(f"   F1-Score:  {f1:.2%}")

print(f"\n📈 Improvement:")
print(f"   Previous accuracy: 79.83%")
print(f"   Current accuracy:  {accuracy:.2%}")
improvement = (accuracy - 0.7983) * 100
print(f"   Improvement:       {improvement:+.2f}%")

# Detailed report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_val, y_pred, target_names=['Rejected', 'Approved']))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_pred)
print(f"              Predicted")
print(f"              Rejected  Approved")
print(f"Actual Rejected    {cm[0][0]:3d}       {cm[0][1]:3d}")
print(f"       Approved    {cm[1][0]:3d}       {cm[1][1]:3d}")

# Feature Importance
print("\n" + "="*60)
print("TOP 10 MOST IMPORTANT FEATURES")
print("="*60)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# Save model
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

import os
os.makedirs('models', exist_ok=True)

model_path = 'models/loan_model_v2.pkl'
joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")

# Save preprocessor
preprocessor_path = 'models/preprocessor.pkl'
joblib.dump(preprocessor, preprocessor_path)
print(f"✅ Preprocessor saved to {preprocessor_path}")

# Save feature names
feature_names_path = 'models/feature_names.txt'
with open(feature_names_path, 'w') as f:
    f.write('\n'.join(X.columns.tolist()))
print(f"✅ Feature names saved to {feature_names_path}")

# Save model info
model_info = {
    'features': X.columns.tolist(),
    'num_features': len(X.columns),
    'training_samples': len(X_train),
    'validation_samples': len(X_val),
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'cv_mean': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'model_params': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    }
}

import json
with open('models/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)
print(f"✅ Model info saved to models/model_info.json")

# Generate predictions for test set (for submission)
print("\n" + "="*60)
print("GENERATING TEST PREDICTIONS")
print("="*60)

# Get Loan_IDs from original test file
test_ids = test_df['Loan_ID']

# Make predictions
test_predictions = model.predict(test_processed)
test_predictions_proba = model.predict_proba(test_processed)

# Create submission file
submission = pd.DataFrame({
    'Loan_ID': test_ids,
    'Loan_Status': ['Y' if pred == 1 else 'N' for pred in test_predictions]
})

submission_path = 'models/submission.csv'
submission.to_csv(submission_path, index=False)
print(f"✅ Submission file saved to {submission_path}")
print(f"   Total predictions: {len(submission)}")
print(f"   Approved: {(test_predictions == 1).sum()}")
print(f"   Rejected: {(test_predictions == 0).sum()}")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE")
print("="*60)