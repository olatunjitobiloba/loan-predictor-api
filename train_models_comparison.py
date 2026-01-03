"""
Train multiple ML models and compare performance
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
import time
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the loan data"""
    print("Loading data...")
    # Primary dataset (some environments use a single combined file)
    try:
        df = pd.read_csv('data/loan_data.csv')
    except FileNotFoundError:
        # Fallback: use train + test CSVs available in the repo
        train_path = 'data/train_u6lujuX_CVtuZ9i.csv'
        test_path = 'data/test_Y3wMUE5_7gLdaTN.csv'
        if os.path.exists(train_path):
            df_train = pd.read_csv(train_path)
            df = df_train.copy()
            # If a test file exists, append it for consistent preprocessing
            if os.path.exists(test_path):
                df_test = pd.read_csv(test_path)
                # test file doesn't have Loan_Status - add placeholder
                if 'Loan_Status' not in df_test.columns:
                    df_test['Loan_Status'] = np.nan
                df = pd.concat([df, df_test], ignore_index=True, sort=False)
        else:
            raise
    
    # Handle missing values
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    
    # Feature Engineering
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Loan_to_Income_Ratio'] = df['LoanAmount'] / (df['Total_Income'] + 1)
    df['Dependents'] = df['Dependents'].replace('3+', '3')
    df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce')
    df['Income_per_Dependent'] = df['ApplicantIncome'] / (df['Dependents'] + 1)
    df['Log_ApplicantIncome'] = np.log1p(df['ApplicantIncome'])
    df['Log_LoanAmount'] = np.log1p(df['LoanAmount'])
    
    # Encode categorical variables
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, columns=['Property_Area'], prefix='Property', drop_first=True)
    
    # Encode target
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    
    # Drop Loan_ID
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)
    
    # Separate features and target
    # If some rows (e.g. from test set) don't have Loan_Status, keep them aside
    labeled = df['Loan_Status'].notna()

    X = df.drop('Loan_Status', axis=1)
    y = df.loc[labeled, 'Loan_Status']

    feature_names = list(X.columns)

    # Save processed test features (unlabeled) for potential later prediction
    unlabeled_X = X.loc[~labeled]
    if not unlabeled_X.empty:
        os.makedirs('models', exist_ok=True)
        unlabeled_X.to_csv('models/test_features_processed.csv', index=False)
        print(f"✅ Saved processed test features: models/test_features_processed.csv ({unlabeled_X.shape[0]} rows)")

    # Return only labeled rows for training
    return X.loc[labeled], y, feature_names

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Train model and return comprehensive metrics"""
    print(f"\n{'='*70}")
    print(f"Training {model_name}...")
    print(f"{'='*70}")
    
    # Time training
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Time prediction
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    avg_prediction_time = prediction_time / len(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # ROC AUC (if model has predict_proba)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except:
        roc_auc = None
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc) if roc_auc else None,
        'cv_mean': float(cv_mean),
        'cv_std': float(cv_std),
        'training_time': float(training_time),
        'avg_prediction_time': float(avg_prediction_time),
        'confusion_matrix': cm.tolist(),
        'total_params': get_model_params(model)
    }
    
    # Print results
    print(f"\n✅ {model_name} Results:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    if roc_auc:
        print(f"   ROC AUC:   {roc_auc:.4f}")
    print(f"   CV Score:  {cv_mean:.4f} (+/- {cv_std:.4f})")
    print(f"   Training time: {training_time:.3f}s")
    print(f"   Avg prediction time: {avg_prediction_time*1000:.2f}ms")
    
    return model, results

def get_model_params(model):
    """Get number of parameters/complexity of model"""
    if hasattr(model, 'n_estimators'):
        return model.n_estimators
    elif hasattr(model, 'coef_'):
        return model.coef_.shape[1]
    else:
        return 'N/A'

def main():
    """Train and compare multiple models"""
    print("="*70)
    print("ML MODEL COMPARISON TRAINING")
    print("="*70)
    
    # Load data
    X, y, feature_names = load_and_preprocess_data()
    print(f"\n✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Train set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='liblinear'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'Support Vector Machine': SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
    }
    
    # Train all models
    trained_models = {}
    all_results = {}
    
    for model_name, model in models.items():
        trained_model, results = train_and_evaluate_model(
            model, model_name, X_train, X_test, y_train, y_test
        )
        trained_models[model_name] = trained_model
        all_results[model_name] = results
    
    # Find best model
    print(f"\n{'='*70}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    comparison_df = pd.DataFrame(all_results).T
    comparison_df = comparison_df.sort_values('accuracy', ascending=False)
    
    print("\nRanking by Accuracy:")
    for i, (model_name, row) in enumerate(comparison_df.iterrows(), 1):
        print(f"{i}. {model_name:25s} - {row['accuracy']:.4f} ({row['accuracy']*100:.2f}%)")
    
    best_model_name = comparison_df.index[0]
    best_accuracy = comparison_df.iloc[0]['accuracy']
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    # Save all models
    print(f"\n{'='*70}")
    print("SAVING MODELS")
    print(f"{'='*70}")
    
    for model_name, model in trained_models.items():
        filename = f"models/{model_name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, filename)
        print(f"✅ Saved: {filename}")
    
    # Save feature names
    with open('models/feature_names.txt', 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    print(f"✅ Saved: models/feature_names.txt")
    
    # Save comparison results
    comparison_data = {
        'models': all_results,
        'best_model': best_model_name,
        'best_accuracy': float(best_accuracy),
        'feature_count': len(feature_names),
        'training_date': pd.Timestamp.now().isoformat()
    }
    
    with open('models/model_comparison.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"✅ Saved: models/model_comparison.json")
    
    # Create detailed comparison report
    print(f"\n{'='*70}")
    print("DETAILED COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Train Time':<12}")
    print("-"*85)
    for model_name, results in all_results.items():
        print(f"{model_name:<25} "
              f"{results['accuracy']:<10.4f} "
              f"{results['precision']:<10.4f} "
              f"{results['recall']:<10.4f} "
              f"{results['f1_score']:<10.4f} "
              f"{results['training_time']:<12.3f}s")
    
    print(f"\n{'='*70}")
    print("✅ MODEL TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nAll models saved to models/ directory")
    print(f"Best model: {best_model_name} ({best_accuracy*100:.2f}% accuracy)")

if __name__ == '__main__':
    main()
