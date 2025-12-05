import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

import os

# Load data
print("Loading data...")
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct path to data file
data_path = os.path.join(script_dir, "../data/processed_dataset.csv")
df = pd.read_csv(data_path)

# Feature Selection (Biomarkers Only)
# Using the same feature set as in final_validation.py for consistency
biomarkers = ['Age', 'Sleep Duration', 'Systolic_BP', 'Diastolic_BP', 'BMI Category']
target = 'Sleep Disorder'

X = df[biomarkers]
y = df[target]

print(f"Features used: {biomarkers}")
print("-" * 50)

# Define base models
# Note: SVC needs probability=True for Soft Voting
models = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('rf', RandomForestClassifier(random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('xgb', XGBClassifier(random_state=42, eval_metric='logloss'))
]

# Define Ensemble Methods
ensemble_methods = {
    "Majority Vote (Hard Voting)": VotingClassifier(estimators=models, voting='hard'),
    "Probability Averaging (Soft Voting)": VotingClassifier(estimators=models, voting='soft'),
    # Weighted Voting: Giving more weight to XGBoost and Random Forest as they typically perform well
    "Weighted Soft Voting": VotingClassifier(estimators=models, voting='soft', weights=[1, 2, 1, 2])
}

# Define Stratified K-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define metrics
scoring = {
    'accuracy': 'accuracy',
    'recall_macro': 'recall_macro',
    'precision_macro': 'precision_macro',
    'f1_macro': 'f1_macro'
}

print("Evaluating Ensemble Methods...")
print("=" * 50)

results = {}

for name, model in ensemble_methods.items():
    print(f"\nEnsemble Method: {name}")
    print("-" * 30)
    
    # Perform Cross-Validation
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    
    # Calculate mean scores
    acc = scores['test_accuracy'].mean()
    recall = scores['test_recall_macro'].mean()
    prec = scores['test_precision_macro'].mean()
    f1 = scores['test_f1_macro'].mean()
    
    results[name] = {'Recall': recall, 'Accuracy': acc, 'F1': f1}
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Generate Confusion Matrix
    y_pred = cross_val_predict(model, X, y, cv=cv)
    cm = confusion_matrix(y, y_pred)
    
    print("\nConfusion Matrix:")
    print(cm)
    print("(Rows: True, Cols: Predicted)")
    print("0: None, 1: Insomnia, 2: Sleep Apnea")

print("=" * 50)
print("\nSummary of Results:")
for name, metrics in results.items():
    print(f"{name}: Accuracy={metrics['Accuracy']:.4f}, Recall={metrics['Recall']:.4f}, F1={metrics['F1']:.4f}")

best_model = max(results, key=lambda k: results[k]['Recall'])
print(f"\nBest Ensemble Method based on Recall: {best_model}")
