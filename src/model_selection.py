import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


df = pd.read_csv("../data/processed_dataset.csv")

# 'Sleep Disorder' is our target variable (0: None, 1: Insomnia, 2: Sleep Apnea)
X = df.drop('Sleep Disorder', axis=1)
y = df['Sleep Disorder']



# Define models with default hyperparameters
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
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
# Define evaluation function
def evaluate_models(X, y, models, cv, scoring, title):
    print("=" * 50)
    print(title)
    print("=" * 50)
    
    results = {}
    
    for name, model in models.items():
        print(f"\nModel: {name}")
        print("-" * 30)
        
        # Perform Cross-Validation for Metrics
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        
        # Calculate scores
        acc = scores['test_accuracy'].mean()
        recall = scores['test_recall_macro'].mean()
        prec = scores['test_precision_macro'].mean()
        f1 = scores['test_f1_macro'].mean()
        
        results[name] = {'Recall': recall}
        
        print(f"Accuracy:  {acc:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        # Generate Confusion Matrix using Cross-Validation Predictions
        y_pred = cross_val_predict(model, X, y, cv=cv)
        cm = confusion_matrix(y, y_pred)
        
        print("\nConfusion Matrix:")
        print(cm)
        print("(Rows: True, Cols: Predicted)")
        print("0: None, 1: Insomnia, 2: Sleep Apnea")
    
    print("\n\nAnalysis:")
    best_recall_model = max(results, key=lambda k: results[k]['Recall'])
    print(f"Best model for Recall: {best_recall_model} ({results[best_recall_model]['Recall']:.4f})")


# 1. Evaluation with ALL features
evaluate_models(X, y, models, cv, scoring, "Model Evaluation Results with all features")


# 2. Evaluation WITHOUT Occupation features
cols_to_drop = [col for col in df.columns if col.startswith('Occupation_')]
X_no_occ = df.drop(cols_to_drop + ['Sleep Disorder'], axis=1) # Ensure target is not in X
evaluate_models(X_no_occ, y, models, cv, scoring, "Model Evaluation Results without Occupation features")


# 3. Evaluation with BIOMARKERS ONLY
biomarkers = ['Age', 'Sleep Duration', 'Systolic_BP', 'Diastolic_BP', 'BMI Category']
X_bio = df[biomarkers]
evaluate_models(X_bio, y, models, cv, scoring, "Model Evaluation Results with Biomarkers Only")