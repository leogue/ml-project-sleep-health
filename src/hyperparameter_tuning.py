import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score


df = pd.read_csv("../data/processed_dataset.csv")

# Select Biomarkers Only
biomarkers = ['Age', 'Sleep Duration', 'Systolic_BP', 'Diastolic_BP', 'BMI Category']
target = 'Sleep Disorder'

X = df[biomarkers]
y = df[target]

print(f"Starting Hyperparameter Tuning for XGBoost")
print(f"Features: {biomarkers}")
print(f"Target: {target}")
print("-" * 50)

xgb = XGBClassifier(random_state=42, eval_metric='logloss')

param_grid = {
    'n_estimators': [25, 50, 100, 200],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05 ,0.1, 0.15],
    'subsample': [0.8, 1.0]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scorer = make_scorer(recall_score, average='macro')

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring=scorer,
    cv=cv,
    n_jobs=-1,
    verbose=1
)

print("Running GridSearchCV...")
grid_search.fit(X, y)


print(f"Best Recall Score: {grid_search.best_score_:.4f}")
print("Best Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
