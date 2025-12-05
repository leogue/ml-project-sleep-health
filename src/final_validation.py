import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix



# Load datasets
print("Loading datasets...")
train_df = pd.read_csv("../data/processed_dataset.csv")
val_df = pd.read_csv("../data/validation_dataset.csv")

# Feature Selection (Biomarkers Only)
biomarkers = ['Age', 'Sleep Duration', 'Systolic_BP', 'Diastolic_BP', 'BMI Category']
target = 'Sleep Disorder'

X_train = train_df[biomarkers]
y_train = train_df[target]

X_val = val_df[biomarkers]
y_val = val_df[target]

print(f"Features: {biomarkers}")
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print("-" * 50)

# Initialize Model with Optimized Hyperparameters
print("Initializing XGBoost with optimized parameters...")
xgb_final = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=1.0,
    random_state=42,
    eval_metric='logloss'
)

# Train Model
print("Training model...")
xgb_final.fit(X_train, y_train)

# Evaluate on Validation Set
print("Evaluating on Validation Set...")
y_pred = xgb_final.predict(X_val)

# Calculate Metrics
acc = accuracy_score(y_val, y_pred)
recall = recall_score(y_val, y_pred, average='macro')
precision = precision_score(y_val, y_pred, average='macro')
f1 = f1_score(y_val, y_pred, average='macro')
cm = confusion_matrix(y_val, y_pred)

print("-" * 50)
print("FINAL VALIDATION RESULTS")
print("-" * 50)
print(f"Accuracy:  {acc:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("(Rows: True, Cols: Predicted)")
print("0: None, 1: Insomnia, 2: Sleep Apnea")
print("-" * 50)

