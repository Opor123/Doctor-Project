import pandas as pd
import numpy as np
import pickle
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Create output directory
output_dir = Path("Model/limit-input/output")
output_dir.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv("data/Breast_cancer/breast-cancer-age.csv")

# Feature selection
features = ['Age', 'Menopause', 'Tumor-Size', 'Inv-Nodes', 'Node-Caps', 'Deg-Malig', 'Breast', 'Breast-Quad', 'Irradiat']
X = df[features].copy()
y = df['Class']

# Convert numerical features
for col in ['Age', 'Tumor-Size', 'Inv-Nodes']:
    X[col] = X[col].apply(lambda val: int(val.split('-')[0]) if isinstance(val, str) and '-' in val else int(val))

# Fill missing values
X.fillna(X.median(numeric_only=True), inplace=True)

# Encode categorical features
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Feature Engineering
X["Severity_Score"] = X["Tumor-Size"] * X["Deg-Malig"]
X["Age_Group"] = pd.cut(X["Age"], bins=[0, 39, 59, 100], labels=["Young", "Middle", "Old"])
X["Age_Group"] = LabelEncoder().fit_transform(X["Age_Group"])
X["Tumor-Node_Caps"] = X["Tumor-Size"] * (X["Node-Caps"] == 1).astype(int)

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ['Age', 'Tumor-Size', 'Inv-Nodes', 'Deg-Malig', 'Severity_Score', 'Tumor-Node_Caps']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Encode target labels
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Hyperparameter tuning with Optuna
def objective(trial):
    model_type = trial.suggest_categorical("model_type", ["xgb", "lgbm", "cat"])
    
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
    }
    
    if model_type == "xgb":
        model = XGBClassifier(eval_metric='logloss', **params)
    elif model_type == "lgbm":
        model = LGBMClassifier(**params)
    else:
        model = CatBoostClassifier(verbose=0, **params)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)

best_params = study.best_params
best_model_type = best_params.pop("model_type")

if best_model_type == "xgb":
    best_model = XGBClassifier(eval_metric='logloss', **best_params)
elif best_model_type == "lgbm":
    best_model = LGBMClassifier(**best_params)
else:
    best_model = CatBoostClassifier(verbose=0, **best_params)

best_model.fit(X_train, y_train)

# Predictions
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=y_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=y_encoder.classes_, yticklabels=y_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(output_dir / "confusion_matrix.png")

# Save model & preprocessors
model_path = output_dir / "breast_cancer_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump((best_model, label_encoders, scaler, y_encoder), f)

print(f"\nModel saved at: {model_path}")
