import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv("data/Breast_cancer/breast-cancer-age.csv")
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")

# Display dataset info
print(f"\nTarget value distribution:\n{df['Class'].value_counts()}")
print(f"Missing values:\n{df.isnull().sum()}")

# Feature selection
features = ['Age', 'Menopause', 'Tumor-Size', 'Inv-Nodes', 'Node-Caps', 'Deg-Malig', 'Breast', 'Breast-Quad', 'Irradiat']
X = df[features].copy()
y = df['Class']

# Helper function to extract numerical values
def extract_numeric(value):
    if isinstance(value, str) and '-' in value:
        return int(value.split('-')[0])
    try:
        return int(value)
    except ValueError:
        return np.nan

# Apply numeric conversion
for col in ['Age', 'Tumor-Size', 'Inv-Nodes']:
    X[col] = X[col].apply(extract_numeric)

# Fill missing values with median
X.fillna(X.median(numeric_only=True), inplace=True)

# Encode categorical features
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ['Age', 'Tumor-Size', 'Inv-Nodes', 'Deg-Malig']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Encode target labels
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Train RandomForest model
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("Model training completed")

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate model
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

# Feature Importance
plt.figure(figsize=(10, 6))
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.title('Feature Importance')
plt.savefig(output_dir / "feature_importance.png")

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")

# Save model & preprocessors
model_path = output_dir / "breast_cancer_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump((model, label_encoders, scaler, y_encoder), f)
print(f"\nModel saved at: {model_path}")
