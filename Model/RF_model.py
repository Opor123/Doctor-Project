import pandas as pd
import numpy as np
import os
import json
import joblib as jb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest

# Create directory to save model
save = 'Model\Models\Random Forrest'
os.makedirs(save, exist_ok=True)

# Load the dataset
data = pd.read_csv('data/Merge_data.csv')

# Identify target column
target_col = 'diagnosis' if 'diagnosis' in data.columns else 'Class'

# Drop unnecessary columns
if 'id' in data.columns:
    data = data.drop(columns=['id'])

# Convert target variable to numerical values
data[target_col] = data[target_col].map({'M': 1, 'B': 0, 2: 0, 4: 1})

# Handle missing values
imputer = SimpleImputer(strategy='median')
data.iloc[:, :-1] = imputer.fit_transform(data.iloc[:, :-1])

# Separate features and target variable
X = data.drop(columns=[target_col])
y = data[target_col]

# Save feature names
with open('feature_names.json', 'w') as f:
    json.dump(list(X.columns), f)

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Reapply imputation (important after splitting!)
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# Standardize the features (important for many models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
scaler_path = os.path.join(save, 'scaler_merge.pkl')
jb.dump(scaler, scaler_path)

# Apply SMOTE to handle class imbalance (if needed)
smote = SMOTE(random_state=42)
X_train, Y_train = smote.fit_resample(X_train, Y_train)


# Create and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42) # Add hyperparameters here if you want to tune them
rf_model.fit(X_train, Y_train)


# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("\nClassification Report:\n", classification_report(Y_test, y_pred))

# Compute AUC-ROC Score (using predict_proba)
auc = roc_auc_score(Y_test, rf_model.predict_proba(X_test)[:, 1])  # Correct way to get probabilities for AUC
print(f'\nAUC-ROC Score: {auc:.4f}')

# Compute Confusion Matrix
conf_matrix = confusion_matrix(Y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# Evaluate (alternative - accuracy only for RF)
accuracy = rf_model.score(X_test, Y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the model
model_path = os.path.join(save, 'rf_model.pkl') # Changed file extension to .pkl for joblib
jb.dump(rf_model, model_path)
print(f"Random Forest model has been saved at: {model_path}")

print("Random Forest model and scaler saved successfully!")