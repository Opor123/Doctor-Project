import pandas as pd
import numpy as np
import os
import json
import joblib as jb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Create directory to save models
save_dir = 'Model/Models/RF'
os.makedirs(save_dir, exist_ok=True)

def preprocess_and_train_rf(file_path, target_col, model_name):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Strip whitespace from column names
    data.columns = data.columns.str.strip()
    
    # Drop unnecessary columns
    if 'id' in data.columns or 'Sample code number' in data.columns:
        data = data.drop(columns=['id', 'Sample code number'], errors='ignore')
    
    # Convert target variable to numerical values
    data[target_col] = data[target_col].map({'M': 1, 'B': 0}).fillna(-1)
    
    # Remove rows with missing target values
    data = data[data[target_col] != -1]

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    data.iloc[:, :-1] = imputer.fit_transform(data.iloc[:, :-1])
    
    # Separate features and target variable
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Save feature names
    feature_names_path = os.path.join(save_dir, f'feature_names_{model_name}.json')
    with open(feature_names_path, 'w') as f:
        json.dump(list(X.columns), f)
    
    # Split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Reapply imputation to ensure no missing values
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save the scaler
    scaler_path = os.path.join(save_dir, f'scaler_{model_name}.pkl')
    jb.dump(scaler, scaler_path)
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train, Y_train = smote.fit_resample(X_train, Y_train)
    
    # Define and train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, Y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Evaluate the model
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(Y_test, y_pred))
    
    # Compute AUC-ROC Score
    auc = roc_auc_score(Y_test, rf_model.predict_proba(X_test)[:, 1])
    print(f'\nAUC-ROC Score for {model_name}: {auc:.4f}')
    
    # Compute Confusion Matrix
    conf_matrix = confusion_matrix(Y_test, y_pred)
    print(f"\nConfusion Matrix for {model_name}:\n", conf_matrix)
    
    # Save the model
    model_path = os.path.join(save_dir, f'{model_name}.pkl')
    jb.dump(rf_model, model_path)
    print(f"Model for {model_name} saved at: {model_path}\n")

# Train model for Breast Cancer dataset
preprocess_and_train_rf('data/Breast_cancer.csv', 'diagnosis', 'breast_cancer_rf_model')
