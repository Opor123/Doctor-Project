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

# Create directory to save models
save_dir = 'Model/Models/Random Forrest'
os.makedirs(save_dir, exist_ok=True)

def preprocess_and_train(file_path, target_col, model_name):
    try:
        # Load the dataset
        data = pd.read_csv(file_path)

        # Drop unnecessary columns (more robust)
        cols_to_drop = ['id', 'Sample code number']
        data = data.drop(columns=[col for col in cols_to_drop if col in data.columns], errors='ignore')

        # Convert target variable to numerical values (handle potential errors)
        mapping = {'M': 1, 'B': 0, 2: 0, 4: 1}
        data[target_col] = data[target_col].map(mapping).fillna(-1)  # Map and handle missing values
        data = data[data[target_col] != -1] # Remove rows with unmapped values

        # Handle missing values (after mapping target)
        imputer = SimpleImputer(strategy='median')
        numerical_cols = data.select_dtypes(include=np.number).columns[:-1] # Impute only numerical features, excluding target
        data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

        # Separate features and target variable
        X = data.drop(columns=[target_col])
        y = data[target_col]

        # Save feature names
        feature_names_path = os.path.join(save_dir, f'feature_names_{model_name}.json')
        with open(feature_names_path, 'w') as f:
            json.dump(list(X.columns), f)

        # Split the dataset
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # Impute and Scale AFTER splitting (prevent data leakage)
        X_train[numerical_cols] = imputer.fit_transform(X_train[numerical_cols]) # Impute only numerical features
        X_test[numerical_cols] = imputer.transform(X_test[numerical_cols]) # Impute only numerical features

        scaler = StandardScaler()
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols]) # Scale only numerical features
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols]) # Scale only numerical features

        # Save the scaler
        scaler_path = os.path.join(save_dir, f'scaler_{model_name}.pkl')
        jb.dump(scaler, scaler_path)

        # Apply SMOTE to handle class imbalance (AFTER splitting and scaling)
        smote = SMOTE(random_state=42)
        X_train, Y_train = smote.fit_resample(X_train, Y_train)

        # Create and Train Random Forest Model
        rf_model = RandomForestClassifier(random_state=42) # Add hyperparameters here if you want to tune them
        rf_model.fit(X_train, Y_train)

        # Make predictions
        y_pred = rf_model.predict(X_test)

        # Evaluate the model
        print(f"\nClassification Report for {model_name}:")
        print(classification_report(Y_test, y_pred))

        # Compute AUC-ROC Score (using predict_proba)
        auc = roc_auc_score(Y_test, rf_model.predict_proba(X_test)[:, 1])  # Correct way to get probabilities for AUC
        print(f'\nAUC-ROC Score for {model_name}: {auc:.4f}')

        # Compute Confusion Matrix
        conf_matrix = confusion_matrix(Y_test, y_pred)
        print(f"\nConfusion Matrix for {model_name}:\n", conf_matrix)

        # Evaluate (alternative - accuracy only for RF)
        accuracy = rf_model.score(X_test, Y_test)
        print(f'Test Accuracy for {model_name}: {accuracy * 100:.2f}%')

        # Save the model (using joblib)
        model_path = os.path.join(save_dir, f'{model_name}.pkl')  # Changed file extension to .pkl for joblib
        jb.dump(rf_model, model_path)
        print(f"Random Forest model for {model_name} saved at: {model_path}\n")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Train separate models for Breast Cancer and Tumor datasets
preprocess_and_train('data/Breast_cancer.csv', 'diagnosis', 'breast_cancer_model')
preprocess_and_train('data/tumor.csv', 'Class', 'tumor_model')