import json
import joblib as jb
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, confusion_matrix

def test_model_with_dataset(model_name, test_file_path, target_col):
    """Load a trained model and test it on a different dataset"""

    # Load the trained model
    model = tf.keras.models.load_model(model_name)

    # Load the test dataset
    test_data = pd.read_csv(test_file_path)

    # Print columns for debugging
    print("Columns in test dataset:", test_data.columns)

    # Drop unnecessary columns
    if 'id' in test_data.columns or 'Sample code number' in test_data.columns:
        test_data = test_data.drop(columns=['id', 'Sample code number'], errors='ignore')

    # Convert target variable (if necessary)
    if target_col in test_data.columns:
        test_data[target_col] = test_data[target_col].map({'M': 1, 'B': 0, 2: 0, 4: 1})
    else:
        print(f"Warning: Target column '{target_col}' not found in test dataset.")
        return  # Exit the function, or handle the missing column appropriately

    # Separate features and target variable
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]

    # Load feature names used during training
    with open(f'feature_names_{model_name}.json', 'r') as f:
        feature_names = json.load(f)

    # Ensure test dataset has the same features as training dataset
    X_test = X_test[feature_names]

    # Load the trained scaler
    scaler = jb.load(f'scaler_{model_name}.pkl')

    # Standardize the test dataset
    X_test = scaler.transform(X_test)

    # Make predictions (as percentage)
    y_pred_proba = model.predict(X_test) * 100  # Convert to percentage

    # Display probabilities for the first 10 samples
    print(f"\nTesting {model_name} with {test_file_path}")
    for i, prob in enumerate(y_pred_proba[:10]):  # Show first 10 predictions
        print(f"Sample {i+1}: {prob[0]:.2f}% Malignant")

    # Compute AUC-ROC Score
    auc = roc_auc_score(y_test, model.predict(X_test))
    print(f'\nAUC-ROC Score for {model_name}: {auc:.4f}')

    # Compute Confusion Matrix (using threshold 50% for classification)
    y_pred = (y_pred_proba >= 50).astype("int32")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix for {model_name}:\n", conf_matrix)

# Example: Testing Tumor Model with Breast Cancer Dataset
target_column_name = 'diagnosis'  # MUST match the actual target column name
test_model_with_dataset('Model/Models/breast_cancer/breast_cancer_model.keras', 'testing/modified_dataset.csv', target_column_name)