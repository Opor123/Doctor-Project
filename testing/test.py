# Import necessary libraries
import pandas as pd
import numpy as np
import joblib as jb
from tensorflow.keras.models import load_model

# Load the saved model and scaler
model = load_model('merge_model.keras')
scaler = jb.load('scaler_merge.pkl')

# Define a function to preprocess new input data
def preprocess_input(data, scaler):
    """
    Preprocess the input data for prediction.
    Args:
        data (pd.DataFrame): New input data.
        scaler (StandardScaler): Fitted scaler object.
    Returns:
        np.array: Scaled and preprocessed data.
    """
    # Drop unnecessary columns (if present)
    if 'id' in data.columns:
        data = data.drop(columns=['id'])
    
    # Ensure the test data has the same columns as the training data
    expected_features = scaler.feature_names_in_  # Features used during fitting
    missing_features = set(expected_features) - set(data.columns)
    
    if missing_features:
        raise ValueError(f"Test data is missing the following features used during training: {missing_features}")
    
    # Reorder columns to match the training data
    data = data[expected_features]
    
    # Handle missing values (if any)
    data.fillna(data.mean(), inplace=True)
    
    # Standardize the features using the pre-fitted scaler
    scaled_data = scaler.transform(data)
    
    return scaled_data

# Define a function to make predictions
def predict(model, data):
    """
    Make predictions using the trained model.
    Args:
        model (keras.Model): Trained model.
        data (np.array): Preprocessed input data.
    Returns:
        np.array: Predicted classes (0 or 1).
    """
    predictions = (model.predict(data) > 0.5).astype("int32")
    return predictions.flatten()

# Example test case
if __name__ == "__main__":
# Example input data (replace with your own data)
    # Load test data from CSV file
    test_data = pd.read_csv("testing\\test.csv")

    # Preprocess the test data
    try:
        processed_data = preprocess_input(test_data, scaler)
    except ValueError as e:
        print(f"Error: {e}")
        exit()

    # Make predictions
    predictions = predict(model, processed_data)

    # Map predictions to class labels
    diagnosis_labels = {0: 'Benign (B)', 1: 'Malignant (M)'}
    predicted_labels = [diagnosis_labels[pred] for pred in predictions]

    # Display results
    print("Test Data Predictions:")
    for i, (input_data, pred_label) in enumerate(zip(test_data.values, predicted_labels)):
        print(f"Sample {i + 1}:")
        print(f"  Predicted Diagnosis: {pred_label}")
        print()
