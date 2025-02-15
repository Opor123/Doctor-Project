import pandas as pd
import numpy as np
import joblib as jb
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Constants for file paths
MODEL_PATH = 'Model\\merge_model.keras'
SCALER_PATH = 'Model\\scaler_merge.pkl'
TEST_DATA_PATH = "testing\\test.csv"

# Load model and scaler (handle exceptions early)
try:
    model = load_model(MODEL_PATH)
    scaler = jb.load(SCALER_PATH)
except FileNotFoundError:
    print(f"Error: Model or scaler file not found.")
    exit(1)  # Use a non-zero exit code for errors
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    exit(1)

def preprocess_input(data, scaler):
    """Preprocesses input data for prediction.  Handles missing features and values efficiently."""

    if 'id' in data.columns:
        data = data.drop(columns=['id'])

    expected_features = getattr(scaler, 'feature_names_in_', data.columns)  # Simplified feature name retrieval

    missing_features = set(expected_features) - set(data.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    data = data[expected_features]  # Reorder columns

    # Efficient missing value handling (using mean imputation)
    data = data.fillna(data[expected_features].mean())  # No need for inplace=True if reassigning

    scaled_data = scaler.transform(data)
    return scaled_data

def predict(model, data):
    """Makes predictions."""
    try:
        predictions = (model.predict(data) > 0.5).astype(int).flatten() # More concise type conversion
        return predictions
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def evaluate_predictions(true_labels, predicted_labels):
    """Evaluates and prints prediction statistics."""
    print("\nEvaluation:")
    print(classification_report(true_labels, predicted_labels))
    print("Confusion Matrix:\n", confusion_matrix(true_labels, predicted_labels))
    print("Accuracy:", accuracy_score(true_labels, predicted_labels))


if __name__ == "__main__":
    try:
        test_data = pd.read_csv(TEST_DATA_PATH)
        true_labels = test_data.pop('diagnosis').map({'M': 1, 'B': 0}) if 'diagnosis' in test_data else None #Combined label extraction and removal
    except FileNotFoundError:
        print(f"Error: Test data file not found at {TEST_DATA_PATH}")
        exit(1)
    except Exception as e:
        print(f"Error loading test data: {e}")
        exit(1)

    processed_data = preprocess_input(test_data, scaler)
    if processed_data is None:
        exit(1)

    predictions = predict(model, processed_data)
    if predictions is None:
        exit(1)

    diagnosis_labels = {0: 'Benign (B)', 1: 'Malignant (M)'}
    predicted_labels = [diagnosis_labels[pred] for pred in predictions]

    print("Test Data Predictions:")
    for i, pred_label in enumerate(predicted_labels):
        print(f"Sample {i + 1}: Predicted Diagnosis: {pred_label}")

    if true_labels is not None:
        evaluate_predictions(true_labels, predictions)