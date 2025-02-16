import pandas as pd
import numpy as np
import joblib as jb
import logging
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants for file paths
MODEL_PATH = 'Model/Models/breast_cancer/breast_cancer_model.keras'
SCALER_PATH = 'Model/Models/breast_cancer/scaler_breast_cancer_model.pkl'
TEST_DATA_PATH = "testing/testing.csv"

# Load model and scaler
try:
    model = load_model(MODEL_PATH)
    scaler = jb.load(SCALER_PATH)
    logging.info("Model and scaler loaded successfully.")
except FileNotFoundError:
    logging.error("Error: Model or scaler file not found.")
    exit(1)
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")
    exit(1)

def preprocess_input(data, scaler):
    """Preprocesses input data by handling missing values and scaling."""
    if 'id' in data.columns:
        data = data.drop(columns=['id'])

    # Auto-detect target column and remove it
    target_col = 'diagnosis' if 'diagnosis' in data.columns else 'Class' if 'Class' in data.columns else None
    if target_col:
        true_labels = data.pop(target_col).map({'M': 1, 'B': 0, 2: 0, 4: 1})  # Handle both classification formats
    else:
        true_labels = None

    # Handle missing features by adding them with zero values
    expected_features = getattr(scaler, 'feature_names_in_', list(data.columns))
    for feature in expected_features:
        if feature not in data.columns:
            data[feature] = 0  # Assign 0 to missing features

    # Reorder columns to match training order
    data = data[expected_features]

    # Fill any remaining missing values with column means
    data.fillna(data.mean(), inplace=True)

    # Scale the input data
    scaled_data = scaler.transform(data)
    return scaled_data, true_labels

def predict(model, data):
    """Makes predictions using the trained model."""
    try:
        predictions = (model.predict(data) > 0.5).astype(int).flatten()
        return predictions
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return None

def evaluate_predictions(true_labels, predicted_labels):
    """Evaluates predictions and prints statistics."""
    logging.info("\nEvaluation:")
    print(classification_report(true_labels, predicted_labels))
    print("Confusion Matrix:\n", confusion_matrix(true_labels, predicted_labels))
    print("Accuracy:", accuracy_score(true_labels, predicted_labels))

if __name__ == "__main__":
    try:
        test_data = pd.read_csv(TEST_DATA_PATH)
        logging.info("Test data loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Error: Test data file not found at {TEST_DATA_PATH}")
        exit(1)
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        exit(1)

    processed_data, true_labels = preprocess_input(test_data, scaler)
    if processed_data is None:
        exit(1)

    predictions = predict(model, processed_data)
    if predictions is None:
        exit(1)

    # Convert numerical predictions to meaningful labels
    diagnosis_labels = {0: 'Benign (B)', 1: 'Malignant (M)'}
    predicted_labels = [diagnosis_labels[pred] for pred in predictions]

    logging.info("Test Data Predictions:")
    for i, pred_label in enumerate(predicted_labels):
        print(f"Sample {i + 1}: Predicted Diagnosis: {pred_label}")

    # Evaluate if true labels exist
    if true_labels is not None:
        evaluate_predictions(true_labels, predictions)
