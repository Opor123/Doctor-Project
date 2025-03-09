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
TEST_DATA_PATH = "testing/modified_dataset.csv"

# Load model and scaler
try:
    model = load_model(MODEL_PATH)
    scaler = jb.load(SCALER_PATH)
    logging.info("✅ Model and scaler loaded successfully.")
except FileNotFoundError:
    logging.error("❌ Error: Model or scaler file not found.")
    exit(1)
except Exception as e:
    logging.error(f"❌ Error loading model or scaler: {e}")
    exit(1)

def preprocess_input(data, scaler):
    """Preprocesses input data by handling missing values and scaling."""
    if 'id' in data.columns:
        data = data.drop(columns=['id'])

    # Auto-detect target column and remove it
    target_col = 'diagnosis' if 'diagnosis' in data.columns else 'Class' if 'Class' in data.columns else None
    if target_col:
        true_labels = data.pop(target_col).map({'M': 1, 'B': 0, 2: 0, 4: 1})  # Handle different label formats
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
    """Makes predictions using the trained model and returns probabilities."""
    try:
        probabilities = model.predict(data) * 100  # Convert to percentage
        return probabilities.flatten()
    except Exception as e:
        logging.error(f"❌ Prediction error: {e}")
        return None

def evaluate_predictions(true_labels, probabilities):
    """Evaluates predictions and prints statistics."""
    predicted_labels = (probabilities >= 50).astype(int)  # Convert probabilities to binary labels

    logging.info("\n📊 Evaluation Metrics:")
    print(classification_report(true_labels, predicted_labels))
    print("📌 Confusion Matrix:\n", confusion_matrix(true_labels, predicted_labels))
    print(f"🎯 Accuracy: {accuracy_score(true_labels, predicted_labels) * 100:.2f}%")

if __name__ == "__main__":
    try:
        test_data = pd.read_csv(TEST_DATA_PATH)
        logging.info("✅ Test data loaded successfully.")
    except FileNotFoundError:
        logging.error(f"❌ Error: Test data file not found at {TEST_DATA_PATH}")
        exit(1)
    except Exception as e:
        logging.error(f"❌ Error loading test data: {e}")
        exit(1)

    processed_data, true_labels = preprocess_input(test_data, scaler)
    if processed_data is None:
        exit(1)

    probabilities = predict(model, processed_data)
    if probabilities is None:
        exit(1)

    # Convert numerical predictions to meaningful labels
    logging.info("📌 Test Data Predictions:")
    for i, prob in enumerate(probabilities[:60]):  # Display first 10 predictions
        classification = "Malignant (M)" if prob >= 50 else "Benign (B)"
        print(f"🔹 Sample {i+1}: {prob:.2f}% Malignant | Predicted: {classification}")

    # Evaluate if true labels exist
    if true_labels is not None:
        evaluate_predictions(true_labels, probabilities)