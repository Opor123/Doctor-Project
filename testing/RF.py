import numpy as np
import pandas as pd
import joblib as jb
from sklearn.metrics import classification_report
from tabulate import tabulate  # For better table formatting

# Load the trained Random Forest model
model_path = "Model/Models/RF/breast_cancer_rf_model.pkl"
model = jb.load(model_path)
print(f"\n✅ Model loaded successfully from: {model_path}")

# Load the scaler
scaler_path = "Model/Models/RF/scaler_breast_cancer_rf_model.pkl"

try:
    scaler = jb.load(scaler_path)
    print(f"✅ Scaler loaded successfully from: {scaler_path}")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    exit()  # Exit if scaler loading fails

# Load datasets
original_data = pd.read_csv("testing/expanded_testing.csv")
modified_data = pd.read_csv("testing/modified_dataset.csv")

# Normalize data
original_scaled = scaler.transform(original_data)
modified_scaled = scaler.transform(modified_data)

# Get predictions
original_preds = model.predict(original_scaled)
modified_preds = model.predict(modified_scaled)

# Convert predictions to binary labels
original_labels = (original_preds > 0.5).astype(int)  # No need to threshold for RF
modified_labels = (modified_preds > 0.5).astype(int)  # No need to threshold for RF

# Compare predictions
comparison_df = pd.DataFrame({
    "Sample": np.arange(1, len(original_labels) + 1),
    "Original Prediction": ["Malignant (M)" if x == 1 else "Benign (B)" for x in original_labels],
    "Modified Prediction": ["Malignant (M)" if x == 1 else "Benign (B)" for x in modified_labels]
})

# Print comparison table
print("\n🔍 **Prediction Comparison:**")
print(tabulate(comparison_df, headers="keys", tablefmt="pretty"))

# Check changes
changes = np.sum(original_labels != modified_labels)
print(f"\n🔄 **Number of changed predictions:** {changes}/{len(original_labels)}")

# Generate and print classification report
report = classification_report(original_labels, modified_labels, target_names=["Benign (B)", "Malignant (M)"], digits=4)
print("\n📊 **Classification Report for Modified Data:**")
print(report)
