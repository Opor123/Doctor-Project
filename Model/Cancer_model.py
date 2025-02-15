import pandas as pd
import numpy as np
import os
import json
import joblib as jb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Create directory to save model
save = 'Model'
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

# Reapply imputation to ensure no missing values
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
scaler_path = os.path.join(save, 'scaler_merge.pkl')
jb.dump(scaler, scaler_path)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train, Y_train = smote.fit_resample(X_train, Y_train)

# Define the deep learning model
model = Sequential([
    Dense(256, input_shape=(X_train.shape[1],), activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

# Define callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# Train the model
history = model.fit(
    X_train, Y_train,
    epochs=300,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Make predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Evaluate the model
print("\nClassification Report:\n", classification_report(Y_test, y_pred))

# Compute AUC-ROC Score
auc = roc_auc_score(Y_test, model.predict(X_test))
print(f'\nAUC-ROC Score: {auc:.4f}')

# Compute Confusion Matrix
conf_matrix = confusion_matrix(Y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# Evaluate the model on the test set
loss, accuracy, auc = model.evaluate(X_test, Y_test, verbose=0)
print(f'\nTest Loss: {loss * 100:.2f}%')
print(f'Test Accuracy: {accuracy * 100:.2f}%')
print(f'Test AUC: {auc * 100:.2f}%')

print("Model and scaler saved successfully!")

# Save the model
model_path = os.path.join(save, 'merge_model.keras')
model.save(model_path)
print(f"Model has been saved at: {model_path}")
