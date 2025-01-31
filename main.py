# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load the dataset
data = pd.read_csv('data.csv')

# Prepare features and target variable
x = data.drop(columns=["diagnosis(1=m, 0=b)"])
y = data["diagnosis(1=m, 0=b)"]

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define the model
model = tf.keras.models.Sequential()

# Add layers to the model
model.add(tf.keras.layers.Dense(256, input_shape=(x_train.shape[1],), activation='sigmoid'))  # Correct input shape
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(x_train, y_train, epochs=1000, verbose=1)  # Set verbose=1 to see training progress

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss*100}, Test Accuracy: {accuracy*100}')