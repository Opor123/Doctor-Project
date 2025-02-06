#Import any libraries here
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score
# Load the dataset
data = pd.read_csv('Breast_cancer.csv')

# Prepare features and target variable
x = data.drop(columns=["diagnosis(1=m, 0=b)"])
y = data["diagnosis(1=m, 0=b)"]

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test=train_test_split(x,y,test_size=0.3,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

smote=SMOTE()
X_train,Y_train=smote.fit_resample(X_train,Y_train)

#cover label
Y_train=Y_train.values
Y_test=Y_test.values

# Define the model
model = Sequential()

# Add layers to the model
model.add(Dense(256, input_shape=(X_train.shape[1],), activation='relu'))  # Correct input shape
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))   # Output layer for binary classification

# Compile the model
optimize=Adam(learning_rate=0.0001)

model.compile(optimizer=optimize, loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X_train, Y_train, epochs=1000, batch_size=32, validation_split=0.2)  # Set verbose=1 to see training progress

# Prediction
y_pred=(model.predict(X_test)>0.5).astype("int32")

print(classification_report(Y_test,y_pred))

auc=roc_auc_score(Y_test,model.predict(X_test))
print(f'AUC-ROC Score: {auc:.2f}')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Test Loss: {loss*100:.2f}, Test Accuracy: {accuracy*100:.2f}')