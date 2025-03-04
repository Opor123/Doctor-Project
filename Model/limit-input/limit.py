import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("data/Breast_cancer/breast-cancer-age.csv")

# Selecting features and target
features = ['Age', 'Menopause', 'Tumor-Size', 'Inv-Nodes', 'Node-Caps', 'Deg-Malig', 'Breast', 'Breast-Quad', 'Irradiat']
X = df[features]
y = df['Class']

# Convert Age and Tumor-Size to numerical values (extract lower bound of the range)
def extract_numeric(value):
    return int(value.split('-')[0]) if '-' in value else int(value)

X['Age'] = X['Age'].apply(extract_numeric)
X['Tumor-Size'] = X['Tumor-Size'].apply(extract_numeric)

# Encode categorical features
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Standardize numerical feature
scaler = StandardScaler()
X[['Deg-Malig']] = scaler.fit_transform(X[['Deg-Malig']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and preprocessing tools
with open("limited_input_model.pkl", "wb") as f:
    pickle.dump((model, label_encoders, scaler), f)

print("Model and preprocessing tools saved as 'limited_input_model.pkl'")
