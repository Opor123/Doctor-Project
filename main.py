from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import joblib as jb
import json
import io
from PIL import Image
from typing import List
import logging
import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = FastAPI()

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DoctorAI")

# Global variables
text_model, scalar, limit_model, image_model, responses, feature_names, label_encoders, scaler_limit, y_encoder = None, None, None, None, None, None, None, None, None

# Load Models
def load_med_data_model():
    global text_model, scalar, feature_names
    try:
        text_model = tf.keras.models.load_model('Model/Models/breast_cancer/breast_cancer_model.keras')
        scalar = jb.load('Model/Models/breast_cancer/scaler_breast_cancer_model.pkl')
        with open('Model/Models/breast_cancer/feature_names_breast_cancer_model.json', 'r') as f:
            feature_names = json.load(f)
        logger.info("[INFO] Breast cancer text model loaded successfully.")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load breast cancer model: {e}")
        text_model, scalar, feature_names = None, None, None


def load_limit_model():
    global limit_model, label_encoders, scaler_limit, y_encoder
    try:
        with open("Model/limit-input/output/limit_input_model.pkl", "rb") as f:
            limit_model, label_encoders, scaler_limit, y_encoder = pickle.load(f)
        logger.info(f"[INFO] Limit model type: {type(limit_model).__name__}")
        logger.info(f"[INFO] Label encoders: {label_encoders}") # add this
        logger.info(f"[INFO] Scaler limit: {scaler_limit}") # add this
        logger.info(f"[INFO] Y encoder: {y_encoder}") #add this
    except Exception as e:
        logger.error(f"[ERROR] Failed to load limit model: {e}")
        limit_model, label_encoders, scaler_limit, y_encoder = None, None, None, None


def load_image_model():
    global image_model
    try:
        image_model = tf.keras.models.load_model('Model/Image_recognize/Model/breast_cancer_image_model_efficientnet.keras')
        logger.info("[INFO] Image recognition model loaded successfully.")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load Image model: {e}")
        image_model = None

def load_response():
    global responses
    try:
        with open('Model/main/Data/data.json', 'r') as f:
            responses = json.load(f)
        logger.info("[INFO] Response Data loaded successfully.")
    except Exception as e:
        logger.error(f"[ERROR] Could not load Responses: {e}")
        responses = None

def health_data():
    try:
        with open("data/cancer_data/cancer_prevention.json", 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"[ERROR] Could not load health tips data: {e}")
        return []

def initialize():
    try:
        # Create directory structure if it doesn't exist
        os.makedirs('Model/Models/breast_cancer', exist_ok=True)
        os.makedirs('Model/limit-input/model/output', exist_ok=True)
        os.makedirs('Model/Image_recognize/Model', exist_ok=True)
        os.makedirs('Model/main/Data', exist_ok=True)
        os.makedirs('data/cancer_data', exist_ok=True)
        
        # Load models
        load_image_model()
        load_med_data_model()
        load_limit_model()
        load_response()
        logger.info("\n✅ Doctor AI is ready!")
    except Exception as e:
        logger.error(f"[ERROR] Initialization failed: {e}")
        logger.info("⚠️ Doctor AI is running in limited mode. Some features may not be available.")

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    initialize()

# Model Input Classes
class BreastCancerInput(BaseModel):
    symptoms: List[float]  # Expecting 30 symptoms

class LimitModelInput(BaseModel):
    Age: int
    Menopause: str
    Tumor_Size: str
    Inv_Nodes: str
    Node_Caps: str
    Deg_Malig: int
    Breast: str
    Breast_Quad: str
    Irradiat: str

# Unified Prediction Endpoint
@app.post("/predict/")
async def predict(input_data: dict):
    try:
        if all(key in input_data for key in ["Age", "Menopause", "Tumor_Size", "Inv_Nodes", "Node_Caps", "Deg_Malig", "Breast", "Breast_Quad", "Irradiat"]):
            # Handle underscores to hyphens conversion
            hyphen_keys = ["Tumor-Size", "Inv-Nodes", "Node-Caps", "Deg-Malig", "Breast-Quad"]
            underscore_keys = ["Tumor_Size", "Inv_Nodes", "Node_Caps", "Deg_Malig", "Breast_Quad"]
            
            df_data = {}
            for uk, hk in zip(underscore_keys, hyphen_keys):
                if uk in input_data:
                    df_data[hk] = input_data[uk]
                elif hk in input_data:  # In case data is already sent with hyphens
                    df_data[hk] = input_data[hk]
            
            # Add remaining fields
            for key in ["Age", "Menopause", "Breast", "Irradiat"]:
                if key in input_data:
                    df_data[key] = input_data[key]
            
            # Create DataFrame
            df = pd.DataFrame([df_data])
            
            logger.info(f"Limit Model Input DataFrame: {df}")

            # Convert numerical features
            for col in ['Age', 'Tumor-Size', 'Inv-Nodes']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda val: int(val.split('-')[0]) if isinstance(val, str) and '-' in val else int(val))

            logger.info(f"Limit Model DataFrame after numerical conversion: {df}")

            # Fill missing values
            df.fillna(df.median(numeric_only=True), inplace=True)

            logger.info(f"Limit Model DataFrame after filling missing values: {df}")

            # Encode categorical features
            for col, le in label_encoders.items():
                if col in df.columns:
                    df[col] = le.transform(df[col])

            logger.info(f"Limit Model DataFrame after encoding categorical features: {df}")

            # Feature Engineering
            df["Severity_Score"] = df["Tumor-Size"] * df["Deg-Malig"]
            df["Age_Group"] = pd.cut(df["Age"], bins=[0, 39, 59, 100], labels=["Young", "Middle", "Old"])
            df["Age_Group"] = LabelEncoder().fit_transform(df["Age_Group"])
            df["Tumor-Node_Caps"] = df["Tumor-Size"] * (df["Node-Caps"] == 1).astype(int)

            logger.info(f"Limit Model DataFrame after feature engineering: {df}")

            # Standardize numerical features
            numerical_cols = ['Age', 'Tumor-Size', 'Inv-Nodes', 'Deg-Malig', 'Severity_Score', 'Tumor-Node_Caps']
            df[numerical_cols] = scaler_limit.transform(df[numerical_cols])

            logger.info(f"Limit Model DataFrame after scaling: {df}")

            # Make prediction
            prediction_proba = limit_model.predict_proba(df)[:, 1][0]
            prediction = limit_model.predict(df)[0]
            
            # Get clear diagnosis from model prediction
            diagnosis = y_encoder.inverse_transform([prediction])[0]
            # Ensure diagnosis is clearly "Malignant" or "Benign"
            if diagnosis.lower() not in ["malignant", "benign"]:
                diagnosis = "Malignant" if prediction_proba > 0.5 else "Benign"
                
            confidence = f"{prediction_proba * 100:.2f}%"

            return {
                "model": "Limit Model",
                "diagnosis": diagnosis,
                "is_malignant": diagnosis.lower() == "malignant",
                "confidence": confidence,
                "raw_prediction": float(prediction_proba)
            }

        elif "symptoms" in input_data:
            # Validate Breast Cancer Model Input
            symptoms = [float(s) for s in input_data["symptoms"]]  # Ensure all symptoms are float
            expected_symptoms = 30  # Define based on your model's training data

            if not text_model or not scalar:
                raise HTTPException(status_code=503, detail="Breast cancer model not loaded.")
            if len(symptoms) != expected_symptoms:
                raise HTTPException(status_code=400, detail=f"Breast cancer model expects {expected_symptoms} symptoms.")

            # Prepare input
            data = np.array(symptoms).reshape(1, -1)
            data = scalar.transform(data)
            prediction = text_model.predict(data)[0][0]
            result = "Malignant" if prediction > 0.5 else "Benign"

            return {
                "model": "Breast Cancer Model", 
                "diagnosis": result, 
                "is_malignant": result.lower() == "malignant",
                "confidence": f"{float(prediction) * 100:.2f}%"
            }

        else:
            raise HTTPException(status_code=400, detail="Invalid input format.")

    except Exception as e:
        logger.error(f"[ERROR] Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
# Image Prediction Endpoint

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    if not image_model:
        raise HTTPException(status_code=503, detail="Image model not loaded.")
    
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB').resize((640, 640))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        
        prediction = image_model.predict(image)
        result = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0][result]) * 100
        
        diagnosis = "Malignant" if result == 1 else "Benign"
        
        return {
            "diagnosis": diagnosis,
            "is_malignant": diagnosis.lower() == "malignant",
            "confidence": f"{confidence:.2f}%",
            "result_code": int(result)
        }
    except Exception as e:
        logger.error(f"[ERROR] Image prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")
    
# Health Tips Endpoint
@app.get("/health-tips/")
def get_health_tips():
    tips = health_data()
    if not tips:
        raise HTTPException(status_code=404, detail="Health tips not available")
    return {"health_tips": tips}

# Model Status Endpoint
@app.get("/status/")
def model_status():
    status = {
        "text_model": "Loaded" if text_model else "Not loaded",
        "limit_model": "Loaded" if limit_model else "Not loaded",
        "image_model": "Loaded" if image_model else "Not loaded",
        "responses": "Loaded" if responses else "Not loaded"
    }
    
    # Add more details about the limit model if it's loaded
    if limit_model:
        status["limit_model_type"] = type(limit_model).__name__
        status["limit_model_methods"] = [method for method in dir(limit_model) if not method.startswith('_') and callable(getattr(limit_model, method))]
    
    return status

# Debug Endpoint for Limit Model
@app.post("/debug-limit-model/")
async def debug_limit_model(input_data: dict):
    if not limit_model:
        raise HTTPException(status_code=503, detail="Limit model not loaded.")
    
    try:
        age = float(input_data.get("age", 45))
        symptoms = [float(s) for s in input_data.get("symptoms", [0.5] * 10)]
        
        # Log model details
        model_info = {
            "type": type(limit_model).__name__,
            "dir": [m for m in dir(limit_model) if not m.startswith('_')],
            "has_predict": hasattr(limit_model, 'predict'),
            "has_predict_proba": hasattr(limit_model, 'predict_proba')
        }
        
        # Try different input formats
        input_array = [age] + symptoms
        data_flat = np.array(input_array)
        data_2d = np.array(input_array).reshape(1, -1)
        
        results = {"model_info": model_info, "input": input_array}
        
        # Try different prediction methods
        try:
            if hasattr(limit_model, 'predict'):
                results["predict_flat"] = float(limit_model.predict(data_flat))
            if hasattr(limit_model, 'predict'):
                results["predict_2d"] = float(limit_model.predict(data_2d)[0])
            if hasattr(limit_model, 'predict_proba'):
                proba = limit_model.predict_proba(data_2d)
                results["predict_proba"] = [float(p) for p in proba[0]]
        except Exception as e:
            results["error"] = str(e)
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

# Home Route
@app.get("/")
def home():
    return {"message": "Welcome to Doctor AI!", "status": "active"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)