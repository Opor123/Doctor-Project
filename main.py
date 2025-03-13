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

app = FastAPI()

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DoctorAI")

# Global variables
text_model, scalar, limit_model, image_model, responses, feature_names = None, None, None, None, None, None

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
    global limit_model
    try:
        limit_model = jb.load('Model/limit-input/model/output/breast_cancer_model.pkl')
        # Log model type for debugging
        logger.info(f"[INFO] Limit model type: {type(limit_model).__name__}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load limit model: {e}")
        limit_model = None

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
    age: float
    symptoms: List[float]  # Example: Expecting 10 symptoms

# Unified Prediction Endpoint
@app.post("/predict/")
async def predict(input_data: dict):
    try:
        if "age" in input_data and "symptoms" in input_data:
            # Validate Limit Model Input
            age = float(input_data["age"])  # Ensure age is float
            symptoms = [float(s) for s in input_data["symptoms"]]  # Ensure all symptoms are float
            expected_symptoms = 10  # Define based on your model's training data

            if not limit_model:
                raise HTTPException(status_code=503, detail="Limit model not loaded.")
            
            if len(symptoms) != expected_symptoms:
                raise HTTPException(status_code=400, detail=f"Limit model expects {expected_symptoms} symptoms.")

            # Prepare input - as a flat array first
            input_array = [age] + symptoms
            
            # Log the input data for debugging
            logger.info(f"[INFO] Limit model input: {input_array}")
            
            # Handle different model types
            model_type = type(limit_model).__name__
            logger.info(f"[INFO] Processing with model type: {model_type}")
            
            # Create a robust prediction function
            prediction = 0.0
            try:
                # Reshape data appropriately for the model
                data = np.array(input_array).reshape(1, -1)
                
                # Different approaches based on common model types
                if hasattr(limit_model, 'predict_proba'):
                    # For sklearn classifiers with predict_proba
                    proba = limit_model.predict_proba(data)
                    # Handle both binary and multi-class cases
                    if proba.shape[1] >= 2:
                        prediction = float(proba[0][1])  # Probability of positive class
                    else:
                        prediction = float(proba[0][0])
                elif hasattr(limit_model, 'predict'):
                    # For models with only predict method
                    pred = limit_model.predict(data)
                    if isinstance(pred, np.ndarray) and pred.size > 0:
                        prediction = float(pred[0])
                    else:
                        prediction = float(pred)
                else:
                    # Fallback for unknown model types
                    prediction = 0.5
                    logger.warning("[WARNING] Unknown model type, using default prediction value.")
            except Exception as model_error:
                logger.error(f"[ERROR] Model prediction failed: {model_error}")
                # Fallback to ensure API doesn't crash
                prediction = 0.5
            
            # Ensure prediction is normalized to 0-1 range for consistent results
            prediction = max(0.0, min(1.0, prediction))
            result = "High Risk" if prediction > 0.5 else "Low Risk"

            return {
                "model": "Limit Model",
                "diagnosis": result, 
                "confidence": f"{prediction * 100:.2f}%",
                "raw_prediction": float(prediction)
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

            return {"model": "Breast Cancer Model", "diagnosis": result, "confidence": f"{float(prediction) * 100:.2f}%"}

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