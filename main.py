from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import json
import io
from PIL import Image
from typing import List
import logging
import os
from fastapi.responses import JSONResponse
import joblib as jb  # ✅ Use this format
import joblib

# Define paths
original_model_path = "Model/limit-input/model/output/breast_cancer_model.pkl"
fixed_model_path = "Model/limit-input/model/output/breast_cancer_model_fixed.pkl"

try:
    # Load the original model file
    model_tuple = joblib.load(original_model_path)
    print(f"✅ Model loaded successfully: {type(model_tuple)}")

    # Check if it's a tuple and extract the actual ML model
    if isinstance(model_tuple, tuple):
        print("🚨 Model is a tuple, extracting the correct ML model...")
        actual_model = model_tuple[0]  # First element is the actual model

        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(fixed_model_path), exist_ok=True)

        # Save only the actual ML model
        joblib.dump(actual_model, fixed_model_path)
        print(f"✅ Fixed model saved successfully at: {fixed_model_path}")

    else:
        print("✅ Model is already a valid ML model, no extraction needed.")

except Exception as e:
    print(f"❌ Error: {e}")

# Load the incorrect model (tuple)
model_tuple = joblib.load("Model/limit-input/model/output/breast_cancer_model.pkl")

# Extract the actual model (RandomForestClassifier)
actual_model = model_tuple[0]  # First element of the tuple is the ML model

# Save only the model back
fixed_model_path = "Model/limit-input/model/output/breast_cancer_model_fixed.pkl"
joblib.dump(actual_model, fixed_model_path)

print(f"✅ Fixed model saved successfully at: {fixed_model_path}")

app = FastAPI()
# ✅ Fix: Enable CORS Middleware (Allow OPTIONS for preflight)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # ✅ Allow ALL HTTP methods, including OPTIONS
    allow_headers=["*"],  # Allow all headers
)

# ✅ Fix: Explicitly Handle OPTIONS Request for /predict/
@app.options("/predict/")
async def options_predict():
    return JSONResponse(content={"message": "OK"}, status_code=200)


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
        limit_model = jb.load("Model/limit-input/model/output/breast_cancer_model_fixed.pkl")
        logger.info(f"[INFO] Limit model loaded successfully: {type(limit_model).__name__}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load limit model: {e}")
        limit_model = None

   

# Load Models
def load_models():
    global limit_model, text_model, scalar
    try:
        model = jb.load('Model/limit-input/model/output/breast_cancer_model.pkl')
        
        # ✅ Automatically extract correct model if it's a tuple
        if isinstance(model, tuple):
            logger.warning("[WARNING] Model is a tuple, extracting actual model...")
            limit_model = model[0]  # Extract actual ML model
        else:
            limit_model = model  # Use as-is
        
        logger.info(f"[INFO] Limit model loaded successfully: {type(limit_model).__name__}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load limit model: {e}")
        limit_model = None

    try:
        text_model = jb.load('Model/Models/breast_cancer/breast_cancer_model.pkl')
        scalar = jb.load('Model/Models/breast_cancer/scaler_breast_cancer_model.pkl')
        logger.info("[INFO] Breast cancer model and scaler loaded successfully.")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load breast cancer model: {e}")
        text_model, scalar = None, None


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



@app.post("/predict/")
async def predict(input_data: dict):
    try:
        if "age" in input_data and "symptoms" in input_data:
            age = float(input_data["age"])  # Keep age as a number
            symptoms = [float(s) for s in input_data["symptoms"]]

            expected_symptoms = 8  # Now we expect 10 symptoms
            if not limit_model:
                raise HTTPException(status_code=503, detail="Limit model not loaded.")
            if len(symptoms) != expected_symptoms:
                raise HTTPException(status_code=400, detail=f"Limit model expects {expected_symptoms} symptoms, but got {len(symptoms)}.")

            # ✅ Prepare input for prediction (age is separate, symptoms follow)
            input_array = [age] + symptoms
            logger.info(f"[INFO] Model input (Age + Symptoms): {input_array}")

            try:
                data = np.array(input_array).reshape(1, -1)
                logger.info(f"[INFO] Reshaped Data for Prediction: {data.shape}")

                if hasattr(limit_model, 'predict_proba'):
                    proba = limit_model.predict_proba(data)
                    prediction = float(proba[0][1]) if proba.shape[1] >= 2 else float(proba[0][0])
                elif hasattr(limit_model, 'predict'):
                    pred = limit_model.predict(data)
                    prediction = float(pred[0]) if isinstance(pred, np.ndarray) else float(pred)
                else:
                    prediction = 0.5
                    logger.warning("[WARNING] Unknown model type, using default prediction value.")
            except Exception as model_error:
                logger.error(f"[ERROR] Model prediction failed: {model_error}")
                prediction = 0.5

            prediction = max(0.0, min(1.0, prediction))
            result = "High Risk" if prediction > 0.5 else "Low Risk"

            return {
                "model": "Limit Model",
                "diagnosis": result,
                "confidence": f"{prediction * 100:.2f}%",
                "raw_prediction": prediction
            }

        else:
            raise HTTPException(status_code=400, detail="Invalid input format.")
    except Exception as e:
        logger.error(f"[ERROR] Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")



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
        print("Received input data:", input_data)

        
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