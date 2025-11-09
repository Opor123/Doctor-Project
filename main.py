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
text_model, scalar, limit_model, image_model = None, None, None, None
responses, feature_names = None, None
label_encoders, scaler_limit, y_encoder = None, None, None

# ------------------------- Loaders -------------------------
def load_med_data_model():
    global text_model, scalar, feature_names
    try:
        text_model = tf.keras.models.load_model('Model/Breast/Models/brest_cancer/breast_cancer_model.keras')
        scalar = jb.load('Model/Breast/Models/brest_cancer/scaler_breast_cancer_model.pkl')
        with open('Model/Breast/Models/brest_cancer/feature_names_breast_cancer_model.json', 'r') as f:
            feature_names = json.load(f)
        logger.info("[INFO] Breast cancer text model loaded successfully.")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load breast cancer model: {e}")

def load_limit_model():
    global limit_model, label_encoders, scaler_limit, y_encoder
    try:
        with open("Model/Breast_cancer/limit-input/output/limit_input_model.pkl", "rb") as f:
            limit_model, label_encoders, scaler_limit, y_encoder = pickle.load(f)
        logger.info("[INFO] Limit model loaded successfully.")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load limit model: {e}")

def load_image_model():
    global image_model
    try:
        image_model = tf.keras.models.load_model('Model/Breast_cancer/Image_recognize/Model/breast_cancer_image_model_efficientnet.keras')
        logger.info("[INFO] Image recognition model loaded successfully.")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load Image model: {e}")

def load_response():
    global responses
    try:
        with open('Model/main/Data/data.json', 'r') as f:
            responses = json.load(f)
        logger.info("[INFO] Response Data loaded successfully.")
    except Exception as e:
        logger.error(f"[ERROR] Could not load Responses: {e}")

def health_data():
    try:
        with open("data/cancer_data/cancer_prevention.json", 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"[ERROR] Could not load health tips data: {e}")
        return []

def initialize():
    os.makedirs('Model/Breast_cancer/Models/breast_cancer', exist_ok=True)
    os.makedirs('Model/Breast_cancer/limit-input/model/output', exist_ok=True)
    os.makedirs('Model/Breast_cancer/Image_recognize/Model', exist_ok=True)
    os.makedirs('Model/main/Data', exist_ok=True)
    os.makedirs('data/cancer_data', exist_ok=True)

    load_image_model()
    load_med_data_model()
    load_limit_model()
    load_response()
    logger.info("\n✅ Doctor AI is ready!")

@app.on_event("startup")
async def startup_event():
    initialize()

# ------------------------- Input Models -------------------------
class BreastCancerInput(BaseModel):
    symptoms: List[float]

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

# ------------------------- Endpoints -------------------------
@app.post("/predict/breast-cancer/")
async def predict_breast_cancer(input_data: BreastCancerInput):
    try:
        if not text_model or not scalar:
            raise HTTPException(status_code=503, detail="Breast cancer model not loaded.")
        if len(input_data.symptoms) != 30:
            raise HTTPException(status_code=400, detail="Exactly 30 symptom features are required.")

        # Prepare the data
        data = np.array(input_data.symptoms, dtype=np.float32).reshape(1, -1)
        
        # Debug: Check input data
        logger.info(f"Input data range: min={data.min():.4f}, max={data.max():.4f}")
        
        # Scale the data
        data_scaled = scalar.transform(data)
        
        # Debug: Check scaled data - this is where the problem is!
        logger.info(f"Scaled data range: min={data_scaled.min():.4f}, max={data_scaled.max():.4f}")
        
        # Check if scaling produced extreme values
        if np.abs(data_scaled).max() > 10:
            logger.warning(f"Extreme scaling detected! Max absolute value: {np.abs(data_scaled).max():.2f}")
            logger.warning("This suggests the scaler was trained on different data or there's a data format mismatch.")
            
            # Option 1: Try to clip extreme values
            data_scaled_clipped = np.clip(data_scaled, -5, 5)
            logger.info(f"Clipped scaled data range: min={data_scaled_clipped.min():.4f}, max={data_scaled_clipped.max():.4f}")
            
            # Use clipped data for prediction
            prediction = text_model.predict(data_scaled_clipped, verbose=0)
            logger.info("Used clipped scaling for prediction")
        else:
            # Normal prediction
            prediction = text_model.predict(data_scaled, verbose=0)
        
        # Debug: Log raw prediction details
        logger.info(f"Raw prediction: {prediction[0][0]}")
        
        # Handle the prediction
        raw_prob = float(prediction[0][0])
        
        # The model outputs very small numbers due to scaling issues
        # We need to handle this more intelligently
        if raw_prob < 1e-10:  # Very small number
            logger.warning(f"Very small prediction value: {raw_prob}")
            # This indicates the model is very confident it's benign
            # But we should still provide a more reasonable confidence
            confidence_malignant = max(raw_prob, 1e-6)  # At least 0.0001%
        elif raw_prob > (1 - 1e-10):  # Very close to 1
            logger.warning(f"Very large prediction value: {raw_prob}")
            # This indicates the model is very confident it's malignant
            confidence_malignant = min(raw_prob, 1 - 1e-6)  # At most 99.9999%
        else:
            confidence_malignant = raw_prob
        
        confidence_benign = 1 - confidence_malignant
        
        # Determine diagnosis
        threshold = 0.5
        result = "Malignant" if confidence_malignant > threshold else "Benign"
        confidence_percentage = max(confidence_malignant, confidence_benign) * 100
        
        # Special handling for very confident predictions
        if raw_prob < 1e-6:
            # Very confident it's benign
            confidence_display = 99.99  # Show as very confident but not 100%
            confidence_malignant_display = 0.01
            confidence_benign_display = 99.99
        elif raw_prob > (1 - 1e-6):
            # Very confident it's malignant  
            confidence_display = 99.99
            confidence_malignant_display = 99.99
            confidence_benign_display = 0.01
        else:
            confidence_display = confidence_percentage
            confidence_malignant_display = confidence_malignant * 100
            confidence_benign_display = confidence_benign * 100
        
        logger.info(f"Final diagnosis: {result}, Display confidence: {confidence_display:.2f}%")

        return {
            "model": "Breast Cancer Model",
            "diagnosis": result,
            "is_malignant": result.lower() == "malignant",
            "confidence": f"{confidence_display:.2f}%",
            "probability_malignant": f"{confidence_malignant_display:.2f}%",
            "probability_benign": f"{confidence_benign_display:.2f}%",
            "raw_prediction": float(raw_prob),
            "scaling_warning": np.abs(data_scaled).max() > 10,
            "debug_info": {
                "raw_model_output": float(raw_prob),
                "scaled_data_range": {
                    "min": float(data_scaled.min()),
                    "max": float(data_scaled.max())
                },
                "scaling_max_abs": float(np.abs(data_scaled).max())
            }
        }
    except Exception as e:
        logger.error(f"[ERROR] Breast cancer prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/limit-model/")
async def predict_limit_model(input_data: LimitModelInput):
    try:
        if not limit_model:
            raise HTTPException(status_code=503, detail="Limit model not loaded.")

        # Create DataFrame from input
        df = pd.DataFrame([input_data.dict()])
        df.rename(columns={
            "Tumor_Size": "Tumor-Size",
            "Inv_Nodes": "Inv-Nodes",
            "Node_Caps": "Node-Caps",
            "Deg_Malig": "Deg-Malig",
            "Breast_Quad": "Breast-Quad"
        }, inplace=True)

        # Process numerical ranges
        for col in ['Tumor-Size', 'Inv-Nodes']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: int(x.split('-')[0]) if isinstance(x, str) and '-' in x else int(x))
        
        df["Age"] = df["Age"].astype(int)
        df.fillna(df.median(numeric_only=True), inplace=True)

        # Apply label encoders
        for col, le in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = le.transform(df[col])
                except ValueError as e:
                    logger.warning(f"Unknown label for column {col}: {df[col].iloc[0]}. Using most frequent class.")
                    # Use the most frequent class if unknown label
                    df[col] = 0

        # Feature engineering
        df["Severity_Score"] = df["Tumor-Size"] * df["Deg-Malig"]
        df["Age_Group"] = pd.cut(df["Age"], bins=[0, 39, 59, 100], labels=["Young", "Middle", "Old"])
        age_group_encoder = LabelEncoder()
        df["Age_Group"] = age_group_encoder.fit_transform(df["Age_Group"])
        df["Tumor-Node_Caps"] = df["Tumor-Size"] * (df["Node-Caps"] == 1).astype(int)

        # Scale numerical features
        numerical_cols = ['Age', 'Tumor-Size', 'Inv-Nodes', 'Deg-Malig', 'Severity_Score', 'Tumor-Node_Caps']
        existing_numerical_cols = [col for col in numerical_cols if col in df.columns]
        df[existing_numerical_cols] = scaler_limit.transform(df[existing_numerical_cols])

        # Get predictions
        if hasattr(limit_model, 'predict_proba'):
            prediction_proba = limit_model.predict_proba(df)[0]
            if len(prediction_proba) == 2:
                probability_malignant = float(prediction_proba[1])
                probability_benign = float(prediction_proba[0])
            else:
                probability_malignant = float(prediction_proba[0])
                probability_benign = 1 - probability_malignant
        else:
            # If no predict_proba, use predict and assume binary output
            prediction_raw = limit_model.predict(df)[0]
            probability_malignant = float(prediction_raw) if prediction_raw <= 1 else 1.0
            probability_benign = 1 - probability_malignant

        # Determine diagnosis
        prediction_class = limit_model.predict(df)[0]
        
        if y_encoder is not None:
            try:
                diagnosis = y_encoder.inverse_transform([prediction_class])[0]
            except:
                diagnosis = "Malignant" if probability_malignant > 0.5 else "Benign"
        else:
            diagnosis = "Malignant" if probability_malignant > 0.5 else "Benign"

        # Ensure diagnosis is properly formatted
        if diagnosis.lower() not in ["malignant", "benign"]:
            diagnosis = "Malignant" if probability_malignant > 0.5 else "Benign"

        confidence_percentage = max(probability_malignant, probability_benign) * 100
        
        logger.info(f"Limit model prediction - Probabilities: [{probability_benign:.3f}, {probability_malignant:.3f}], Result: {diagnosis}")

        return {
            "model": "Limit Model",
            "diagnosis": diagnosis,
            "is_malignant": diagnosis.lower() == "malignant",
            "confidence": f"{confidence_percentage:.2f}%",
            "probability_malignant": f"{probability_malignant * 100:.2f}%",
            "probability_benign": f"{probability_benign * 100:.2f}%",
            "raw_prediction": float(probability_malignant)
        }
    except Exception as e:
        logger.error(f"[ERROR] Limit model prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    if not image_model:
        raise HTTPException(status_code=503, detail="Image model not loaded.")
    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB').resize((640, 640))
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_batch = np.expand_dims(image_array, axis=0)

        # Get prediction
        prediction = image_model.predict(image_batch, verbose=0)
        
        # Handle different output formats
        if prediction.shape[-1] == 1:
            # Single output (sigmoid activation)
            probability_malignant = float(prediction[0][0])
            probability_benign = 1 - probability_malignant
            result_code = 1 if probability_malignant > 0.5 else 0
        elif prediction.shape[-1] == 2:
            # Two outputs (softmax activation)
            probability_benign = float(prediction[0][0])
            probability_malignant = float(prediction[0][1])
            result_code = np.argmax(prediction[0])
        else:
            raise ValueError(f"Unexpected model output shape: {prediction.shape}")

        # Determine diagnosis
        diagnosis = "Malignant" if result_code == 1 else "Benign"
        confidence_percentage = max(probability_malignant, probability_benign) * 100
        
        logger.info(f"Image prediction - Raw output: {prediction}, Probabilities: [benign: {probability_benign:.3f}, malignant: {probability_malignant:.3f}], Result: {diagnosis}")

        return {
            "diagnosis": diagnosis,
            "is_malignant": diagnosis.lower() == "malignant",
            "confidence": f"{confidence_percentage:.2f}%",
            "probability_malignant": f"{probability_malignant * 100:.2f}%",
            "probability_benign": f"{probability_benign * 100:.2f}%",
            "result_code": int(result_code),
            "raw_prediction": float(probability_malignant)
        }
    except Exception as e:
        logger.error(f"[ERROR] Image prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

@app.get("/health-tips/")
def get_health_tips():
    tips = health_data()
    if not tips:
        raise HTTPException(status_code=404, detail="Health tips not available")
    return {"health_tips": tips}

@app.get("/status/")
def model_status():
    return {
        "text_model": "Loaded" if text_model else "Not loaded",
        "limit_model": "Loaded" if limit_model else "Not loaded",
        "image_model": "Loaded" if image_model else "Not loaded",
        "responses": "Loaded" if responses else "Not loaded"
    }

@app.post("/debug/model-info/")
async def debug_model_info():
    """Debug endpoint to inspect model architecture"""
    if not text_model:
        raise HTTPException(status_code=503, detail="Breast cancer model not loaded.")
    
    try:
        model_info = {
            "model_layers": [],
            "model_summary": [],
            "output_shape": None,
            "last_layer_activation": None
        }
        
        # Get layer information
        for i, layer in enumerate(text_model.layers):
            layer_info = {
                "index": i,
                "name": layer.name,
                "type": type(layer).__name__,
                "output_shape": str(layer.output_shape) if hasattr(layer, 'output_shape') else "Unknown"
            }
            
            # Try to get activation function for the last layer
            if hasattr(layer, 'activation'):
                layer_info["activation"] = str(layer.activation.__name__) if hasattr(layer.activation, '__name__') else str(layer.activation)
            
            model_info["model_layers"].append(layer_info)
        
        # Get output shape
        if hasattr(text_model, 'output_shape'):
            model_info["output_shape"] = str(text_model.output_shape)
        
        # Get last layer activation
        if len(text_model.layers) > 0:
            last_layer = text_model.layers[-1]
            if hasattr(last_layer, 'activation'):
                model_info["last_layer_activation"] = str(last_layer.activation.__name__) if hasattr(last_layer.activation, '__name__') else str(last_layer.activation)
        
        return model_info
    
    except Exception as e:
        logger.error(f"[ERROR] Debug model info failed: {e}")
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

@app.post("/debug/scaler-info/")
async def debug_scaler_info():
    """Debug endpoint to inspect scaler parameters"""
    if not scalar:
        raise HTTPException(status_code=503, detail="Scaler not loaded.")
    
    try:
        scaler_info = {
            "scaler_type": type(scalar).__name__,
            "n_features": getattr(scalar, 'n_features_in_', 'Unknown'),
            "feature_names": feature_names if feature_names else "Not available"
        }
        
        # Try to get scaler parameters
        if hasattr(scalar, 'mean_'):
            scaler_info["mean"] = scalar.mean_.tolist() if hasattr(scalar.mean_, 'tolist') else str(scalar.mean_)
        if hasattr(scalar, 'scale_'):
            scaler_info["scale"] = scalar.scale_.tolist() if hasattr(scalar.scale_, 'tolist') else str(scalar.scale_)
        if hasattr(scalar, 'var_'):
            scaler_info["variance"] = scalar.var_.tolist() if hasattr(scalar.var_, 'tolist') else str(scalar.var_)
        if hasattr(scalar, 'min_'):
            scaler_info["min"] = scalar.min_.tolist() if hasattr(scalar.min_, 'tolist') else str(scalar.min_)
        if hasattr(scalar, 'max_'):
            scaler_info["max"] = scalar.max_.tolist() if hasattr(scalar.max_, 'tolist') else str(scalar.max_)
        if hasattr(scalar, 'data_min_'):
            scaler_info["data_min"] = scalar.data_min_.tolist() if hasattr(scalar.data_min_, 'tolist') else str(scalar.data_min_)
        if hasattr(scalar, 'data_max_'):
            scaler_info["data_max"] = scalar.data_max_.tolist() if hasattr(scalar.data_max_, 'tolist') else str(scalar.data_max_)
        if hasattr(scalar, 'data_range_'):
            scaler_info["data_range"] = scalar.data_range_.tolist() if hasattr(scalar.data_range_, 'tolist') else str(scalar.data_range_)
            
        return scaler_info
    
    except Exception as e:
        logger.error(f"[ERROR] Debug scaler info failed: {e}")
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

@app.post("/predict/breast-cancer-no-scaling/")
async def predict_breast_cancer_no_scaling(input_data: BreastCancerInput):
    """Alternative endpoint that tries prediction without scaling to test if scaling is the issue"""
    try:
        if not text_model:
            raise HTTPException(status_code=503, detail="Breast cancer model not loaded.")
        if len(input_data.symptoms) != 30:
            raise HTTPException(status_code=400, detail="Exactly 30 symptom features are required.")

        # Prepare the data WITHOUT scaling
        data = np.array(input_data.symptoms, dtype=np.float32).reshape(1, -1)
        
        logger.info(f"Testing without scaling - Input data range: min={data.min():.4f}, max={data.max():.4f}")
        
        # Get prediction without scaling
        prediction = text_model.predict(data, verbose=0)
        raw_prob = float(prediction[0][0])
        
        logger.info(f"Prediction without scaling: {raw_prob}")
        
        # Handle the prediction
        confidence_malignant = raw_prob
        confidence_benign = 1 - raw_prob
        
        result = "Malignant" if confidence_malignant > 0.5 else "Benign"
        confidence_percentage = max(confidence_malignant, confidence_benign) * 100
        
        return {
            "model": "Breast Cancer Model (No Scaling)",
            "diagnosis": result,
            "is_malignant": result.lower() == "malignant",
            "confidence": f"{confidence_percentage:.2f}%",
            "probability_malignant": f"{confidence_malignant * 100:.2f}%",
            "probability_benign": f"{confidence_benign * 100:.2f}%",
            "raw_prediction": float(raw_prob),
            "note": "This prediction was made WITHOUT data scaling"
        }
    except Exception as e:
        logger.error(f"[ERROR] Breast cancer prediction (no scaling) failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def home():
    return {"message": "Welcome to Doctor AI!", "status": "active"}