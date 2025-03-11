from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import joblib as jb
import json
import io
from PIL import Image
from fuzzywuzzy import process
from typing import List, Optional
import logging

app = FastAPI()

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DoctorAI")

# Global variables
text_model, scalar, image_model, responses = None, None, None, None

# Load Models
def load_med_data_model():
    global text_model, scalar
    try:
        text_model = tf.keras.models.load_model('Model/Models/breast_cancer/breast_cancer_model.keras')
        scalar = jb.load('Model/Models/breast_cancer/scaler_breast_cancer_model.pkl')
        logger.info("[INFO] Breast cancer text model loaded successfully.")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load breast cancer model: {e}")

def load_image_model():
    global image_model
    try:
        image_model = tf.keras.models.load_model('Model/Image_recognize/Model/breast_cancer_image_model_efficientnet.keras')
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
    load_image_model()
    load_med_data_model()
    load_response()
    logger.info("\n✅ Doctor AI is ready!")

initialize()

# Symptom Data Validation & Interactive Feature Collection
class Symptoms(BaseModel):
    symptoms: List[float]

@app.post("/predict/med_data/")
async def predict_med_data(input_data: Symptoms):
    if not text_model or not scalar:
        raise HTTPException(status_code=500, detail="Text model not loaded.")
    
    try:
        if len(input_data.symptoms) < 30:
            return {"error": "Incomplete data. Please provide all 30 required symptoms."}
        
        data = np.array(input_data.symptoms).reshape(1, -1)
        data = scalar.transform(data)
        prediction = text_model.predict(data)[0][0]
        result = "Malignant" if prediction > 0.5 else "Benign"
        return {"diagnosis": result, "confidence": f"{float(prediction) * 100:.2f}%"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Image Prediction Endpoint
@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    if not image_model:
        raise HTTPException(status_code=500, detail="Image model not loaded.")
    
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).resize((640, 640))
        image = np.array(image) / 255.0
        image = image.reshape(1, 640, 640, 3)
        
        prediction = image_model.predict(image)
        result = np.argmax(prediction)
        return {"Diagnosis": int(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chatbot API
class ChatInput(BaseModel):
    message: str

@app.post("/chat/")
def chat_bot(input_data: ChatInput):
    if not responses:
        return {"Response": "Sorry, chatbot responses are not loaded."}
    
    best_match, score = process.extractOne(input_data.message, [entry["user_input"] for entry in responses])
    
    if score > 60:
        for entry in responses:
            if entry["user_input"] == best_match:
                return {"Response": entry["general_ai_response"]}
    
    return {"Response": "I'm here to help! Could you describe your symptoms more clearly?"}

# Health Tips API
class HealthTips(BaseModel):
    title: str
    description: str
    recommendation: str
    actionable: bool
    keywords: List[str]
    priority: str

healthTips_data = health_data()

@app.get("/health-tips", response_model=List[HealthTips])
async def health_tips(
    category: Optional[str] = Query(None, description="Filter by category"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    keyword: Optional[str] = Query(None, description="Filter by keyword")
):
    if not healthTips_data:
        return []
    
    filtered = healthTips_data
    if category:
        filtered = [tip for tip in filtered if tip.get("category") == category]
    if priority:
        filtered = [tip for tip in filtered if tip.get("priority") == priority]
    if keyword:
        filtered = [tip for tip in filtered if keyword in tip.get("keywords", [])]
    
    return filtered

# Home Route
@app.get("/")
def home():
    return {"Message": "Welcome to Doctor AI!"}

# Warm-Up Model Request (To prevent cold starts)
@app.get("/warmup")
def warmup():
    try:
        dummy_input = np.zeros((1, 30))
        _ = text_model.predict(dummy_input)
        return {"Message": "Model warmed up!"}
    except Exception as e:
        return {"Error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
