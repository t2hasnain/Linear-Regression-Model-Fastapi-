from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="Weight Prediction System")

# Browser security (CORS) allow karein
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = joblib.load('weight_model_pipeline.pkl')

class PredictionInput(BaseModel):
    Gender: str
    Height: float = Field(..., gt=0)

# Prediction API Endpoint
@app.post("/predict")
def predict(data: PredictionInput):
    try:
        input_df = pd.DataFrame([{
            "Gender": data.Gender,
            "Height": data.Height
        }])
        
        raw_prediction = pipeline.predict(input_df)[0]
        
        
        weight_kg = raw_prediction * 0.45359237 if raw_prediction > 100 else raw_prediction

        return {
            "predicted_weight_kg": float(round(weight_kg, 2)),
            "predicted_weight_raw": float(round(raw_prediction, 2))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

# Frontend UI Endpoint
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()