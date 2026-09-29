from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "champion_credit_default_model.joblib"

app = FastAPI(
    title="Credit Default Risk ML API",
    description="Production-style REST API wrapping the Task 04 champion classification pipeline.",
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)

class PredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    credit_score: float = Field(..., ge=300, le=900)
    employment_years: float = Field(..., ge=0, le=80)
    debt_to_income: float = Field(..., ge=0, le=2)
    late_payments: int = Field(..., ge=0, le=100)
    home_ownership: str
    loan_purpose: str
    employment_type: str

class PredictionResponse(BaseModel):
    prediction: int
    probability_default: float
    probability_non_default: float
    model_version: str

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        payload = request.model_dump()
        features = pd.DataFrame([payload])
        probability_default = float(model.predict_proba(features)[0][1])
        prediction = int(probability_default >= 0.5)

        return PredictionResponse(
            prediction=prediction,
            probability_default=round(probability_default, 6),
            probability_non_default=round(1 - probability_default, 6),
            model_version="task04-champion-v1",
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}")
