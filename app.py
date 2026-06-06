"""
app.py - FastAPI Backend with ONNX Runtime Inference
=====================================================
Heart Disease Risk Prediction API using an XGBoost model exported to ONNX.
"""

import json
import pickle
import os

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import onnxruntime as ort

# ---------------------------------------------------------------------------
# 1. APPLICATION SETUP
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Heart Disease Risk Prediction",
    description="Clinical-grade ML inference via ONNX Runtime",
    version="4.0.0",
)

templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------------------------------
# 2. LOAD MODEL ARTIFACTS
# ---------------------------------------------------------------------------
MODEL_DIR = "models"

# ONNX Inference Session (loaded once at startup)
onnx_session = ort.InferenceSession(
    os.path.join(MODEL_DIR, "heart_disease_model.onnx"),
    providers=["CPUExecutionProvider"],
)
INPUT_NAME = onnx_session.get_inputs()[0].name

# Scaler for numeric features
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# Label encoders for categorical features
with open(os.path.join(MODEL_DIR, "encoders.pkl"), "rb") as f:
    encoders = pickle.load(f)

# Feature configuration
with open(os.path.join(MODEL_DIR, "feature_config.json"), "r") as f:
    feature_config = json.load(f)

FEATURE_ORDER = feature_config["feature_order"]
CATEGORICAL_COLS = feature_config["categorical_cols"]
NUMERIC_COLS = feature_config["numeric_cols"]
ENCODING_MAPS = feature_config["encoding_maps"]

print(f"[OK] ONNX session loaded  | Features: {len(FEATURE_ORDER)}")
print(f"[OK] Input tensor name: {INPUT_NAME}")

# ---------------------------------------------------------------------------
# 3. PYDANTIC REQUEST SCHEMA
# ---------------------------------------------------------------------------
class PatientInput(BaseModel):
    """Schema for the 13 clinical attributes sent from the frontend."""
    age: float = Field(..., ge=1, le=120, description="Patient age in years")
    sex: str = Field(..., description="Male or Female")
    chest_pain_type: str = Field(..., description="Chest pain category")
    resting_blood_pressure: float = Field(..., ge=50, le=300, description="Resting BP in mm Hg")
    cholestoral: float = Field(..., ge=50, le=600, description="Serum cholesterol in mg/dl")
    fasting_blood_sugar: str = Field(..., description="Fasting blood sugar level category")
    rest_ecg: str = Field(..., description="Resting ECG result")
    Max_heart_rate: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exercise_induced_angina: str = Field(..., description="Exercise induced angina (Yes/No)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: str = Field(..., description="Slope of peak exercise ST segment")
    vessels_colored_by_flourosopy: str = Field(..., description="Number of major vessels (0-4)")
    thalassemia: str = Field(..., description="Thalassemia type")

# ---------------------------------------------------------------------------
# 4. HEALTH ADVICE LOGIC
# ---------------------------------------------------------------------------
def get_health_advice(prediction: int, probability: float) -> dict:
    """Return structured advice based on the prediction."""
    if prediction == 1:
        if probability >= 0.8:
            risk_level = "Critical"
            badge_class = "critical"
        elif probability >= 0.6:
            risk_level = "High"
            badge_class = "high"
        else:
            risk_level = "Moderate"
            badge_class = "moderate"
        return {
            "risk_level": risk_level,
            "badge_class": badge_class,
            "title": "Immediate Actions Required",
            "items": [
                "Consult a cardiologist as soon as possible for a comprehensive evaluation.",
                "Follow any prescribed medication strictly and do not skip doses.",
                "Adopt a heart-healthy diet: reduce sodium, saturated fats, and processed sugars.",
                "Avoid strenuous physical activity until cleared by a medical professional.",
                "Monitor for symptoms like chest discomfort, unusual fatigue, or shortness of breath.",
            ],
        }
    else:
        risk_level = "Low"
        badge_class = "low"
        return {
            "risk_level": risk_level,
            "badge_class": badge_class,
            "title": "Maintenance & Prevention Tips",
            "items": [
                "Maintain a regular exercise routine (at least 150 minutes of moderate activity per week).",
                "Keep a balanced diet rich in whole grains, lean proteins, and plenty of vegetables.",
                "Schedule annual check-ups to monitor blood pressure and cholesterol levels.",
                "Practice stress-management techniques like meditation or deep breathing exercises.",
                "Stay hydrated and ensure you get 7-9 hours of quality sleep daily.",
            ],
        }

# ---------------------------------------------------------------------------
# 5. ROUTES
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(patient: PatientInput):
    """
    Accept patient clinical metrics, preprocess them, run ONNX inference,
    and return the risk prediction with probability scores.
    """
    try:
        # Build the raw feature dict in the correct order
        raw = patient.model_dump()

        # --- Encode categoricals ---
        encoded = {}
        for col in FEATURE_ORDER:
            if col in CATEGORICAL_COLS:
                label = raw[col]
                # Use the encoding map from feature_config
                if label in ENCODING_MAPS[col]:
                    encoded[col] = float(ENCODING_MAPS[col][label])
                else:
                    # Fallback: try label encoder directly
                    encoded[col] = float(encoders[col].transform([label])[0])
            else:
                encoded[col] = float(raw[col])

        # --- Scale numerics ---
        numeric_values = np.array([[encoded[c] for c in NUMERIC_COLS]])
        scaled_values = scaler.transform(numeric_values)[0]
        for i, col in enumerate(NUMERIC_COLS):
            encoded[col] = float(scaled_values[i])

        # --- Assemble feature vector in correct order ---
        feature_vector = np.array(
            [[encoded[col] for col in FEATURE_ORDER]], dtype=np.float32
        )

        # --- ONNX Inference ---
        onnx_result = onnx_session.run(None, {INPUT_NAME: feature_vector})
        prediction = int(onnx_result[0][0])

        # Probability scores from ONNX (ZipMap output)
        prob_output = onnx_result[1]
        if isinstance(prob_output, list):
            # ZipMap returns list of dicts
            prob_dict = prob_output[0]
            prob_no_risk = float(prob_dict.get(0, prob_dict.get("0", 0.0)))
            prob_risk = float(prob_dict.get(1, prob_dict.get("1", 0.0)))
        elif isinstance(prob_output, np.ndarray):
            prob_no_risk = float(prob_output[0][0])
            prob_risk = float(prob_output[0][1])
        else:
            prob_no_risk = 0.0
            prob_risk = 0.0

        # Build response
        advice = get_health_advice(prediction, prob_risk)

        return {
            "status": "success",
            "prediction": prediction,
            "risk_level": advice["risk_level"],
            "badge_class": advice["badge_class"],
            "probability": {
                "no_risk": round(prob_no_risk * 100, 1),
                "risk": round(prob_risk * 100, 1),
            },
            "advice": {
                "title": advice["title"],
                "items": advice["items"],
            },
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring."""
    return {
        "status": "healthy",
        "model": "heart_disease_xgboost_onnx",
        "features": len(FEATURE_ORDER),
        "encoding_maps": ENCODING_MAPS,
    }


# ---------------------------------------------------------------------------
# 6. ENTRYPOINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)