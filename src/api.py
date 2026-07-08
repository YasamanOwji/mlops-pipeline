from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

root_dir = Path(__file__).resolve().parent.parent
config_path = root_dir / "config" / "mlops_config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = root_dir / config["deployment"]["production_dir"] / "model"

app = FastAPI(title="Churn Prediction API", version="1.0")

try:
    model = mlflow.sklearn.load_model(str(MODEL_PATH))
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as e:
    logger.error("Failed to load model: %s", e)
    model = None


class PredictionRequest(BaseModel):
    features: dict


class PredictionResponse(BaseModel):
    prediction: int
    probability: float


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        input_df = pd.DataFrame([request.features])
        if "student_id" in input_df.columns:
            input_df.drop(columns=["student_id"], inplace=True)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        return PredictionResponse(prediction=int(prediction), probability=float(probability))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(e))
