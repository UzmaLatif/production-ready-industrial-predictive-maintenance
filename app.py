import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from src.logger import get_logger
from src.exception import PredictiveMaintenanceException
from src.model_predictor import ModelPredictor

# ── Initialize app and logger ──────────────────────────────────────────────────
app = FastAPI(
    title="Industrial Predictive Maintenance API",
    description="Production ML API for predicting industrial machine failures — modelled on Ferrari S.p.A production work",
    version="1.0.0"
)

logger = get_logger("app")

# ── Load predictor once at startup ─────────────────────────────────────────────
predictor = ModelPredictor()


# ── Input schema ───────────────────────────────────────────────────────────────
class SensorReading(BaseModel):
    Type: int
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
    TWF: int = 0
    HDF: int = 0
    PWF: int = 0
    OSF: int = 0
    RNF: int = 0

    class Config:
        json_schema_extra = {
            "example": {
                "Type": 1,
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1551,
                "torque": 42.8,
                "tool_wear": 0,
                "TWF": 0,
                "HDF": 0,
                "PWF": 0,
                "OSF": 0,
                "RNF": 0
            }
        }


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "Industrial Predictive Maintenance API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Random Forest",
        "f1_score": "98.51%",
        "precision": "100%"
    }


@app.post("/predict")
def predict(reading: SensorReading):
    """
    Predict machine failure from sensor readings.
    Returns prediction, confidence and recommended action.
    """
    try:
        logger.info("Prediction request received")

        # Convert input to format predictor expects
        input_data = {
            "Type": reading.Type,
            "Air temperature [K]": reading.air_temperature,
            "Process temperature [K]": reading.process_temperature,
            "Rotational speed [rpm]": reading.rotational_speed,
            "Torque [Nm]": reading.torque,
            "Tool wear [min]": reading.tool_wear,
            "TWF": reading.TWF,
            "HDF": reading.HDF,
            "PWF": reading.PWF,
            "OSF": reading.OSF,
            "RNF": reading.RNF
        }

        result = predictor.predict(input_data)
        logger.info(f"Prediction served: {result['status']}")
        return result

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Run server ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )