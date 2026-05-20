import os
import sys
import pickle
import numpy as np
import pandas as pd
import yaml
from src.logger import get_logger
from src.exception import PredictiveMaintenanceException

logger = get_logger("model_predictor")


class ModelPredictor:
    def __init__(self, config_path: str = "config/config.yaml"):
        logger.info("ModelPredictor initialized")
        self.config = self._load_config(config_path)
        self.model = None
        self.scaler = None
        self._load_artifacts()

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def _load_artifacts(self) -> None:
        try:
            model_path = self.config["artifacts"]["model_path"]
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from: {model_path} ✅")

            scaler_path = self.config["artifacts"]["scaler_path"]
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            logger.info(f"Scaler loaded from: {scaler_path} ✅")
        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def predict(self, input_data: dict) -> dict:
        try:
            logger.info("Making prediction on new sensor readings...")
            df = pd.DataFrame([input_data])
            df = self._feature_engineering(df)
            df_scaled = self.scaler.transform(df)
            prediction = self.model.predict(df_scaled)[0]

            confidence = None
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(df_scaled)[0]
                confidence = round(float(max(proba)) * 100, 2)

            result = {
                "prediction": int(prediction),
                "status": "⚠️ MACHINE FAILURE DETECTED" if prediction == 1 else "✅ NORMAL OPERATION",
                "confidence": f"{confidence}%" if confidence else "N/A",
                "action": "Immediate maintenance required" if prediction == 1 else "Continue monitoring"
            }

            logger.info(f"Prediction: {result['status']}")
            return result
        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df["temp_difference"] = (
                df["Process temperature [K]"] - df["Air temperature [K]"]
            )
            df["power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]
            df["wear_rate"] = df["Tool wear [min]"] * df["Torque [Nm]"]
            return df
        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info(f"Batch prediction on {len(df)} samples...")
            df = self._feature_engineering(df)
            df_scaled = self.scaler.transform(df)
            predictions = self.model.predict(df_scaled)
            df["prediction"] = predictions
            df["status"] = df["prediction"].map({
                0: "✅ Normal",
                1: "⚠️ Failure"
            })
            failures = (predictions == 1).sum()
            logger.info(f"Batch complete — {failures} failures detected ✅")
            return df
        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)