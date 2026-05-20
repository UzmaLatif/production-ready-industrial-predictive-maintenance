import sys
from src.logger import get_logger
from src.exception import PredictiveMaintenanceException
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from src.model_trainer import ModelTrainer
from src.model_predictor import ModelPredictor

# ── Initialize logger ──────────────────────────────────────────────────────────
logger = get_logger("main")


def run_training_pipeline():
    """
    Runs the complete training pipeline:
    1. Load data
    2. Preprocess data
    3. Train and evaluate models
    4. Save best model
    """
    try:
        logger.info("=" * 60)
        logger.info("PREDICTIVE MAINTENANCE PIPELINE STARTED")
        logger.info("=" * 60)

        # ── Step 1: Data Ingestion ─────────────────────────────────────────
        logger.info("STEP 1: Data Ingestion")
        ingestion = DataIngestion()
        df = ingestion.load_data()

        # ── Step 2: Data Preprocessing ────────────────────────────────────
        logger.info("STEP 2: Data Preprocessing")
        preprocessor = DataPreprocessing()
        X_train, X_test, y_train, y_test = preprocessor.preprocess(df)

        # ── Step 3: Model Training ────────────────────────────────────────
        logger.info("STEP 3: Model Training")
        trainer = ModelTrainer()
        best_model = trainer.train(X_train, X_test, y_train, y_test)

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETE ✅")
        logger.info("Model saved to: artifacts/model.pkl")
        logger.info("Scaler saved to: artifacts/scaler.pkl")
        logger.info("=" * 60)

        return best_model

    except Exception as e:
        raise PredictiveMaintenanceException(str(e), sys)


def run_prediction_pipeline(input_data: dict) -> dict:
    """
    Runs prediction on new sensor readings.
    Can be called independently after training is complete.
    
    Args:
        input_data: Dictionary of sensor readings
        
    Returns:
        Prediction result dictionary
    """
    try:
        logger.info("PREDICTION PIPELINE STARTED")

        predictor = ModelPredictor()
        result = predictor.predict(input_data)

        logger.info("PREDICTION PIPELINE COMPLETE ✅")
        return result

    except Exception as e:
        raise PredictiveMaintenanceException(str(e), sys)


if __name__ == "__main__":

    # ── Run training pipeline ──────────────────────────────────────────────
    run_training_pipeline()

    # ── Test prediction on sample sensor reading ───────────────────────────
    sample_input = {
        "Type": 1,
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
        "TWF": 0,
        "HDF": 0,
        "PWF": 0,
        "OSF": 0,
        "RNF": 0
    }

    logger.info("Testing prediction on sample input...")
    result = run_prediction_pipeline(sample_input)

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    for key, value in result.items():
        print(f"{key:15s}: {value}")
    print("=" * 60)