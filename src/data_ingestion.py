import os
import sys
import pandas as pd
import yaml
from src.logger import get_logger
from src.exception import PredictiveMaintenanceException

# ── Initialize logger ──────────────────────────────────────────────────────────
logger = get_logger("data_ingestion")


class DataIngestion:
    """
    Responsible for loading and validating raw data.
    Reads config from config.yaml — nothing hardcoded.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        logger.info("DataIngestion initialized")
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> dict:
        """Load settings from config.yaml"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Config loaded from: {config_path}")
            return config
        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def load_data(self) -> pd.DataFrame:
        """
        Load raw CSV data from path defined in config.yaml
        
        Returns:
            Raw DataFrame
        """
        try:
            data_path = self.config["data"]["raw_data_path"]
            logger.info(f"Loading data from: {data_path}")

            # ── Load CSV ───────────────────────────────────────────────────
            df = pd.read_csv(data_path)
            logger.info(f"Data loaded successfully — shape: {df.shape}")

            # ── Validate data ──────────────────────────────────────────────
            self._validate_data(df)

            return df

        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Basic validation checks on raw data.
        Logs warnings if issues found.
        """
        try:
            # Check for empty dataframe
            if df.empty:
                raise ValueError("Dataset is empty")

            # Check for missing values
            missing = df.isnull().sum()
            if missing.any():
                logger.warning(f"Missing values found:\n{missing[missing > 0]}")
            else:
                logger.info("No missing values found ✅")

            # Check target column exists
            target = self.config["features"]["target_column"]
            if target not in df.columns:
                raise ValueError(f"Target column '{target}' not found in dataset")
            else:
                logger.info(f"Target column '{target}' found ✅")

            # Log class distribution
            target_counts = df[target].value_counts()
            logger.info(f"Class distribution:\n{target_counts}")

            logger.info("Data validation complete ✅")

        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)