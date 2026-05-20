import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import yaml
from src.logger import get_logger
from src.exception import PredictiveMaintenanceException

# ── Initialize logger ──────────────────────────────────────────────────────────
logger = get_logger("data_preprocessing")


class DataPreprocessing:
    """
    Responsible for cleaning, feature engineering,
    encoding, scaling, and splitting data.
    Reads all settings from config.yaml.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        logger.info("DataPreprocessing initialized")
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

    def _load_config(self, config_path: str) -> dict:
        """Load settings from config.yaml"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Config loaded from: {config_path}")
            return config
        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def preprocess(self, df: pd.DataFrame):
        """
        Full preprocessing pipeline.
        
        Args:
            df: Raw DataFrame from DataIngestion
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        try:
            logger.info("Starting preprocessing pipeline...")

            # Step 1: Drop unnecessary columns
            df = self._drop_columns(df)

            # Step 2: Engineer new features
            df = self._feature_engineering(df)

            # Step 3: Encode categorical columns
            df = self._encode_categoricals(df)

            # Step 4: Split features and target
            X, y = self._split_features_target(df)

            # Step 5: Scale numerical features
            X = self._scale_features(X)

            # Step 6: Train test split
            X_train, X_test, y_train, y_test = self._split_data(X, y)

            logger.info("Preprocessing pipeline complete ✅")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that are not useful for modelling"""
        try:
            drop_cols = self.config["features"]["drop_columns"]
            df = df.drop(columns=drop_cols, errors="ignore")
            logger.info(f"Dropped columns: {drop_cols}")
            return df
        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones.
        Domain knowledge from industrial sensor data.
        """
        try:
            # Temperature difference — heat dissipation indicator
            df["temp_difference"] = (
                df["Process temperature [K]"] - df["Air temperature [K]"]
            )

            # Power = Torque × Rotational speed (physics formula)
            df["power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]

            # Tool wear rate — how fast is the tool degrading?
            df["wear_rate"] = df["Tool wear [min]"] * df["Torque [Nm]"]

            logger.info("Feature engineering complete ✅")
            logger.info("New features created: temp_difference, power, wear_rate")
            return df

        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns using LabelEncoder"""
        try:
            cat_cols = self.config["features"]["categorical_columns"]
            for col in cat_cols:
                df[col] = self.encoder.fit_transform(df[col])
                logger.info(f"Encoded column: {col}")
            return df
        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def _split_features_target(self, df: pd.DataFrame):
        """Separate features (X) from target (y)"""
        try:
            target = self.config["features"]["target_column"]
            X = df.drop(columns=[target])
            y = df[target]
            logger.info(f"Features shape: {X.shape}")
            logger.info(f"Target shape: {y.shape}")
            logger.info(f"Failure rate: {y.mean()*100:.2f}%")
            return X, y
        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def _scale_features(self, X: pd.DataFrame) -> np.ndarray:
        """Scale numerical features using StandardScaler"""
        try:
            X_scaled = self.scaler.fit_transform(X)
            logger.info("Feature scaling complete ✅")

            # Save scaler for later use in predictions
            scaler_path = self.config["artifacts"]["scaler_path"]
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Scaler saved to: {scaler_path}")
            return X_scaled
        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def _split_data(self, X, y):
        """Split data into train and test sets"""
        try:
            test_size = self.config["data"]["test_size"]
            random_state = self.config["data"]["random_state"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y  # maintains class distribution in both splits
            )
            logger.info(f"Train size: {X_train.shape[0]} samples")
            logger.info(f"Test size: {X_test.shape[0]} samples")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)