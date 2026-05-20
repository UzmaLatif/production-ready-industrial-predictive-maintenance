import os
import sys
import pickle
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)
from src.logger import get_logger
from src.exception import PredictiveMaintenanceException

# ── Initialize logger ──────────────────────────────────────────────────────────
logger = get_logger("model_trainer")


class ModelTrainer:
    """
    Trains Random Forest and Isolation Forest models,
    evaluates them, compares results, and saves the best model.
    Reads all settings from config.yaml.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        logger.info("ModelTrainer initialized")
        self.config = self._load_config(config_path)
        self.rf_model = None
        self.if_model = None
        self.best_model = None
        self.best_model_name = None

    def _load_config(self, config_path: str) -> dict:
        """Load settings from config.yaml"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Config loaded from: {config_path}")
            return config
        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def train(self, X_train, X_test, y_train, y_test):
        """
        Full training pipeline:
        1. Train Random Forest
        2. Train Isolation Forest
        3. Evaluate both
        4. Compare and save best model
        
        Args:
            X_train, X_test: Feature matrices
            y_train, y_test: Target vectors
            
        Returns:
            Best model object
        """
        try:
            logger.info("=" * 60)
            logger.info("Starting model training pipeline...")
            logger.info("=" * 60)

            # Step 1: Train Random Forest
            rf_metrics = self._train_random_forest(
                X_train, X_test, y_train, y_test
            )

            # Step 2: Train Isolation Forest
            if_metrics = self._train_isolation_forest(
                X_train, X_test, y_train, y_test
            )

            # Step 3: Compare and select best model
            self._select_best_model(rf_metrics, if_metrics)

            # Step 4: Save best model
            self._save_model()

            # Step 5: Plot results
            self._plot_confusion_matrix(X_test, y_test)
            self._plot_feature_importance(X_train)

            logger.info("=" * 60)
            logger.info(f"Training complete ✅ Best model: {self.best_model_name}")
            logger.info("=" * 60)

            return self.best_model

        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def _train_random_forest(self, X_train, X_test, y_train, y_test) -> dict:
        """Train and evaluate Random Forest Classifier"""
        try:
            logger.info("Training Random Forest Classifier...")

            # Get settings from config
            rf_config = self.config["model"]["random_forest"]

            # Train model
            self.rf_model = RandomForestClassifier(
                n_estimators=rf_config["n_estimators"],
                max_depth=rf_config["max_depth"],
                min_samples_split=rf_config["min_samples_split"],
                random_state=rf_config["random_state"],
                n_jobs=rf_config["n_jobs"]
            )
            self.rf_model.fit(X_train, y_train)
            logger.info("Random Forest training complete ✅")

            # Evaluate
            y_pred = self.rf_model.predict(X_test)
            metrics = self._evaluate_model(y_test, y_pred, "Random Forest")

            return metrics

        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def _train_isolation_forest(self, X_train, X_test, y_train, y_test) -> dict:
        """Train and evaluate Isolation Forest"""
        try:
            logger.info("Training Isolation Forest...")

            # Get settings from config
            if_config = self.config["model"]["isolation_forest"]

            # Train model
            self.if_model = IsolationForest(
                n_estimators=if_config["n_estimators"],
                contamination=if_config["contamination"],
                random_state=if_config["random_state"]
            )
            self.if_model.fit(X_train)
            logger.info("Isolation Forest training complete ✅")

            # Isolation Forest predicts -1 for anomaly, 1 for normal
            # Convert to 0 and 1 to match our target
            y_pred_raw = self.if_model.predict(X_test)
            y_pred = np.where(y_pred_raw == -1, 1, 0)

            metrics = self._evaluate_model(y_test, y_pred, "Isolation Forest")

            return metrics

        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def _evaluate_model(self, y_test, y_pred, model_name: str) -> dict:
        """
        Evaluate model and log all metrics.
        
        Returns:
            Dictionary of metrics
        """
        try:
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred)

            logger.info(f"\n{'='*40}")
            logger.info(f"{model_name} Results:")
            logger.info(f"  Precision : {precision:.4f}")
            logger.info(f"  Recall    : {recall:.4f}")
            logger.info(f"  F1 Score  : {f1:.4f}")
            logger.info(f"  ROC-AUC   : {roc_auc:.4f}")
            logger.info(f"\n{classification_report(y_test, y_pred)}")
            logger.info(f"{'='*40}")

            return {
                "model_name": model_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc
            }

        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def _select_best_model(self, rf_metrics: dict, if_metrics: dict) -> None:
        """
        Compare both models and select the best one based on F1 score.
        F1 is used because our dataset is imbalanced.
        """
        try:
            logger.info("Comparing models...")

            if rf_metrics["f1"] >= if_metrics["f1"]:
                self.best_model = self.rf_model
                self.best_model_name = "Random Forest"
                logger.info(f"✅ Random Forest selected — F1: {rf_metrics['f1']:.4f}")
            else:
                self.best_model = self.if_model
                self.best_model_name = "Isolation Forest"
                logger.info(f"✅ Isolation Forest selected — F1: {if_metrics['f1']:.4f}")

        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def _save_model(self) -> None:
        """Save best model to artifacts folder"""
        try:
            model_path = self.config["artifacts"]["model_path"]
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            with open(model_path, "wb") as f:
                pickle.dump(self.best_model, f)

            logger.info(f"Best model saved to: {model_path} ✅")

        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def _plot_confusion_matrix(self, X_test, y_test) -> None:
        """Plot and save confusion matrix"""
        try:
            if self.best_model_name == "Random Forest":
                y_pred = self.best_model.predict(X_test)
            else:
                y_pred_raw = self.best_model.predict(X_test)
                y_pred = np.where(y_pred_raw == -1, 1, 0)

            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Failure"],
                yticklabels=["Normal", "Failure"]
            )
            plt.title(
                f"Confusion Matrix — {self.best_model_name}",
                fontweight="bold"
            )
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            plt.tight_layout()
            plt.savefig("artifacts/confusion_matrix.png", dpi=150)
            plt.close()
            logger.info("Confusion matrix saved ✅")

        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)

    def _plot_feature_importance(self, X_train) -> None:
        """Plot feature importance (Random Forest only)"""
        try:
            if self.best_model_name != "Random Forest":
                logger.info(
                    "Feature importance only available for Random Forest"
                )
                return

            # Get feature names
            feature_names = [
                "Type",
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
                "TWF", "HDF", "PWF", "OSF", "RNF",
                "temp_difference",
                "power",
                "wear_rate"
            ]

            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(12, 6))
            plt.bar(
                range(len(importances)),
                importances[indices],
                color="#2196F3",
                alpha=0.85
            )
            plt.xticks(
                range(len(importances)),
                [feature_names[i] for i in indices],
                rotation=45,
                ha="right"
            )
            plt.title(
                "Feature Importance — Random Forest",
                fontweight="bold",
                fontsize=14
            )
            plt.ylabel("Importance Score")
            plt.tight_layout()
            plt.savefig("artifacts/feature_importance.png", dpi=150)
            plt.close()
            logger.info("Feature importance plot saved ✅")

        except Exception as e:
            raise PredictiveMaintenanceException(str(e), sys)