"""
Isolation Forest Model for Anomaly Detection
Unsupervised learning approach for fraud detection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib
import mlflow
import mlflow.sklearn
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IsolationForestModel:
    """Isolation Forest model for fraud detection."""
    
    def __init__(self, contamination: float = 0.01, random_state: int = 42):
        """
        Initialize Isolation Forest model.
        
        Args:
            contamination: Expected proportion of outliers (fraud rate)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.feature_names = []
        
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame = None) -> Dict[str, float]:
        """
        Train the Isolation Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels (optional, for evaluation)
            
        Returns:
            Dictionary of training metrics
        """
        logger.info("Training Isolation Forest model...")
        
        self.feature_names = X_train.columns.tolist()
        
        # Initialize model
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
        
        # Train model
        self.model.fit(X_train)
        
        # Predictions (isolation forest returns -1 for outliers, 1 for inliers)
        train_pred = self.model.predict(X_train)
        train_pred_binary = (train_pred == -1).astype(int)
        
        # Calculate metrics if labels are provided
        metrics = {}
        if y_train is not None:
            metrics = self._calculate_metrics(y_train, train_pred_binary)
            logger.info(f"Training metrics: {metrics}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict
            
        Returns:
            Binary predictions (1 = fraud, 0 = legitimate)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        pred = self.model.predict(X)
        return (pred == -1).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get anomaly scores.
        
        Args:
            X: Features to score
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        scores = self.model.score_samples(X)
        # Convert to probability-like scores (0 to 1)
        probas = (scores - scores.min()) / (scores.max() - scores.min())
        return 1 - probas  # Higher = more likely fraud
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Calculate AUC if we have probability scores
        try:
            y_proba = self.predict_proba(self.model._fit_X[:len(y_true)])
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        except:
            pass
            
        return metrics
    
    def save(self, filepath: str):
        """Save model to disk."""
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk."""
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def log_to_mlflow(self, metrics: Dict[str, float], params: Dict[str, any] = None):
        """Log model and metrics to MLflow."""
        try:
            if params is None:
                params = {
                    'contamination': self.contamination,
                    'n_estimators': self.model.n_estimators,
                    'max_samples': self.model.max_samples
                }
            
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(self.model, "isolation_forest_model")
            logger.info("Model logged to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
