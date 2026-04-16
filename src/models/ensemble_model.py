"""
Ensemble Model combining Isolation Forest and XGBoost
Improves prediction accuracy by combining multiple models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict
import logging

from .isolation_forest_model import IsolationForestModel
from .xgboost_model import XGBoostModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleModel:
    """Ensemble model combining Isolation Forest and XGBoost."""
    
    def __init__(self, iso_weight: float = 0.3, xgb_weight: float = 0.7):
        """
        Initialize ensemble model.
        
        Args:
            iso_weight: Weight for Isolation Forest predictions
            xgb_weight: Weight for XGBoost predictions
        """
        self.iso_weight = iso_weight
        self.xgb_weight = xgb_weight
        self.iso_model = IsolationForestModel()
        self.xgb_model = XGBoostModel()
        self.threshold = 0.5  # Classification threshold
        
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
              X_val: pd.DataFrame = None, y_val: pd.DataFrame = None) -> Dict[str, float]:
        """
        Train both models in the ensemble.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary of ensemble metrics
        """
        logger.info("Training ensemble model...")
        
        # Train Isolation Forest
        iso_metrics = self.iso_model.train(X_train, y_train)
        
        # Train XGBoost
        xgb_metrics = self.xgb_model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate ensemble on validation set
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            metrics = self._calculate_metrics(y_val, val_pred, X_val)
        else:
            train_pred = self.predict(X_train)
            metrics = self._calculate_metrics(y_train, train_pred, X_train)
        
        logger.info(f"Ensemble metrics: {metrics}")
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Binary predictions (1 = fraud, 0 = legitimate)
        """
        # Get probability scores from both models
        iso_proba = self.iso_model.predict_proba(X)
        xgb_proba = self.xgb_model.predict_proba(X)
        
        # Weighted average
        ensemble_proba = (self.iso_weight * iso_proba + 
                         self.xgb_weight * xgb_proba)
        
        # Convert to binary predictions
        return (ensemble_proba >= self.threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get ensemble probability scores.
        
        Args:
            X: Features to score
            
        Returns:
            Probability scores for fraud class
        """
        iso_proba = self.iso_model.predict_proba(X)
        xgb_proba = self.xgb_model.predict_proba(X)
        
        return (self.iso_weight * iso_proba + 
                self.xgb_weight * xgb_proba)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          X: pd.DataFrame = None) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Calculate AUC
        if X is not None:
            try:
                y_proba = self.predict_proba(X)
                metrics['auc'] = roc_auc_score(y_true, y_proba)
            except:
                pass
        
        return metrics
    
    def save(self, iso_path: str, xgb_path: str):
        """Save both models to disk."""
        self.iso_model.save(iso_path)
        self.xgb_model.save(xgb_path)
        logger.info("Ensemble models saved")
    
    def load(self, iso_path: str, xgb_path: str):
        """Load both models from disk."""
        self.iso_model.load(iso_path)
        self.xgb_model.load(xgb_path)
        logger.info("Ensemble models loaded")
