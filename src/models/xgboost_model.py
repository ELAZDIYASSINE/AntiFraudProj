"""
XGBoost Model for Fraud Detection
Supervised gradient boosting approach.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import mlflow
import mlflow.xgboost
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostModel:
    """XGBoost model for fraud detection."""
    
    def __init__(self, scale_pos_weight: float = 10, random_state: int = 42):
        """
        Initialize XGBoost model.
        
        Args:
            scale_pos_weight: Weight for positive class (fraud)
            random_state: Random seed for reproducibility
        """
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.model = None
        self.feature_names = []
        
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
              X_val: pd.DataFrame = None, y_val: pd.DataFrame = None) -> Dict[str, float]:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary of training metrics
        """
        logger.info("Training XGBoost model...")
        
        self.feature_names = X_train.columns.tolist()
        
        # Calculate scale_pos_weight if not provided
        if self.scale_pos_weight is None:
            self.scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        
        # Initialize model
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='auc',
            early_stopping_rounds=20
        )
        
        # Train with validation if provided
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            val_pred = self.model.predict(X_val)
            metrics = self._calculate_metrics(y_val, val_pred, X_val)
        else:
            self.model.fit(X_train, y_train)
            train_pred = self.model.predict(X_train)
            metrics = self._calculate_metrics(y_train, train_pred, X_train)
        
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
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features to score
            
        Returns:
            Probability scores for fraud class
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)[:, 1]
    
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
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save(self, filepath: str):
        """Save model to disk."""
        self.model.save_model(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk."""
        self.model = xgb.XGBClassifier()
        self.model.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def log_to_mlflow(self, metrics: Dict[str, float], params: Dict[str, any] = None):
        """Log model and metrics to MLflow."""
        try:
            if params is None:
                params = {
                    'n_estimators': self.model.n_estimators,
                    'max_depth': self.model.max_depth,
                    'learning_rate': self.model.learning_rate,
                    'scale_pos_weight': self.scale_pos_weight
                }
            
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(self.model, "xgboost_model")
            logger.info("Model logged to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
