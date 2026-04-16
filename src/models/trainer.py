"""
Model Training Pipeline with MLflow Tracking
Automated training with hyperparameter tuning.
"""

import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from contextlib import nullcontext
import logging
import xgboost as xgb

from .isolation_forest_model import IsolationForestModel
from .xgboost_model import XGBoostModel
from .ensemble_model import EnsembleModel
from ..preprocessing.preprocessor import DataPreprocessor, FeatureEngineer
from ..preprocessing.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Automated model training with MLflow tracking."""
    
    def __init__(self, experiment_name: str = "fraud_detection"):
        """
        Initialize model trainer.
        
        Args:
            experiment_name: MLflow experiment name
        """
        self.experiment_name = experiment_name
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        Config.ensure_directories()
        
        # Set up MLflow (with error handling)
        try:
            mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
            mlflow.set_experiment(experiment_name)
            self.mlflow_enabled = True
        except Exception as e:
            logger.warning(f"MLflow not available: {e}. Training without MLflow logging.")
            self.mlflow_enabled = False
        
    def prepare_data(self, sample_size: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare data for training.
        
        Args:
            sample_size: Number of samples to use (for faster training)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Preparing data...")
        
        # Load data
        sample_size = sample_size or Config.SAMPLE_SIZE
        df = self.preprocessor.load_data(Config.RAW_DATA_FILE, sample_size)
        
        # Clean data
        df = self.preprocessor.clean_data(df)
        
        # Encode categorical
        df = self.preprocessor.encode_categorical(df)
        
        # Create features
        df = self.feature_engineer.create_features(df)
        
        # Split data
        train_df, val_df, test_df = self.preprocessor.split_data(df)
        
        return train_df, val_df, test_df
    
    def train_isolation_forest(self, train_df: pd.DataFrame, 
                               val_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train Isolation Forest with hyperparameter tuning.
        
        Args:
            train_df: Training data
            val_df: Validation data
            
        Returns:
            Best metrics achieved
        """
        logger.info("Training Isolation Forest with hyperparameter tuning...")
        
        # Prepare features
        feature_cols = self.feature_engineer.get_feature_columns()
        X_train = train_df[feature_cols]
        y_train = train_df['isFraud']
        X_val = val_df[feature_cols]
        y_val = val_df['isFraud']
        
        # Hyperparameter grid
        param_grid = {
            'contamination': [0.005, 0.01, 0.02],
            'n_estimators': [50, 100, 200]
        }
        
        best_metrics = {}
        best_score = 0
        
        with mlflow.start_run(run_name="isolation_forest_tuning") if self.mlflow_enabled else nullcontext():
            for params in ParameterGrid(param_grid):
                with (mlflow.start_run(nested=True) if self.mlflow_enabled else nullcontext()):
                    # Train model
                    model = IsolationForestModel(
                        contamination=params['contamination'],
                        random_state=42
                    )
                    model.train(X_train, y_train)
                    
                    # Evaluate
                    val_pred = model.predict(X_val)
                    metrics = model._calculate_metrics(y_val, val_pred)
                    
                    # Log to MLflow
                    if self.mlflow_enabled:
                        model.log_to_mlflow(metrics, params)
                    
                    # Track best model
                    if metrics['f1_score'] > best_score:
                        best_score = metrics['f1_score']
                        best_metrics = metrics
                        best_params = params
                        
                        # Save best model
                        model_path = os.path.join(Config.PROCESSED_DATA_PATH, 'isolation_forest_best.pkl')
                        model.save(model_path)
            
            logger.info(f"Best Isolation Forest metrics: {best_metrics}")
            logger.info(f"Best params: {best_params}")
        
        return best_metrics
    
    def train_xgboost(self, train_df: pd.DataFrame, 
                      val_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train XGBoost with hyperparameter tuning.
        
        Args:
            train_df: Training data
            val_df: Validation data
            
        Returns:
            Best metrics achieved
        """
        logger.info("Training XGBoost with hyperparameter tuning...")
        
        # Prepare features
        feature_cols = self.feature_engineer.get_feature_columns()
        X_train = train_df[feature_cols]
        y_train = train_df['isFraud']
        X_val = val_df[feature_cols]
        y_val = val_df['isFraud']
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'scale_pos_weight': [5, 10, 20]
        }
        
        best_metrics = {}
        best_score = 0
        
        with mlflow.start_run(run_name="xgboost_tuning") if self.mlflow_enabled else nullcontext():
            for params in ParameterGrid(param_grid):
                with (mlflow.start_run(nested=True) if self.mlflow_enabled else nullcontext()):
                    # Train model
                    model = XGBoostModel(
                        scale_pos_weight=params['scale_pos_weight'],
                        random_state=42
                    )
                    # Set model parameters
                    model_params = {k: v for k, v in params.items() if k != 'scale_pos_weight'}
                    model.model = xgb.XGBClassifier(**model_params, random_state=42)
                    model.scale_pos_weight = params['scale_pos_weight']
                    
                    metrics = model.train(X_train, y_train, X_val, y_val)
                    
                    # Log to MLflow
                    if self.mlflow_enabled:
                        model.log_to_mlflow(metrics, params)
                    
                    # Track best model
                    if metrics['f1_score'] > best_score:
                        best_score = metrics['f1_score']
                        best_metrics = metrics
                        best_params = params
                        
                        # Save best model
                        model_path = os.path.join(Config.PROCESSED_DATA_PATH, 'xgboost_best.json')
                        model.save(model_path)
            
            logger.info(f"Best XGBoost metrics: {best_metrics}")
            logger.info(f"Best params: {best_params}")
        
        return best_metrics
    
    def train_ensemble(self, train_df: pd.DataFrame, 
                       val_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train ensemble model.
        
        Args:
            train_df: Training data
            val_df: Validation data
            
        Returns:
            Ensemble metrics
        """
        logger.info("Training ensemble model...")
        
        # Prepare features
        feature_cols = self.feature_engineer.get_feature_columns()
        X_train = train_df[feature_cols]
        y_train = train_df['isFraud']
        X_val = val_df[feature_cols]
        y_val = val_df['isFraud']
        
        with mlflow.start_run(run_name="ensemble_model") if self.mlflow_enabled else nullcontext():
            # Train ensemble
            ensemble = EnsembleModel(iso_weight=0.3, xgb_weight=0.7)
            metrics = ensemble.train(X_train, y_train, X_val, y_val)
            
            # Log metrics
            if self.mlflow_enabled:
                mlflow.log_metrics(metrics)
                mlflow.log_params({
                    'iso_weight': ensemble.iso_weight,
                    'xgb_weight': ensemble.xgb_weight
                })
            
            # Save models
            iso_path = os.path.join(Config.PROCESSED_DATA_PATH, 'isolation_forest_ensemble.pkl')
            xgb_path = os.path.join(Config.PROCESSED_DATA_PATH, 'xgboost_ensemble.json')
            ensemble.save(iso_path, xgb_path)
            
            logger.info(f"Ensemble metrics: {metrics}")
        
        return metrics
    
    def run_full_pipeline(self, sample_size: int = None) -> Dict[str, Dict[str, float]]:
        """
        Run complete training pipeline for all models.
        
        Args:
            sample_size: Number of samples to use
            
        Returns:
            Dictionary of metrics for all models
        """
        logger.info("Starting full training pipeline...")
        
        # Prepare data
        train_df, val_df, test_df = self.prepare_data(sample_size)
        
        # Train all models
        iso_metrics = self.train_isolation_forest(train_df, val_df)
        xgb_metrics = self.train_xgboost(train_df, val_df)
        ensemble_metrics = self.train_ensemble(train_df, val_df)
        
        all_metrics = {
            'isolation_forest': iso_metrics,
            'xgboost': xgb_metrics,
            'ensemble': ensemble_metrics
        }
        
        logger.info("Training pipeline completed!")
        logger.info(f"Final metrics: {all_metrics}")
        
        return all_metrics


if __name__ == "__main__":
    trainer = ModelTrainer()
    metrics = trainer.run_full_pipeline(sample_size=100000)
    print("\n=== Training Results ===")
    for model, metrics in metrics.items():
        print(f"\n{model.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
