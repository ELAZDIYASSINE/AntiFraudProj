"""
Automated ML Pipeline with Optuna Hyperparameter Tuning
Advanced hyperparameter optimization with MLflow tracking.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import train_test_split
import logging
import xgboost as xgb

from src.models.isolation_forest_model import IsolationForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.ensemble_model import EnsembleModel
from src.preprocessing.preprocessor import DataPreprocessor, FeatureEngineer
from src.preprocessing.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoMLPipeline:
    """Automated ML pipeline with Optuna hyperparameter tuning."""
    
    def __init__(self, experiment_name: str = "fraud_detection_automl"):
        """
        Initialize AutoML pipeline.
        
        Args:
            experiment_name: MLflow experiment name
        """
        self.experiment_name = experiment_name
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        Config.ensure_directories()
        
        # Set up MLflow
        try:
            mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
            mlflow.set_experiment(experiment_name)
            self.mlflow_enabled = True
        except Exception as e:
            logger.warning(f"MLflow not available: {e}. Training without MLflow logging.")
            self.mlflow_enabled = False
        
        # Optuna study storage
        self.study_storage = f"sqlite:///{Config.PROCESSED_DATA_PATH}/optuna_studies.db"
        
    def prepare_data(self, sample_size: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare data for training.
        
        Args:
            sample_size: Number of samples to use
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Preparing data...")
        
        sample_size = sample_size or Config.SAMPLE_SIZE
        df = self.preprocessor.load_data(Config.RAW_DATA_FILE, sample_size)
        df = self.preprocessor.clean_data(df)
        df = self.preprocessor.encode_categorical(df)
        df = self.feature_engineer.create_features(df)
        
        train_df, val_df, test_df = self.preprocessor.split_data(df)
        
        return train_df, val_df, test_df
    
    def objective_xgboost(self, trial: optuna.Trial, X_train: pd.DataFrame, 
                          y_train: pd.Series, X_val: pd.DataFrame, 
                          y_val: pd.Series) -> float:
        """
        Optuna objective function for XGBoost hyperparameter tuning.
        
        Args:
            trial: Optuna trial
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            F1 score to maximize
        """
        # Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 50),
            'random_state': 42
        }
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # Predict and calculate F1 score
        y_pred = model.predict(X_val)
        from sklearn.metrics import f1_score
        score = f1_score(y_val, y_pred)
        
        # Log to MLflow
        if self.mlflow_enabled:
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric('f1_score', score)
        
        return score
    
    def objective_isolation_forest(self, trial: optuna.Trial, X_train: pd.DataFrame, 
                                  y_train: pd.Series, X_val: pd.DataFrame, 
                                  y_val: pd.Series) -> float:
        """
        Optuna objective function for Isolation Forest hyperparameter tuning.
        
        Args:
            trial: Optuna trial
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            F1 score to maximize
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.metrics import f1_score
        
        # Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'contamination': trial.suggest_float('contamination', 0.001, 0.05),
            'max_samples': trial.suggest_float('max_samples', 0.1, 1.0),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
            'random_state': 42
        }
        
        # Train model
        model = IsolationForest(**params)
        model.fit(X_train)
        
        # Predict (Isolation Forest returns -1 for anomalies, 1 for normal)
        y_pred = model.predict(X_val)
        y_pred = [1 if x == -1 else 0 for x in y_pred]  # Convert to fraud (1) vs normal (0)
        
        # Calculate F1 score
        score = f1_score(y_val, y_pred)
        
        # Log to MLflow
        if self.mlflow_enabled:
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric('f1_score', score)
        
        return score
    
    def tune_xgboost(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                     n_trials: int = 50) -> Tuple[Dict, float]:
        """
        Tune XGBoost hyperparameters using Optuna.
        
        Args:
            train_df: Training data
            val_df: Validation data
            n_trials: Number of Optuna trials
            
        Returns:
            Tuple of (best_params, best_score)
        """
        logger.info(f"Starting XGBoost hyperparameter tuning with {n_trials} trials...")
        
        feature_cols = self.feature_engineer.get_feature_columns()
        X_train = train_df[feature_cols]
        y_train = train_df['isFraud']
        X_val = val_df[feature_cols]
        y_val = val_df['isFraud']
        
        # Create study
        study = optuna.create_study(
            study_name='xgboost_fraud_detection',
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=20),
            storage=self.study_storage,
            load_if_exists=True
        )
        
        # Run optimization
        with mlflow.start_run(run_name="xgboost_optuna_tuning") if self.mlflow_enabled else nullcontext():
            study.optimize(
                lambda trial: self.objective_xgboost(trial, X_train, y_train, X_val, y_val),
                n_trials=n_trials,
                show_progress_bar=True
            )
        
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Best XGBoost params: {best_params}")
        logger.info(f"Best F1 score: {best_score:.4f}")
        
        # Train final model with best params
        best_params['random_state'] = 42
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_train, y_train)
        
        # Save best model
        model_path = os.path.join(Config.PROCESSED_DATA_PATH, 'xgboost_optuna_best.json')
        final_model.save_model(model_path)
        
        return best_params, best_score
    
    def tune_isolation_forest(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                              n_trials: int = 50) -> Tuple[Dict, float]:
        """
        Tune Isolation Forest hyperparameters using Optuna.
        
        Args:
            train_df: Training data
            val_df: Validation data
            n_trials: Number of Optuna trials
            
        Returns:
            Tuple of (best_params, best_score)
        """
        logger.info(f"Starting Isolation Forest hyperparameter tuning with {n_trials} trials...")
        
        feature_cols = self.feature_engineer.get_feature_columns()
        X_train = train_df[feature_cols]
        y_train = train_df['isFraud']
        X_val = val_df[feature_cols]
        y_val = val_df['isFraud']
        
        # Create study
        study = optuna.create_study(
            study_name='isolation_forest_fraud_detection',
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=20),
            storage=self.study_storage,
            load_if_exists=True
        )
        
        # Run optimization
        with mlflow.start_run(run_name="isolation_forest_optuna_tuning") if self.mlflow_enabled else nullcontext():
            study.optimize(
                lambda trial: self.objective_isolation_forest(trial, X_train, y_train, X_val, y_val),
                n_trials=n_trials,
                show_progress_bar=True
            )
        
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Best Isolation Forest params: {best_params}")
        logger.info(f"Best F1 score: {best_score:.4f}")
        
        # Train final model with best params
        from sklearn.ensemble import IsolationForest
        best_params['random_state'] = 42
        final_model = IsolationForest(**best_params)
        final_model.fit(X_train)
        
        # Save best model
        model_path = os.path.join(Config.PROCESSED_DATA_PATH, 'isolation_forest_optuna_best.pkl')
        import joblib
        joblib.dump(final_model, model_path)
        
        return best_params, best_score
    
    def run_automl_pipeline(self, sample_size: int = None, 
                           n_trials: int = 50) -> Dict[str, Dict]:
        """
        Run complete AutoML pipeline.
        
        Args:
            sample_size: Number of samples to use
            n_trials: Number of Optuna trials per model
            
        Returns:
            Dictionary of best results for all models
        """
        logger.info("Starting AutoML pipeline...")
        
        # Prepare data
        train_df, val_df, test_df = self.prepare_data(sample_size)
        
        # Tune models
        xgb_params, xgb_score = self.tune_xgboost(train_df, val_df, n_trials)
        iso_params, iso_score = self.tune_isolation_forest(train_df, val_df, n_trials)
        
        results = {
            'xgboost': {
                'best_params': xgb_params,
                'best_f1_score': xgb_score
            },
            'isolation_forest': {
                'best_params': iso_params,
                'best_f1_score': iso_score
            }
        }
        
        logger.info("AutoML pipeline completed!")
        logger.info(f"Results: {results}")
        
        return results


if __name__ == "__main__":
    from contextlib import nullcontext
    
    automl = AutoMLPipeline()
    results = automl.run_automl_pipeline(sample_size=100000, n_trials=30)
    
    print("\n=== AutoML Results ===")
    for model, result in results.items():
        print(f"\n{model.upper()}:")
        print(f"  Best F1 Score: {result['best_f1_score']:.4f}")
        print(f"  Best Parameters: {result['best_params']}")
