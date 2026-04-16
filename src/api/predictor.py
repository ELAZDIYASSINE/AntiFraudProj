"""
Fraud Prediction Service
Handles model loading and inference for the API.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from ..models.xgboost_model import XGBoostModel
from ..models.isolation_forest_model import IsolationForestModel
from ..models.ensemble_model import EnsembleModel
from ..preprocessing.preprocessor import FeatureEngineer
from ..cache.redis_cache import RedisCache
from ..cache.feature_cache import FeatureCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudPredictor:
    """Service for fraud prediction using ML models."""
    
    def __init__(self, model_path: str = None, use_cache: bool = True):
        """
        Initialize fraud predictor.
        
        Args:
            model_path: Path to model directory
            use_cache: Whether to use Redis cache
        """
        self.model_path = model_path or os.getenv('MODEL_PATH', './models')
        self.use_cache = use_cache
        self.feature_engineer = FeatureEngineer()
        self.xgb_model = None
        self.iso_model = None
        self.ensemble_model = None
        self.redis_cache = None
        self.feature_cache = None
        self.model_loaded = False
        
        # Initialize cache if enabled
        if self.use_cache:
            try:
                self.redis_cache = RedisCache()
                self.feature_cache = FeatureCache(self.redis_cache)
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
                self.use_cache = False
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load trained ML models."""
        try:
            # Load XGBoost model
            xgb_path = os.path.join(self.model_path, 'xgboost_ensemble.json')
            if os.path.exists(xgb_path):
                self.xgb_model = XGBoostModel()
                self.xgb_model.load(xgb_path)
                logger.info("XGBoost model loaded")
            
            # Load Isolation Forest model
            iso_path = os.path.join(self.model_path, 'isolation_forest_ensemble.pkl')
            if os.path.exists(iso_path):
                self.iso_model = IsolationForestModel()
                self.iso_model.load(iso_path)
                logger.info("Isolation Forest model loaded")
            
            # Create ensemble if both models loaded
            if self.xgb_model and self.iso_model:
                self.ensemble_model = EnsembleModel()
                self.ensemble_model.iso_model = self.iso_model
                self.ensemble_model.xgb_model = self.xgb_model
                logger.info("Ensemble model ready")
            
            self.model_loaded = bool(self.xgb_model or self.iso_model)
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self.model_loaded = False
    
    def preprocess_transaction(self, transaction: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess transaction data for prediction.
        
        Args:
            transaction: Raw transaction data
            
        Returns:
            Preprocessed DataFrame
        """
        # Convert to DataFrame
        df = pd.DataFrame([transaction])
        
        # Encode categorical
        df['type_encoded'] = df['type'].map({
            'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 
            'PAYMENT': 3, 'TRANSFER': 4
        }).fillna(0)
        
        df['nameOrig_type'] = df['nameOrig'].str[0].map({'C': 0, 'M': 1}).fillna(0)
        df['nameDest_type'] = df['nameDest'].str[0].map({'C': 0, 'M': 1}).fillna(0)
        
        # Create features to match training
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_to_balance_orig'] = df['amount'] / (df['oldbalanceOrg'] + 1)
        df['amount_to_balance_dest'] = df['amount'] / (df['oldbalanceDest'] + 1)
        df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
        df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        df['is_zero_balance_orig'] = (df['oldbalanceOrg'] == 0).astype(int)
        df['is_zero_balance_dest'] = (df['oldbalanceDest'] == 0).astype(int)
        df['day_of_week'] = (df['step'] % 7).astype(int)
        df['hour_of_day'] = (df['step'] % 24).astype(int)
        df['is_cash_out'] = (df['type'] == 'CASH_OUT').astype(int)
        df['is_large_amount'] = (df['amount'] > 10000).astype(int)
        df['is_merchant_dest'] = df['nameDest_type'].astype(int)
        
        # Select feature columns
        feature_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                      'oldbalanceDest', 'newbalanceDest', 'type_encoded',
                      'nameOrig_type', 'nameDest_type', 'amount_log',
                      'amount_to_balance_orig', 'amount_to_balance_dest',
                      'balance_diff_orig', 'balance_diff_dest',
                      'is_zero_balance_orig', 'is_zero_balance_dest',
                      'day_of_week', 'hour_of_day', 'is_cash_out',
                      'is_large_amount', 'is_merchant_dest']
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        return df[available_cols]
    
    def predict(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make fraud prediction for a transaction.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Prediction result
        """
        start_time = time.time()
        
        # Get customer ID
        customer_id = transaction.get('nameOrig', '')
        
        # Compute real-time features if cache enabled
        rt_features = {}
        if self.use_cache and self.feature_cache:
            rt_features = self.feature_cache.compute_all_features(customer_id, transaction)
        
        # Preprocess transaction
        features_df = self.preprocess_transaction(transaction)
        
        # Make prediction
        if self.ensemble_model:
            prediction = self.ensemble_model.predict(features_df)[0]
            probability = self.ensemble_model.predict_proba(features_df)[0]
            model_used = "ensemble"
        elif self.xgb_model:
            prediction = self.xgb_model.predict(features_df)[0]
            probability = self.xgb_model.predict_proba(features_df)[0]
            model_used = "xgboost"
        elif self.iso_model:
            prediction = self.iso_model.predict(features_df)[0]
            probability = self.iso_model.predict_proba(features_df)[0]
            model_used = "isolation_forest"
        else:
            # Fallback: rule-based prediction
            prediction = self._rule_based_prediction(transaction)
            probability = 0.5
            model_used = "rule_based"
        
        # Handle NaN probability
        if pd.isna(probability) or np.isnan(probability):
            probability = 0.5
            logger.warning("Probability is NaN, using default 0.5")
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(transaction, probability, rt_features)
        
        # Cache prediction
        if self.use_cache and self.redis_cache:
            transaction_id = f"{customer_id}_{int(time.time())}"
            self.redis_cache.store_model_prediction(transaction_id, {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_score': float(risk_score),
                'model_used': model_used
            })
        
        # Calculate latency
        prediction_time = (time.time() - start_time) * 1000
        
        return {
            'is_fraud': bool(prediction),
            'fraud_probability': float(probability),
            'risk_score': float(risk_score),
            'model_used': model_used,
            'prediction_time_ms': round(prediction_time, 2),
            'features': rt_features
        }
    
    def _rule_based_prediction(self, transaction: Dict[str, Any]) -> int:
        """
        Fallback rule-based prediction when models are not available.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Prediction (0 or 1)
        """
        # Simple rule-based logic
        amount = transaction.get('amount', 0)
        old_balance = transaction.get('oldbalanceOrg', 0)
        new_balance = transaction.get('newbalanceOrig', 0)
        trans_type = transaction.get('type', '')
        
        # High-risk indicators
        is_large_amount = amount > 1000000
        is_zero_balance = (old_balance > 0 and new_balance == 0)
        is_transfer_or_cashout = trans_type in ['TRANSFER', 'CASH_OUT']
        
        # Combine rules
        if is_large_amount and is_zero_balance and is_transfer_or_cashout:
            return 1
        elif amount > 5000000:
            return 1
        else:
            return 0
    
    def _calculate_risk_score(self, transaction: Dict[str, Any], 
                             probability: float, rt_features: Dict) -> float:
        """
        Calculate comprehensive risk score.
        
        Args:
            transaction: Transaction data
            probability: Model probability
            rt_features: Real-time features
            
        Returns:
            Risk score (0-1)
        """
        # Base score from model probability
        risk_score = probability
        
        # Adjust with real-time features
        if rt_features:
            if rt_features.get('is_high_velocity', 0):
                risk_score = min(1.0, risk_score + 0.1)
            if rt_features.get('is_unusual_time', 0):
                risk_score = min(1.0, risk_score + 0.05)
            if rt_features.get('customer_risk_score', 0) > 0.7:
                risk_score = min(1.0, risk_score + 0.1)
        
        return min(1.0, max(0.0, risk_score))
    
    def predict_batch(self, transactions: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """
        Make predictions for a batch of transactions.
        
        Args:
            transactions: List of transaction data
            
        Returns:
            List of prediction results
        """
        predictions = []
        for transaction in transactions:
            prediction = self.predict(transaction)
            predictions.append(prediction)
        return predictions
    
    def is_ready(self) -> bool:
        """Check if predictor is ready for predictions."""
        return self.model_loaded
    
    def get_cache_status(self) -> bool:
        """Check if cache is connected."""
        return self.use_cache and self.redis_cache is not None
