"""
Unit tests for ML models.
"""

import pytest
import pandas as pd
import numpy as np
from src.models.isolation_forest_model import IsolationForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.ensemble_model import EnsembleModel


class TestIsolationForestModel:
    """Test cases for IsolationForestModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = IsolationForestModel(contamination=0.01)
        assert model.contamination == 0.01
        assert model.model is None
    
    def test_train(self):
        """Test model training."""
        model = IsolationForestModel(contamination=0.01)
        
        X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y_train = np.array([0] * 95 + [1] * 5)
        
        metrics = model.train(X_train, y_train)
        
        assert model.model is not None
        assert 'precision' in metrics
        assert 'recall' in metrics
    
    def test_predict(self):
        """Test model prediction."""
        model = IsolationForestModel(contamination=0.01)
        
        X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y_train = np.array([0] * 95 + [1] * 5)
        
        model.train(X_train, y_train)
        
        X_test = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10)
        })
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert predictions.dtype == np.int64
        assert predictions.all() in [0, 1]


class TestXGBoostModel:
    """Test cases for XGBoostModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = XGBoostModel(scale_pos_weight=10)
        assert model.scale_pos_weight == 10
        assert model.model is None
    
    def test_train(self):
        """Test model training."""
        model = XGBoostModel(scale_pos_weight=10)
        
        X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y_train = np.array([0] * 95 + [1] * 5)
        
        X_val = pd.DataFrame({
            'feature1': np.random.randn(20),
            'feature2': np.random.randn(20)
        })
        y_val = np.array([0] * 18 + [1] * 2)
        
        metrics = model.train(X_train, y_train, X_val, y_val)
        
        assert model.model is not None
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
    
    def test_predict(self):
        """Test model prediction."""
        model = XGBoostModel(scale_pos_weight=10)
        
        X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y_train = np.array([0] * 95 + [1] * 5)
        
        model.train(X_train, y_train)
        
        X_test = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10)
        })
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert predictions.dtype in [np.int32, np.int64]


class TestEnsembleModel:
    """Test cases for EnsembleModel."""
    
    def test_initialization(self):
        """Test ensemble initialization."""
        ensemble = EnsembleModel(iso_weight=0.3, xgb_weight=0.7)
        assert ensemble.iso_weight == 0.3
        assert ensemble.xgb_weight == 0.7
    
    def test_train(self):
        """Test ensemble training."""
        ensemble = EnsembleModel(iso_weight=0.3, xgb_weight=0.7)
        
        X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y_train = np.array([0] * 95 + [1] * 5)
        
        X_val = pd.DataFrame({
            'feature1': np.random.randn(20),
            'feature2': np.random.randn(20)
        })
        y_val = np.array([0] * 18 + [1] * 2)
        
        metrics = ensemble.train(X_train, y_train, X_val, y_val)
        
        assert ensemble.iso_model.model is not None
        assert ensemble.xgb_model.model is not None
        assert 'precision' in metrics
