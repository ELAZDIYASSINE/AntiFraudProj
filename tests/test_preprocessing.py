"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing.preprocessor import DataPreprocessor, FeatureEngineer


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def test_encode_categorical(self):
        """Test categorical encoding."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT'],
            'nameOrig': ['C123', 'M456', 'C789'],
            'nameDest': ['M123', 'C456', 'M789']
        })
        
        result = preprocessor.encode_categorical(df)
        
        assert 'type_encoded' in result.columns
        assert 'nameOrig_type' in result.columns
        assert 'nameDest_type' in result.columns
        assert result['nameOrig_type'].isin([0, 1]).all()
    
    def test_split_data(self):
        """Test data splitting."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'step': range(100),
            'isFraud': [0] * 95 + [1] * 5
        })
        
        train, val, test = preprocessor.split_data(df)
        
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == len(df)


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""
    
    def test_create_features(self):
        """Test feature creation."""
        engineer = FeatureEngineer()
        df = pd.DataFrame({
            'step': [1, 2],
            'type': ['PAYMENT', 'TRANSFER'],
            'amount': [1000, 5000],
            'oldbalanceOrg': [5000, 10000],
            'newbalanceOrig': [4000, 5000],
            'oldbalanceDest': [0, 1000],
            'newbalanceDest': [1000, 6000],
            'nameOrig': ['C123', 'C456'],
            'nameDest': ['M123', 'C789'],
            'type_encoded': [3, 4],
            'nameOrig_type': [0, 0],
            'nameDest_type': [1, 0]
        })
        
        result = engineer.create_features(df)
        
        assert 'amount_log' in result.columns
        assert 'balance_diff_orig' in result.columns
        assert 'balance_diff_dest' in result.columns
        assert 'is_large_amount' in result.columns
        assert len(result) == len(df)
    
    def test_get_feature_columns(self):
        """Test feature column retrieval."""
        engineer = FeatureEngineer()
        features = engineer.get_feature_columns()
        
        assert isinstance(features, list)
        assert len(features) > 0
        assert 'amount' in features
        assert 'step' in features
