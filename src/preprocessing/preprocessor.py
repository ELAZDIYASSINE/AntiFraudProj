"""
Data Preprocessing and Feature Engineering Module
Handles data cleaning, transformation, and feature creation for fraud detection.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data cleaning and basic preprocessing."""
    
    def __init__(self):
        self.transaction_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
        
    def load_data(self, filepath: str, sample_size: int = None) -> pd.DataFrame:
        """Load transaction data from CSV."""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        if sample_size:
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} rows")
            
        logger.info(f"Loaded {len(df)} transactions")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate transaction data."""
        logger.info("Cleaning data...")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_count - len(df)} duplicate transactions")
        
        # Handle missing values
        df = df.dropna()
        
        # Validate amounts are positive
        df = df[df['amount'] >= 0]
        
        # Validate balance changes
        df = df[(df['oldbalanceOrg'] >= 0) & (df['newbalanceOrig'] >= 0)]
        df = df[(df['oldbalanceDest'] >= 0) & (df['newbalanceDest'] >= 0)]
        
        logger.info(f"Cleaned data: {len(df)} transactions remaining")
        return df
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        
        # Encode transaction type
        type_mapping = {t: i for i, t in enumerate(self.transaction_types)}
        df['type_encoded'] = df['type'].map(type_mapping)
        
        # Extract customer type from account names
        df['nameOrig_type'] = df['nameOrig'].str[0].map({'C': 0, 'M': 1})
        df['nameDest_type'] = df['nameDest'].str[0].map({'C': 0, 'M': 1})
        
        return df
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                   val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        # Sort by step to maintain temporal order
        df = df.sort_values('step')
        
        # Calculate split points
        n = len(df)
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df


class FeatureEngineer:
    """Advanced feature engineering for fraud detection."""
    
    def __init__(self):
        self.feature_names = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced fraud detection features."""
        logger.info("Engineering features...")
        df = df.copy()
        
        # Basic transaction features
        df['amount_log'] = np.log1p(df['amount'])
        df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
        df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        
        # Ratio features
        df['amount_to_balance_orig'] = np.where(
            df['oldbalanceOrg'] > 0,
            df['amount'] / df['oldbalanceOrg'],
            0
        )
        df['amount_to_balance_dest'] = np.where(
            df['oldbalanceDest'] > 0,
            df['amount'] / df['oldbalanceDest'],
            0
        )
        
        # Transaction behavior features
        df['is_zero_balance_orig'] = (df['newbalanceOrig'] == 0).astype(int)
        df['is_zero_balance_dest'] = (df['newbalanceDest'] == 0).astype(int)
        df['is_large_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        
        # Fraud pattern indicators
        df['is_transfer'] = (df['type'] == 'TRANSFER').astype(int)
        df['is_cash_out'] = (df['type'] == 'CASH_OUT').astype(int)
        df['is_merchant_dest'] = df['nameDest'].str[0].map({'C': 0, 'M': 1}).fillna(0).astype(int)
        
        # Time-based features (step represents hours)
        df['hour_of_day'] = df['step'] % 24
        df['day_of_week'] = (df['step'] // 24) % 7
        
        # Risk score features
        df['risk_amount'] = np.where(df['amount'] > 1000000, 1, 0)
        df['risk_zero_balance'] = np.where(
            (df['balance_diff_orig'] == df['amount']) & (df['newbalanceDest'] == 0),
            1, 0
        )
        
        # Store feature names
        self.feature_names = [
            'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest', 'type_encoded',
            'nameOrig_type', 'nameDest_type', 'amount_log',
            'balance_diff_orig', 'balance_diff_dest',
            'amount_to_balance_orig', 'amount_to_balance_dest',
            'is_zero_balance_orig', 'is_zero_balance_dest',
            'is_large_amount', 'is_transfer', 'is_cash_out',
            'is_merchant_dest', 'hour_of_day', 'day_of_week',
            'risk_amount', 'risk_zero_balance'
        ]
        
        logger.info(f"Created {len(self.feature_names)} features")
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Return list of feature column names."""
        return self.feature_names
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select only engineered features for modeling."""
        available_features = [f for f in self.feature_names if f in df.columns]
        return df[available_features]
    
    def calculate_fraud_rate(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate fraud statistics by transaction type."""
        stats = {}
        for trans_type in df['type'].unique():
            type_df = df[df['type'] == trans_type]
            fraud_rate = type_df['isFraud'].mean()
            stats[trans_type] = fraud_rate
        return stats
