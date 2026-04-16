"""
Feature Cache for Real-time Feature Computation
Computes and caches time-based features for streaming transactions.
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from .redis_cache import RedisCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureCache:
    """Cache for real-time feature computation."""
    
    def __init__(self, redis_cache: RedisCache):
        """
        Initialize feature cache.
        
        Args:
            redis_cache: Redis cache instance
        """
        self.cache = redis_cache
        
    def compute_velocity_features(self, customer_id: str, current_amount: float) -> Dict[str, float]:
        """
        Compute velocity-based features for a customer.
        
        Args:
            customer_id: Customer identifier
            current_amount: Current transaction amount
            
        Returns:
            Dictionary of velocity features
        """
        # Get transaction counts
        count_minute = self.cache.get_transaction_count(customer_id, "minute")
        count_hour = self.cache.get_transaction_count(customer_id, "hour")
        count_day = self.cache.get_transaction_count(customer_id, "day")
        
        # Get customer profile
        profile = self.cache.get_customer_profile(customer_id) or {}
        
        # Compute velocity features
        features = {
            "tx_count_minute": count_minute,
            "tx_count_hour": count_hour,
            "tx_count_day": count_day,
            "tx_velocity_hour": count_hour / max(1, (datetime.now().hour + 1)),
            "amount_ratio_to_avg": current_amount / max(1, profile.get("avg_amount", 1)),
            "is_high_velocity": int(count_hour > 10),
            "is_unusual_time": int(datetime.now().hour < 6 or datetime.now().hour > 22)
        }
        
        return features
    
    def update_customer_profile(self, customer_id: str, transaction: Dict):
        """
        Update customer profile with new transaction.
        
        Args:
            customer_id: Customer identifier
            transaction: Transaction data
        """
        profile = self.cache.get_customer_profile(customer_id) or {
            "total_amount": 0,
            "tx_count": 0,
            "avg_amount": 0,
            "max_amount": 0,
            "last_transaction": None
        }
        
        amount = transaction.get("amount", 0)
        
        # Update profile
        profile["total_amount"] += amount
        profile["tx_count"] += 1
        profile["avg_amount"] = profile["total_amount"] / profile["tx_count"]
        profile["max_amount"] = max(profile["max_amount"], amount)
        profile["last_transaction"] = datetime.now().isoformat()
        
        # Store updated profile
        self.cache.store_customer_profile(customer_id, profile, ttl=86400)
        
        # Increment transaction counts
        self.cache.increment_transaction_count(customer_id, "minute")
        self.cache.increment_transaction_count(customer_id, "hour")
        self.cache.increment_transaction_count(customer_id, "day")
        
        logger.debug(f"Updated profile for customer {customer_id}")
    
    def get_risk_features(self, customer_id: str, transaction: Dict) -> Dict[str, float]:
        """
        Get risk-based features for a transaction.
        
        Args:
            customer_id: Customer identifier
            transaction: Transaction data
            
        Returns:
            Dictionary of risk features
        """
        profile = self.cache.get_customer_profile(customer_id) or {}
        risk_score = self.cache.get_risk_score(customer_id) or 0.0
        
        amount = transaction.get("amount", 0)
        
        features = {
            "customer_risk_score": risk_score,
            "is_new_customer": int(profile.get("tx_count", 0) == 0),
            "is_high_amount_customer": int(amount > profile.get("max_amount", 0) * 1.5),
            "customer_age_hours": self._calculate_customer_age(profile),
            "avg_amount_deviation": abs(amount - profile.get("avg_amount", 0)) / max(1, profile.get("avg_amount", 1))
        }
        
        return features
    
    def _calculate_customer_age(self, profile: Dict) -> float:
        """Calculate customer age in hours."""
        if not profile or not profile.get("last_transaction"):
            return 0.0
        
        try:
            last_tx = datetime.fromisoformat(profile["last_transaction"])
            age = datetime.now() - last_tx
            return age.total_seconds() / 3600
        except:
            return 0.0
    
    def cache_transaction_features(self, transaction_id: str, features: Dict, ttl: int = 3600):
        """
        Cache computed features for a transaction.
        
        Args:
            transaction_id: Transaction identifier
            features: Feature dictionary
            ttl: Time to live in seconds
        """
        key = f"features:{transaction_id}"
        self.cache.client.setex(key, ttl, str(features))
        logger.debug(f"Cached features for transaction {transaction_id}")
    
    def get_cached_features(self, transaction_id: str) -> Optional[Dict]:
        """
        Retrieve cached features for a transaction.
        
        Args:
            transaction_id: Transaction identifier
            
        Returns:
            Feature dictionary or None if not found
        """
        key = f"features:{transaction_id}"
        data = self.cache.client.get(key)
        if data:
            return eval(data)
        return None
    
    def compute_all_features(self, customer_id: str, transaction: Dict) -> Dict:
        """
        Compute all cached features for a transaction.
        
        Args:
            customer_id: Customer identifier
            transaction: Transaction data
            
        Returns:
            Dictionary of all computed features
        """
        # Update customer profile
        self.update_customer_profile(customer_id, transaction)
        
        # Compute feature sets
        velocity_features = self.compute_velocity_features(customer_id, transaction.get("amount", 0))
        risk_features = self.get_risk_features(customer_id, transaction)
        
        # Combine all features
        all_features = {
            **velocity_features,
            **risk_features
        }
        
        return all_features
