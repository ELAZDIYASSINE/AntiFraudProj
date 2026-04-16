"""
Redis Cache for Real-time Feature Storage
Caches customer profiles, transaction patterns, and risk scores.
"""

import redis
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import timedelta
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedisCache:
    """Redis cache for real-time fraud detection features."""
    
    def __init__(self, host: str = None, port: int = None, db: int = 0):
        """
        Initialize Redis connection.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
        """
        self.host = host or os.getenv('REDIS_HOST', 'localhost')
        self.port = port or int(os.getenv('REDIS_PORT', 6379))
        self.db = db
        self.client = None
        self.connect()
        
    def connect(self):
        """Connect to Redis server."""
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=True
            )
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def store_customer_profile(self, customer_id: str, profile: Dict, ttl: int = 3600):
        """
        Store customer profile in cache.
        
        Args:
            customer_id: Customer identifier
            profile: Customer profile data
            ttl: Time to live in seconds
        """
        key = f"customer:{customer_id}"
        self.client.setex(key, ttl, json.dumps(profile))
        logger.debug(f"Stored profile for customer {customer_id}")
    
    def get_customer_profile(self, customer_id: str) -> Optional[Dict]:
        """
        Retrieve customer profile from cache.
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Customer profile or None if not found
        """
        key = f"customer:{customer_id}"
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None
    
    def store_transaction_pattern(self, customer_id: str, pattern: Dict, ttl: int = 1800):
        """
        Store transaction pattern for a customer.
        
        Args:
            customer_id: Customer identifier
            pattern: Transaction pattern data
            ttl: Time to live in seconds
        """
        key = f"pattern:{customer_id}"
        self.client.setex(key, ttl, json.dumps(pattern))
        logger.debug(f"Stored pattern for customer {customer_id}")
    
    def get_transaction_pattern(self, customer_id: str) -> Optional[Dict]:
        """
        Retrieve transaction pattern from cache.
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Transaction pattern or None if not found
        """
        key = f"pattern:{customer_id}"
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None
    
    def store_risk_score(self, customer_id: str, score: float, ttl: int = 600):
        """
        Store risk score for a customer.
        
        Args:
            customer_id: Customer identifier
            score: Risk score (0-1)
            ttl: Time to live in seconds
        """
        key = f"risk:{customer_id}"
        self.client.setex(key, ttl, str(score))
        logger.debug(f"Stored risk score {score} for customer {customer_id}")
    
    def get_risk_score(self, customer_id: str) -> Optional[float]:
        """
        Retrieve risk score from cache.
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Risk score or None if not found
        """
        key = f"risk:{customer_id}"
        data = self.client.get(key)
        if data:
            return float(data)
        return None
    
    def store_fraud_flag(self, transaction_id: str, is_fraud: bool, ttl: int = 86400):
        """
        Store fraud flag for a transaction.
        
        Args:
            transaction_id: Transaction identifier
            is_fraud: Whether transaction is fraudulent
            ttl: Time to live in seconds
        """
        key = f"fraud:{transaction_id}"
        self.client.setex(key, ttl, str(int(is_fraud)))
        logger.debug(f"Stored fraud flag for transaction {transaction_id}")
    
    def get_fraud_flag(self, transaction_id: str) -> Optional[bool]:
        """
        Retrieve fraud flag from cache.
        
        Args:
            transaction_id: Transaction identifier
            
        Returns:
            Fraud flag or None if not found
        """
        key = f"fraud:{transaction_id}"
        data = self.client.get(key)
        if data:
            return bool(int(data))
        return None
    
    def increment_transaction_count(self, customer_id: str, window: str = "hour") -> int:
        """
        Increment transaction count for a customer in a time window.
        
        Args:
            customer_id: Customer identifier
            window: Time window (minute, hour, day)
            
        Returns:
            Updated transaction count
        """
        key = f"count:{customer_id}:{window}"
        count = self.client.incr(key)
        
        # Set expiry based on window
        ttl_map = {"minute": 60, "hour": 3600, "day": 86400}
        if count == 1:
            self.client.expire(key, ttl_map.get(window, 3600))
        
        return count
    
    def get_transaction_count(self, customer_id: str, window: str = "hour") -> int:
        """
        Get transaction count for a customer in a time window.
        
        Args:
            customer_id: Customer identifier
            window: Time window (minute, hour, day)
            
        Returns:
            Transaction count
        """
        key = f"count:{customer_id}:{window}"
        data = self.client.get(key)
        if data:
            return int(data)
        return 0
    
    def store_model_prediction(self, transaction_id: str, prediction: Dict, ttl: int = 3600):
        """
        Store model prediction for a transaction.
        
        Args:
            transaction_id: Transaction identifier
            prediction: Prediction data
            ttl: Time to live in seconds
        """
        key = f"prediction:{transaction_id}"
        self.client.setex(key, ttl, json.dumps(prediction))
        logger.debug(f"Stored prediction for transaction {transaction_id}")
    
    def get_model_prediction(self, transaction_id: str) -> Optional[Dict]:
        """
        Retrieve model prediction from cache.
        
        Args:
            transaction_id: Transaction identifier
            
        Returns:
            Prediction data or None if not found
        """
        key = f"prediction:{transaction_id}"
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None
    
    def store_aggregate_stats(self, stats: Dict, ttl: int = 300):
        """
        Store aggregate statistics.
        
        Args:
            stats: Statistics dictionary
            ttl: Time to live in seconds
        """
        key = "stats:aggregate"
        self.client.setex(key, ttl, json.dumps(stats))
        logger.debug("Stored aggregate statistics")
    
    def get_aggregate_stats(self) -> Optional[Dict]:
        """
        Retrieve aggregate statistics from cache.
        
        Returns:
            Statistics dictionary or None if not found
        """
        key = "stats:aggregate"
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None
    
    def clear_customer_data(self, customer_id: str):
        """
        Clear all cached data for a customer.
        
        Args:
            customer_id: Customer identifier
        """
        patterns = [
            f"customer:{customer_id}",
            f"pattern:{customer_id}",
            f"risk:{customer_id}",
            f"count:{customer_id}:*"
        ]
        
        for pattern in patterns:
            if "*" in pattern:
                keys = self.client.keys(pattern)
                if keys:
                    self.client.delete(*keys)
            else:
                self.client.delete(pattern)
        
        logger.info(f"Cleared cache data for customer {customer_id}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get Redis cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        info = self.client.info()
        return {
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": info.get("used_memory_human", "0B"),
            "total_commands_processed": info.get("total_commands_processed", 0),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0)
        }
    
    def flush_all(self):
        """Flush all data from cache (use with caution)."""
        self.client.flushall()
        logger.warning("Flushed all cache data")
    
    def close(self):
        """Close Redis connection."""
        if self.client:
            self.client.close()
            logger.info("Redis connection closed")


if __name__ == "__main__":
    # Test Redis connection
    cache = RedisCache()
    
    # Test basic operations
    cache.store_customer_profile("C123456", {"name": "Test User", "risk_level": "low"})
    profile = cache.get_customer_profile("C123456")
    print(f"Retrieved profile: {profile}")
    
    # Test risk score
    cache.store_risk_score("C123456", 0.15)
    score = cache.get_risk_score("C123456")
    print(f"Risk score: {score}")
    
    # Get cache stats
    stats = cache.get_cache_stats()
    print(f"Cache stats: {stats}")
    
    cache.close()
