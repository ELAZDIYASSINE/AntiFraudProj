"""
Unit tests for Redis cache module.
"""

import pytest
from src.cache.redis_cache import RedisCache


class TestRedisCache:
    """Test cases for Redis cache."""
    
    @pytest.fixture
    def cache(self):
        """Create cache instance for testing."""
        try:
            cache = RedisCache()
            return cache
        except:
            pytest.skip("Redis not available")
    
    def test_customer_profile_operations(self, cache):
        """Test customer profile storage and retrieval."""
        customer_id = "TEST123"
        profile = {"name": "Test User", "risk_level": "low"}
        
        cache.store_customer_profile(customer_id, profile, ttl=60)
        retrieved = cache.get_customer_profile(customer_id)
        
        assert retrieved is not None
        assert retrieved["name"] == "Test User"
    
    def test_risk_score_operations(self, cache):
        """Test risk score storage and retrieval."""
        customer_id = "TEST456"
        score = 0.75
        
        cache.store_risk_score(customer_id, score, ttl=60)
        retrieved = cache.get_risk_score(customer_id)
        
        assert retrieved == score
    
    def test_transaction_count(self, cache):
        """Test transaction count increment."""
        customer_id = "TEST789"
        
        count1 = cache.increment_transaction_count(customer_id, "minute")
        count2 = cache.increment_transaction_count(customer_id, "minute")
        
        assert count2 == count1 + 1
    
    def test_fraud_flag_operations(self, cache):
        """Test fraud flag storage and retrieval."""
        transaction_id = "TXN12345"
        
        cache.store_fraud_flag(transaction_id, True, ttl=60)
        retrieved = cache.get_fraud_flag(transaction_id)
        
        assert retrieved is True
    
    def test_cache_stats(self, cache):
        """Test cache statistics retrieval."""
        stats = cache.get_cache_stats()
        
        assert "connected_clients" in stats
        assert "used_memory_human" in stats
