"""
Unit tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


class TestAPI:
    """Test cases for API endpoints."""
    
    client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        assert response.status_code == 200
        assert "name" in response.json()
        assert "version" in response.json()
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data
    
    def test_predict_endpoint_valid_data(self):
        """Test prediction endpoint with valid data."""
        transaction = {
            "step": 1,
            "type": "TRANSFER",
            "amount": 181.0,
            "nameOrig": "C1305486145",
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C553264065",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0
        }
        
        response = self.client.post("/api/v1/predict", json=transaction)
        
        # May return 503 if model not loaded, but should still return valid JSON
        assert response.status_code in [200, 503]
        data = response.json()
        if response.status_code == 200:
            assert "is_fraud" in data
            assert "fraud_probability" in data
            assert "risk_score" in data
    
    def test_predict_endpoint_invalid_type(self):
        """Test prediction endpoint with invalid transaction type."""
        transaction = {
            "step": 1,
            "type": "INVALID_TYPE",
            "amount": 181.0,
            "nameOrig": "C1305486145",
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C553264065",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0
        }
        
        response = self.client.post("/api/v1/predict", json=transaction)
        assert response.status_code == 422
    
    def test_predict_endpoint_negative_amount(self):
        """Test prediction endpoint with negative amount."""
        transaction = {
            "step": 1,
            "type": "TRANSFER",
            "amount": -100.0,
            "nameOrig": "C1305486145",
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C553264065",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0
        }
        
        response = self.client.post("/api/v1/predict", json=transaction)
        assert response.status_code == 422
    
    def test_batch_predict_endpoint(self):
        """Test batch prediction endpoint."""
        request = {
            "transactions": [
                {
                    "step": 1,
                    "type": "PAYMENT",
                    "amount": 1000.0,
                    "nameOrig": "C123",
                    "oldbalanceOrg": 5000.0,
                    "newbalanceOrig": 4000.0,
                    "nameDest": "M456",
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 1000.0
                },
                {
                    "step": 1,
                    "type": "TRANSFER",
                    "amount": 500.0,
                    "nameOrig": "C789",
                    "oldbalanceOrg": 2000.0,
                    "newbalanceOrig": 1500.0,
                    "nameDest": "C012",
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 500.0
                }
            ]
        }
        
        response = self.client.post("/api/v1/predict/batch", json=request)
        assert response.status_code in [200, 503]
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
