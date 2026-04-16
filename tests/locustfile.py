"""
Load testing script using Locust.
Tests API performance under high load.
"""

from locust import HttpUser, task, between
import random


class FraudDetectionUser(HttpUser):
    """Simulated user for load testing."""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Actions performed when a user starts."""
        self.client.get("/api/v1/health")
    
    @task(3)
    def predict_fraud(self):
        """Test fraud prediction endpoint."""
        transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT']
        
        transaction = {
            "step": random.randint(1, 744),
            "type": random.choice(transaction_types),
            "amount": round(random.uniform(100, 100000), 2),
            "nameOrig": f"C{random.randint(1000000000, 9999999999)}",
            "oldbalanceOrg": round(random.uniform(0, 100000), 2),
            "newbalanceOrig": round(random.uniform(0, 100000), 2),
            "nameDest": f"{'C' if random.random() > 0.5 else 'M'}{random.randint(1000000000, 9999999999)}",
            "oldbalanceDest": round(random.uniform(0, 100000), 2),
            "newbalanceDest": round(random.uniform(0, 100000), 2)
        }
        
        self.client.post("/api/v1/predict", json=transaction)
    
    @task(1)
    def health_check(self):
        """Test health check endpoint."""
        self.client.get("/api/v1/health")
    
    @task(1)
    def batch_predict(self):
        """Test batch prediction endpoint."""
        transactions = []
        for _ in range(5):
            transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT']
            transaction = {
                "step": random.randint(1, 744),
                "type": random.choice(transaction_types),
                "amount": round(random.uniform(100, 100000), 2),
                "nameOrig": f"C{random.randint(1000000000, 9999999999)}",
                "oldbalanceOrg": round(random.uniform(0, 100000), 2),
                "newbalanceOrig": round(random.uniform(0, 100000), 2),
                "nameDest": f"{'C' if random.random() > 0.5 else 'M'}{random.randint(1000000000, 9999999999)}",
                "oldbalanceDest": round(random.uniform(0, 100000), 2),
                "newbalanceDest": round(random.uniform(0, 100000), 2)
            }
            transactions.append(transaction)
        
        self.client.post("/api/v1/predict/batch", json={"transactions": transactions})
