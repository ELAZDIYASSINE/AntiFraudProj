"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from datetime import datetime


class TransactionRequest(BaseModel):
    """Request model for transaction prediction."""
    
    step: int = Field(..., description="Time step in hours")
    type: str = Field(..., description="Transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)")
    amount: float = Field(..., ge=0, description="Transaction amount")
    nameOrig: str = Field(..., description="Origin customer account")
    oldbalanceOrg: float = Field(..., ge=0, description="Initial balance before transaction")
    newbalanceOrig: float = Field(..., ge=0, description="New balance after transaction")
    nameDest: str = Field(..., description="Destination account")
    oldbalanceDest: float = Field(..., ge=0, description="Initial destination balance")
    newbalanceDest: float = Field(..., ge=0, description="New destination balance")
    
    @validator('type')
    def validate_transaction_type(cls, v):
        valid_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
        if v.upper() not in valid_types:
            raise ValueError(f'Transaction type must be one of {valid_types}')
        return v.upper()
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response model for fraud prediction."""
    
    is_fraud: bool = Field(..., description="Whether transaction is fraudulent")
    fraud_probability: float = Field(..., ge=0, le=1, description="Probability of fraud")
    risk_score: float = Field(..., ge=0, le=1, description="Overall risk score")
    model_used: str = Field(..., description="Model used for prediction")
    prediction_time_ms: float = Field(..., description="Prediction latency in milliseconds")
    features: Optional[Dict[str, Any]] = Field(None, description="Computed features")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "is_fraud": True,
                "fraud_probability": 0.95,
                "risk_score": 0.88,
                "model_used": "ensemble",
                "prediction_time_ms": 45.2,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    cache_connected: bool = Field(..., description="Whether Redis cache is connected")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    
    transactions: list[TransactionRequest] = Field(..., description="List of transactions to predict")
    
    @validator('transactions')
    def validate_transactions(cls, v):
        if len(v) == 0:
            raise ValueError('At least one transaction required')
        if len(v) > 100:
            raise ValueError('Maximum 100 transactions per batch')
        return v


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    
    predictions: list[PredictionResponse] = Field(..., description="List of predictions")
    total_predictions: int = Field(..., description="Total number of predictions")
    total_time_ms: float = Field(..., description="Total processing time in milliseconds")
    avg_time_per_prediction_ms: float = Field(..., description="Average time per prediction")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
