"""
FastAPI Application for Fraud Detection API
Provides REST endpoints for real-time fraud prediction.
"""

import os
import time
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import logging
from dotenv import load_dotenv

from .models import (
    TransactionRequest, 
    PredictionResponse, 
    HealthResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)
from .predictor import FraudPredictor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time financial fraud detection using ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None
start_time = time.time()

# Prometheus metrics (initialized lazily)
prediction_counter = None
prediction_latency = None
fraud_detected_counter = None

def get_metrics():
    """Get or initialize Prometheus metrics."""
    global prediction_counter, prediction_latency, fraud_detected_counter
    if prediction_counter is None:
        from prometheus_client import Counter, Histogram, REGISTRY
        # Clear existing metrics to avoid duplicates
        for metric in list(REGISTRY._collector_to_names.keys()):
            if 'fraud' in str(metric):
                REGISTRY.unregister(metric)
        prediction_counter = Counter('fraud_predictions_total', 'Total fraud predictions', ['model', 'result'])
        prediction_latency = Histogram('fraud_prediction_latency_seconds', 'Prediction latency')
        fraud_detected_counter = Counter('fraud_detected_total', 'Total fraud detected')
    return prediction_counter, prediction_latency, fraud_detected_counter

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize predictor on startup."""
    global predictor
    logger.info("Starting Fraud Detection API...")
    predictor = FraudPredictor()
    logger.info("API startup complete")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Fraud Detection API...")
    if predictor and predictor.redis_cache:
        predictor.redis_cache.close()


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Fraud Detection API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/api/v1/health",
            "predict": "/api/v1/predict",
            "batch_predict": "/api/v1/predict/batch",
            "docs": "/docs"
        }
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns service status and model information.
    """
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy" if predictor and predictor.is_ready() else "degraded",
        version="1.0.0",
        model_loaded=predictor.is_ready() if predictor else False,
        cache_connected=predictor.get_cache_status() if predictor else False,
        uptime_seconds=round(uptime, 2)
    )


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fraud(transaction: TransactionRequest):
    """
    Predict fraud for a single transaction.
    
    Args:
        transaction: Transaction data to analyze
        
    Returns:
        Prediction result with fraud probability and risk score
    """
    if not predictor or not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model not loaded. Please train models first."
        )
    
    try:
        # Convert to dictionary
        transaction_dict = transaction.dict()
        
        # Make prediction
        pred_counter, pred_latency, fraud_counter = get_metrics()
        with pred_latency.time():
            result = predictor.predict(transaction_dict)
        
        # Update metrics
        pred_counter.labels(
            model=result['model_used'],
            result='fraud' if result['is_fraud'] else 'legitimate'
        ).inc()
        
        if result['is_fraud']:
            fraud_counter.inc()
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch_fraud(request: BatchPredictionRequest):
    """
    Predict fraud for multiple transactions.
    
    Args:
        request: Batch of transactions to analyze
        
    Returns:
        Batch prediction results with timing information
    """
    if not predictor or not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model not loaded. Please train models first."
        )
    
    try:
        start_time = time.time()
        
        # Convert transactions to dictionaries
        transactions = [t.dict() for t in request.transactions]
        
        # Make predictions
        predictions = predictor.predict_batch(transactions)
        
        # Calculate timing
        total_time = (time.time() - start_time) * 1000
        avg_time = total_time / len(predictions)
        
        # Update metrics
        pred_counter, pred_latency, fraud_counter = get_metrics()
        for pred in predictions:
            pred_counter.labels(
                model=pred['model_used'],
                result='fraud' if pred['is_fraud'] else 'legitimate'
            ).inc()
            if pred['is_fraud']:
                fraud_counter.inc()
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            total_time_ms=round(total_time, 2),
            avg_time_per_prediction_ms=round(avg_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.
    Returns application metrics for monitoring.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/v1/stats", tags=["Monitoring"])
async def get_stats():
    """
    Get API statistics and cache information.
    """
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    
    stats = {
        "uptime_seconds": time.time() - start_time,
        "model_loaded": predictor.is_ready(),
        "cache_enabled": predictor.use_cache,
        "cache_connected": predictor.get_cache_status()
    }
    
    if predictor.redis_cache:
        try:
            stats["cache_stats"] = predictor.redis_cache.get_cache_stats()
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
    
    return stats


@app.get("/api/v1/transactions/search", tags=["Transactions"])
async def search_transactions(nameOrig: str):
    """
    Search transactions by origin account name.
    
    Args:
        nameOrig: Origin account ID to search for
    """
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    
    try:
        # Load the dataset to search
        import pandas as pd
        data_path = os.path.join(os.path.dirname(__file__), '../../data/PS_20174392719_1491204439457_log.csv')
        
        # Read CSV with sample for faster search
        df = pd.read_csv(data_path, nrows=100000)
        
        # Filter by nameOrig
        results = df[df['nameOrig'] == nameOrig]
        
        # Convert to list of dicts
        transactions = results.to_dict('records')
        
        return {
            "count": len(transactions),
            "transactions": transactions[:100]  # Limit to 100 results
        }
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))
    workers = int(os.getenv('API_WORKERS', 4))
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        reload=True
    )
