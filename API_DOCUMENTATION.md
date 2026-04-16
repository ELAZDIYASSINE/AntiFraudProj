# 📚 API Documentation

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production, implement JWT or OAuth2.

## Endpoints

### 1. Root Endpoint

Get API information and available endpoints.

```http
GET /
```

**Response**:
```json
{
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
```

---

### 2. Health Check

Check API health and model status.

```http
GET /api/v1/health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "cache_connected": true,
  "uptime_seconds": 1234.56,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

**Status Values**:
- `healthy`: All systems operational
- `degraded`: Partial functionality (e.g., cache disconnected)
- `unhealthy`: Critical failure

---

### 3. Predict Fraud (Single Transaction)

Predict fraud for a single transaction.

```http
POST /api/v1/predict
Content-Type: application/json
```

**Request Body**:
```json
{
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
```

**Field Descriptions**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| step | integer | Yes | Time step in hours (1-744) |
| type | string | Yes | Transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER) |
| amount | float | Yes | Transaction amount (>= 0) |
| nameOrig | string | Yes | Origin customer account ID |
| oldbalanceOrg | float | Yes | Initial balance before transaction (>= 0) |
| newbalanceOrig | float | Yes | New balance after transaction (>= 0) |
| nameDest | string | Yes | Destination account ID |
| oldbalanceDest | float | Yes | Initial destination balance (>= 0) |
| newbalanceDest | float | Yes | New destination balance (>= 0) |

**Response**:
```json
{
  "is_fraud": true,
  "fraud_probability": 0.95,
  "risk_score": 0.88,
  "model_used": "ensemble",
  "prediction_time_ms": 45.2,
  "features": {
    "tx_count_hour": 5,
    "is_high_velocity": 0,
    "customer_risk_score": 0.15
  },
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| is_fraud | boolean | Whether transaction is fraudulent |
| fraud_probability | float | Probability of fraud (0-1) |
| risk_score | float | Overall risk score (0-1) |
| model_used | string | Model used for prediction (ensemble, xgboost, isolation_forest, rule_based) |
| prediction_time_ms | float | Prediction latency in milliseconds |
| features | object | Computed real-time features (optional) |
| timestamp | datetime | Prediction timestamp |

**Error Responses**:

**422 Unprocessable Entity**:
```json
{
  "detail": [
    {
      "loc": ["body", "type"],
      "msg": "Transaction type must be one of ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']",
      "type": "value_error"
    }
  ]
}
```

**503 Service Unavailable**:
```json
{
  "detail": "ML model not loaded. Please train models first."
}
```

---

### 4. Batch Prediction

Predict fraud for multiple transactions (max 100 per request).

```http
POST /api/v1/predict/batch
Content-Type: application/json
```

**Request Body**:
```json
{
  "transactions": [
    {
      "step": 1,
      "type": "PAYMENT",
      "amount": 1000.0,
      "nameOrig": "C1234567890",
      "oldbalanceOrg": 5000.0,
      "newbalanceOrig": 4000.0,
      "nameDest": "M9876543210",
      "oldbalanceDest": 0.0,
      "newbalanceDest": 1000.0
    },
    {
      "step": 1,
      "type": "TRANSFER",
      "amount": 500.0,
      "nameOrig": "C0987654321",
      "oldbalanceOrg": 2000.0,
      "newbalanceOrig": 1500.0,
      "nameDest": "C1234567890",
      "oldbalanceDest": 0.0,
      "newbalanceDest": 500.0
    }
  ]
}
```

**Response**:
```json
{
  "predictions": [
    {
      "is_fraud": false,
      "fraud_probability": 0.12,
      "risk_score": 0.15,
      "model_used": "ensemble",
      "prediction_time_ms": 42.1,
      "timestamp": "2024-01-15T10:30:00.000Z"
    },
    {
      "is_fraud": false,
      "fraud_probability": 0.08,
      "risk_score": 0.12,
      "model_used": "ensemble",
      "prediction_time_ms": 43.5,
      "timestamp": "2024-01-15T10:30:00.000Z"
    }
  ],
  "total_predictions": 2,
  "total_time_ms": 85.6,
  "avg_time_per_prediction_ms": 42.8,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

**Constraints**:
- Minimum: 1 transaction
- Maximum: 100 transactions per batch
- All transactions must be valid

---

### 5. API Statistics

Get API statistics and cache information.

```http
GET /api/v1/stats
```

**Response**:
```json
{
  "uptime_seconds": 1234.56,
  "model_loaded": true,
  "cache_enabled": true,
  "cache_connected": true,
  "cache_stats": {
    "connected_clients": 5,
    "used_memory_human": "45.2M",
    "total_commands_processed": 12345,
    "keyspace_hits": 9876,
    "keyspace_misses": 469
  }
}
```

---

### 6. Prometheus Metrics

Get Prometheus metrics for monitoring.

```http
GET /metrics
```

**Response**: Plain text Prometheus format

**Available Metrics**:
- `fraud_predictions_total{model, result}` - Total predictions by model and result
- `fraud_prediction_latency_seconds` - Prediction latency histogram
- `fraud_detected_total` - Total fraud detected

---

## Interactive Documentation

Interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Rate Limiting

Currently, no rate limiting is enforced. In production, implement:

- 1000 requests per minute per IP
- 10,000 requests per minute per API key
- Burst: 100 requests per second

---

## Error Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 422 | Validation Error |
| 500 | Internal Server Error |
| 503 | Service Unavailable (model not loaded) |

---

## Example Usage

### cURL

```bash
# Single prediction
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "step": 1,
    "type": "TRANSFER",
    "amount": 181.0,
    "nameOrig": "C1305486145",
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "nameDest": "C553264065",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
  }'

# Health check
curl "http://localhost:8000/api/v1/health"

# Batch prediction
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "step": 1,
        "type": "PAYMENT",
        "amount": 1000.0,
        "nameOrig": "C1234567890",
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 4000.0,
        "nameDest": "M9876543210",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 1000.0
      }
    ]
  }'
```

### Python

```python
import requests

API_URL = "http://localhost:8000"

# Single prediction
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

response = requests.post(f"{API_URL}/api/v1/predict", json=transaction)
result = response.json()
print(f"Fraud: {result['is_fraud']}, Probability: {result['fraud_probability']:.2f}")

# Health check
health = requests.get(f"{API_URL}/api/v1/health").json()
print(f"Status: {health['status']}")
```

### JavaScript

```javascript
const API_URL = 'http://localhost:8000';

// Single prediction
async function predictFraud(transaction) {
  const response = await fetch(`${API_URL}/api/v1/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(transaction)
  });
  return response.json();
}

const transaction = {
  step: 1,
  type: 'TRANSFER',
  amount: 181.0,
  nameOrig: 'C1305486145',
  oldbalanceOrg: 181.0,
  newbalanceOrig: 0.0,
  nameDest: 'C553264065',
  oldbalanceDest: 0.0,
  newbalanceDest: 0.0
};

predictFraud(transaction).then(result => {
  console.log(`Fraud: ${result.is_fraud}, Probability: ${result.fraud_probability}`);
});
```

---

## WebSocket Support (Future)

Real-time fraud alerts via WebSocket (planned feature):

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/fraud-alerts');

ws.onmessage = (event) => {
  const alert = JSON.parse(event.data);
  console.log('Fraud detected:', alert);
};
```

---

## SDKs (Future)

Official SDKs planned for:
- Python
- JavaScript/TypeScript
- Java
- Go
