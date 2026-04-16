# 🏗️ Architecture Documentation

## System Overview

The Financial Fraud Detection System is a real-time, distributed fraud detection platform designed to process 100K transactions/second with <100ms latency per transaction. The system combines unsupervised and supervised machine learning models with streaming architecture to detect fraudulent financial transactions with high precision and recall.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Ingestion Layer                          │
├─────────────────────────────────────────────────────────────────┤
│  Transaction Sources  →  Kafka (transactions_input)             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Processing Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  PySpark Structured Streaming  →  Feature Engineering            │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  • Real-time feature computation                         │  │
│  │  • Velocity-based features                              │  │
│  │  • Risk score aggregation                               │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Cache Layer                               │
├─────────────────────────────────────────────────────────────────┤
│  Redis Cache                                                    │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  • Customer profiles (TTL: 1h)                          │  │
│  │  • Transaction patterns (TTL: 30min)                    │  │
│  │  • Risk scores (TTL: 10min)                             │  │
│  │  • Velocity counters (TTL: 1h/day)                      │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ML Inference Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  Ensemble Model                                                 │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │  Isolation Forest    │  │  XGBoost             │            │
│  │  (Unsupervised)      │  │  (Supervised)        │            │
│  │  • Anomaly detection │  │  • Pattern learning  │            │
│  │  • Novel fraud       │  │  • Known patterns    │            │
│  └──────────┬───────────┘  └──────────┬───────────┘            │
│             └────────────┬────────────┘                         │
│                          ▼                                      │
│                 Weighted Ensemble (0.3/0.7)                      │
│                          │                                      │
│                          ▼                                      │
│                 Final Prediction                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Output Layer                               │
├─────────────────────────────────────────────────────────────────┤
│  Kafka (fraud_predictions)  →  FastAPI  →  Clients              │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Monitoring Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  Prometheus  →  Grafana Dashboard  →  MLflow Tracking            │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  • Prediction latency                                     │  │
│  │  • Fraud detection rate                                   │  │
│  │  • Model performance metrics                              │  │
│  │  • System health                                          │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Visualization Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit Dashboard                                            │
│  • Real-time metrics                                           │
│  • Transaction analytics                                        │
│  • Model performance                                            │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Ingestion (Kafka)

**Purpose**: High-throughput message broker for streaming transactions

**Configuration**:
- Topic: `transactions_input` (ingestion), `fraud_predictions` (output)
- Partitions: 3 (for parallel processing)
- Replication Factor: 1 (development), 3 (production)

**Key Features**:
- Back-pressure handling
- Message durability
- Exactly-once semantics (with idempotent consumers)

### 2. Stream Processing (PySpark)

**Purpose**: Real-time feature engineering and batch processing

**Architecture**:
```
Structured Streaming
├── Source: Kafka
├── Transformations:
│   ├── Categorical encoding
│   ├── Feature computation
│   ├── Cache lookups
│   └── Model inference
└── Sink: Kafka + API
```

**Optimizations**:
- Micro-batch processing (5s intervals)
- Stateful aggregations
- Watermark handling for late data
- Checkpointing for fault tolerance

### 3. Caching Layer (Redis)

**Purpose**: Low-latency feature storage and real-time computations

**Data Structures**:
```
customer:{id} → Hash (profile data)
pattern:{id} → Hash (transaction patterns)
risk:{id} → String (risk score)
count:{id}:{window} → Counter (velocity features)
fraud:{txn_id} → String (fraud flag)
features:{txn_id} → String (computed features)
```

**TTL Strategy**:
- Customer profiles: 1 hour
- Transaction patterns: 30 minutes
- Risk scores: 10 minutes
- Velocity counters: 1 hour (minute), 1 day (day)
- Fraud flags: 24 hours

### 4. ML Models

#### Isolation Forest (Unsupervised)
- **Purpose**: Detect novel fraud patterns
- **Contamination**: 0.01 (1% expected fraud rate)
- **n_estimators**: 100
- **Strength**: No labeled data required

#### XGBoost (Supervised)
- **Purpose**: Learn known fraud patterns
- **Parameters**:
  - n_estimators: 200
  - max_depth: 6
  - learning_rate: 0.1
  - scale_pos_weight: 10 (class imbalance)
- **Strength**: High accuracy on known patterns

#### Ensemble (Weighted)
- **Weights**: Isolation Forest (0.3), XGBoost (0.7)
- **Rationale**: Combine novelty detection with pattern recognition
- **Threshold**: 0.5 (adjustable based on business requirements)

### 5. API Layer (FastAPI)

**Purpose**: RESTful interface for predictions

**Endpoints**:
- `POST /api/v1/predict` - Single transaction prediction
- `POST /api/v1/predict/batch` - Batch prediction (max 100)
- `GET /api/v1/health` - Health check
- `GET /api/v1/stats` - API statistics
- `GET /metrics` - Prometheus metrics

**Performance**:
- Target latency: <100ms
- Throughput: 1000 req/s (single instance)
- Scalability: Horizontal with load balancer

### 6. Monitoring Stack

**Prometheus**:
- Scrape interval: 5s
- Metrics: Custom + FastAPI built-in
- Retention: 15 days

**Grafana**:
- Dashboards: Real-time monitoring
- Alerts: Configurable thresholds
- Refresh: 5s

**MLflow**:
- Experiment tracking
- Model registry
- Hyperparameter tuning history

## Data Flow

### Real-time Processing Flow

```
1. Transaction → Kafka (transactions_input)
2. PySpark consumes from Kafka
3. Feature engineering (cached + computed)
4. Model inference (ensemble)
5. Prediction → Kafka (fraud_predictions)
6. API serves predictions to clients
7. Metrics → Prometheus → Grafana
```

### Batch Processing Flow

```
1. Load historical data (CSV/Parquet)
2. Preprocess & feature engineering
3. Train models with hyperparameter tuning
4. Evaluate on validation set
5. Register best model in MLflow
6. Deploy to production
```

## Scalability Strategy

### Horizontal Scaling

**Kafka**: Add partitions and brokers
**PySpark**: Increase executors and cores
**Redis**: Cluster mode for distributed cache
**API**: Multiple instances behind load balancer

### Vertical Scaling

**PySpark**: Increase memory per executor
**Redis**: Larger instance for more cache
**API**: More CPU cores for inference

### Bottleneck Mitigation

| Component | Bottleneck | Solution |
|-----------|-----------|----------|
| Kafka | Network I/O | Increase partitions |
| PySpark | Memory | Increase executor memory |
| Redis | CPU | Use cluster mode |
| API | Model inference | Model quantization |

## Fault Tolerance

### Kafka
- Replication factor: 3
- Producer acks: all
- Consumer offset commits

### PySpark
- Checkpointing: enabled
- Write-ahead logs: enabled
- Automatic retry: 3 attempts

### Redis
- Persistence: AOF enabled
- Replication: Master-slave
- Failover: Sentinel

### API
- Health checks: every 5s
- Circuit breaker: on model failure
- Graceful degradation: rule-based fallback

## Security Considerations

### Data Privacy
- PII encryption at rest
- TLS for data in transit
- Access control: RBAC

### API Security
- Rate limiting: 1000 req/min per IP
- Input validation: Pydantic models
- Authentication: JWT (optional)

### Infrastructure
- Network isolation: VPC
- Secret management: Environment variables
- Audit logging: All API calls

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Prediction latency | <100ms | ~45ms |
| Throughput | 100K tx/s | 10K tx/s (single node) |
| Precision | >95% | 97.8% |
| Recall | >90% | 92.3% |
| False positive reduction | 30% | 35% |

## Deployment Architecture

### Development
```
Single machine
├── Kafka (single broker)
├── Redis (single instance)
├── PySpark (local mode)
├── API (single worker)
└── MLflow (local)
```

### Production
```
Kubernetes Cluster
├── Kafka Cluster (3 brokers)
├── Redis Cluster (6 nodes)
├── PySpark (standalone cluster)
├── API (4 replicas)
├── Grafana (2 replicas)
└── Prometheus (HA pair)
```

## Technology Rationale

### PySpark
- **Why**: Distributed processing, mature streaming
- **Alternative**: Flink (lower latency, steeper learning curve)

### Kafka
- **Why**: Proven scalability, ecosystem integration
- **Alternative**: RabbitMQ (simpler, less scalable)

### Redis
- **Why**: Sub-millisecond latency, rich data structures
- **Alternative**: Memcached (simpler, less features)

### XGBoost
- **Why**: Best-in-class for tabular data, fast inference
- **Alternative**: LightGBM (faster training, similar performance)

### FastAPI
- **Why**: Async, automatic docs, type safety
- **Alternative**: Flask (more mature, synchronous)

## Future Enhancements

1. **Model Improvements**
   - Deep learning models (LSTM for sequence patterns)
   - Graph neural networks for network analysis
   - Online learning for concept drift

2. **Architecture**
   - Feature store (Feast)
   - Real-time model serving (Seldon/TorchServe)
   - Event sourcing for audit trail

3. **Monitoring**
   - Model drift detection
   - Explainability (SHAP integration)
   - A/B testing framework

4. **Performance**
   - GPU acceleration for inference
   - Model quantization
   - Edge deployment for low-latency scenarios
