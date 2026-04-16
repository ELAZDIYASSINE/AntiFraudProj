# 🚀 Deployment Guide

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.9+
- Java 11+ (for PySpark)
- 8GB RAM minimum
- 20GB disk space

## Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd fraudeanti
```

### 2. Environment Configuration

Copy and configure environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Start Infrastructure

```bash
docker-compose up -d
```

This starts:
- Kafka (port 9092)
- Zookeeper (port 2181)
- Redis (port 6379)
- MLflow (port 5000)
- Prometheus (port 9090)
- Grafana (port 3000)

### 4. Install Python Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 5. Train Models

```bash
python src/models/trainer.py
```

This will:
- Load and preprocess data
- Train Isolation Forest, XGBoost, and Ensemble models
- Log metrics to MLflow
- Save best models to `data/processed/`

### 6. Start API

```bash
python src/api/main.py
```

API will be available at http://localhost:8000

### 7. Start Dashboard

```bash
streamlit run src/dashboard/app.py
```

Dashboard will be available at http://localhost:8501

### 8. Start Streaming Pipeline (Optional)

```bash
python src/streaming/pipeline.py
```

## Production Deployment

### Docker Deployment

Build production Docker images:

```dockerfile
# Dockerfile for API
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

Build and run:

```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

### Kubernetes Deployment

Create Kubernetes manifests:

```yaml
# api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detection-api
spec:
  replicas: 4
  selector:
    matchLabels:
      app: fraud-api
  template:
    metadata:
      labels:
        app: fraud-api
    spec:
      containers:
      - name: api
        image: fraud-detection-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models"
        - name: REDIS_HOST
          value: "redis-service"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: fraud-api-service
spec:
  selector:
    app: fraud-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy:

```bash
kubectl apply -f api-deployment.yaml
kubectl apply -f kafka-deployment.yaml
kubectl apply -f redis-deployment.yaml
```

### AWS Deployment

#### Using EKS

```bash
# Create EKS cluster
eksctl create cluster --name fraud-detection --region us-east-1

# Deploy using Helm
helm install fraud-api ./helm-chart

# Configure ALB
kubectl apply -f alb-ingress.yaml
```

#### Using EC2

1. Launch EC2 instances (t3.xlarge or larger)
2. Install Docker and Docker Compose
3. Clone repository
4. Run docker-compose in production mode

### Azure Deployment

#### Using AKS

```bash
# Create resource group
az group create --name fraud-detection --location eastus

# Create AKS cluster
az aks create --resource-group fraud-detection --name fraud-cluster --node-count 4

# Get credentials
az aks get-credentials --resource-group fraud-detection --name fraud-cluster

# Deploy
kubectl apply -f k8s-manifests/
```

### GCP Deployment

#### Using GKE

```bash
# Create cluster
gcloud container clusters create fraud-cluster --num-nodes=4 --zone us-central1-a

# Deploy
kubectl apply -f k8s-manifests/
```

## Monitoring Setup

### Grafana Dashboard

1. Access Grafana: http://localhost:3000
2. Login with admin/admin
3. Navigate to Dashboards → Import
4. Upload `monitoring/grafana/dashboards/dashboard.json`
5. Select Prometheus as datasource

### Prometheus Alerts

Configure alerts in `monitoring/prometheus.yml`:

```yaml
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - 'alerts.yml'
```

Create `alerts.yml`:

```yaml
groups:
  - name: fraud_detection
    rules:
      - alert: HighPredictionLatency
        expr: histogram_quantile(0.95, rate(fraud_prediction_latency_seconds_bucket[5m])) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency detected"
```

## Security Hardening

### TLS/SSL

```nginx
server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
    }
}
```

### Authentication

Add JWT authentication to FastAPI:

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/api/v1/predict")
async def predict_fraud(
    transaction: TransactionRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Verify token
    token = credentials.credentials
    # Validate and proceed
```

### Network Security

- Use VPC/private networks
- Configure security groups
- Enable firewall rules
- Use WAF for API protection

## Scaling Strategies

### Horizontal Scaling

```bash
# Scale API
kubectl scale deployment fraud-detection-api --replicas=8

# Scale Kafka
kafka-topics.sh --alter --topic transactions_input --partitions 6
```

### Vertical Scaling

```yaml
resources:
  requests:
    memory: "8Gi"
    cpu: "4000m"
  limits:
    memory: "16Gi"
    cpu: "8000m"
```

## Backup and Recovery

### Database Backup

```bash
# Redis backup
redis-cli BGSAVE
cp /var/lib/redis/dump.rdb /backup/

# MLflow backup
tar -czf mlflow-backup.tar.gz mlflow-artifacts/
```

### Disaster Recovery

1. Restore from backups
2. Restart services in order:
   - Kafka/Zookeeper
   - Redis
   - MLflow
   - API
   - Monitoring

## Performance Tuning

### PySpark Configuration

```python
conf = {
    "spark.executor.memory": "8g",
    "spark.driver.memory": "4g",
    "spark.executor.cores": "4",
    "spark.sql.shuffle.partitions": "200"
}
```

### Redis Configuration

```conf
maxmemory 4gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
```

### API Optimization

```python
# Increase workers
uvicorn main:app --workers 8 --limit-concurrency 1000

# Enable compression
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

## Troubleshooting

### Common Issues

**Kafka connection failed**:
```bash
# Check Kafka status
docker-compose ps kafka
docker-compose logs kafka

# Restart Kafka
docker-compose restart kafka
```

**Redis connection failed**:
```bash
# Check Redis status
docker-compose ps redis
docker-compose logs redis

# Test connection
redis-cli ping
```

**Model not loaded**:
```bash
# Check model files
ls -la data/processed/

# Re-train models
python src/models/trainer.py
```

**High latency**:
```bash
# Check system resources
htop

# Check API logs
docker-compose logs api

# Scale up
kubectl scale deployment fraud-api --replicas=8
```

## Health Checks

```bash
# API health
curl http://localhost:8000/api/v1/health

# Kafka health
docker-compose exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092

# Redis health
redis-cli ping

# MLflow health
curl http://localhost:5000/health
```

## Log Management

### Centralized Logging (ELK Stack)

```yaml
# docker-compose.yml addition
elasticsearch:
  image: elasticsearch:7.15.0
  ports:
    - "9200:9200"

logstash:
  image: logstash:7.15.0
  ports:
    - "5000:5000"

kibana:
  image: kibana:7.15.0
  ports:
    - "5601:5601"
```

### Log Aggregation

Configure API to send logs to Logstash:

```python
import logging
from logstash import TCPLogstashHandler

logger = logging.getLogger()
logger.addHandler(TCPLogstashHandler('logstash', 5000, version=1))
```

## CI/CD Pipeline

### GitHub Actions

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: docker build -t fraud-api .
      - name: Push to registry
        run: docker push fraud-api
      - name: Deploy to Kubernetes
        run: kubectl apply -f k8s/
```

## Cost Optimization

### AWS Cost Saving Tips

- Use Spot Instances for non-critical workloads
- Enable Auto Scaling
- Use Reserved Instances for baseline load
- Monitor costs with Cost Explorer

### Resource Optimization

```python
# Use smaller models
model = XGBoostModel(
    n_estimators=100,  # Reduced from 200
    max_depth=4  # Reduced from 6
)

# Batch processing
batch_size = 50  # Process in batches
```

## Maintenance

### Regular Tasks

**Daily**:
- Check Grafana alerts
- Review error logs
- Monitor system resources

**Weekly**:
- Review model performance
- Update training data
- Check security patches

**Monthly**:
- Retrain models
- Audit access logs
- Review costs
- Backup configurations

### Model Retraining

```bash
# Automated retraining
python src/models/trainer.py --retrain

# Schedule with cron
0 2 * * 0 cd /app && python src/models/trainer.py --retrain
```

## Support

For issues and questions:
- Check documentation
- Review GitHub issues
- Contact: support@example.com
