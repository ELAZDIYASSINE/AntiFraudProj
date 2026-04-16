# 📖 Setup Guide

## Initial Setup

### 1. System Requirements

**Minimum Requirements**:
- CPU: 4 cores
- RAM: 8GB
- Disk: 20GB SSD
- OS: Ubuntu 20.04+ / Windows 10+ / macOS 11+

**Recommended Requirements**:
- CPU: 8 cores
- RAM: 16GB
- Disk: 50GB SSD
- GPU: Optional for model training

### 2. Software Installation

#### Install Docker

**Ubuntu/Debian**:
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

**macOS**:
```bash
brew install docker docker-compose
```

**Windows**:
Download and install Docker Desktop from https://www.docker.com/products/docker-desktop

#### Install Python

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install python3.9 python3.9-venv python3-pip
```

**macOS**:
```bash
brew install python@3.9
```

**Windows**:
Download from https://www.python.org/downloads/

#### Install Java (for PySpark)

**Ubuntu/Debian**:
```bash
sudo apt install openjdk-11-jdk
```

**macOS**:
```bash
brew install openjdk@11
```

**Windows**:
Download from https://adoptium.net/

Set JAVA_HOME environment variable:
```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

### 3. Project Setup

#### Clone Repository

```bash
git clone <repository-url>
cd fraudeanti
```

#### Create Virtual Environment

```bash
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Verify Installation

```bash
python -c "import pyspark; print(pyspark.__version__)"
python -c "import xgboost; print(xgboost.__version__)"
python -c "import fastapi; print(fastapi.__version__)"
```

### 4. Infrastructure Setup

#### Start Services with Docker Compose

```bash
docker-compose up -d
```

Verify services are running:
```bash
docker-compose ps
```

Expected output:
```
NAME                    STATUS          PORTS
fraudeanti-kafka-1      Up              0.0.0.0:9092->9092
fraudeanti-redis-1      Up              0.0.0.0:6379->6379
fraudeanti-mlflow-1     Up              0.0.0.0:5000->5000
fraudeanti-prometheus-1 Up              0.0.0.0:9090->9090
fraudeanti-grafana-1    Up              0.0.0.0:3000->3000
fraudeanti-zookeeper-1  Up              0.0.0.0:2181->2181
```

#### Test Connections

**Kafka**:
```bash
docker-compose exec kafka kafka-topics.sh --list --bootstrap-server localhost:9092
```

**Redis**:
```bash
docker-compose exec redis redis-cli ping
# Should return PONG
```

**MLflow**:
```bash
curl http://localhost:5000/health
```

**Prometheus**:
```bash
curl http://localhost:9090/-/healthy
```

**Grafana**:
Open browser at http://localhost:3000 (admin/admin)

### 5. Data Preparation

#### Verify Dataset

Check if PaySim dataset exists:
```bash
ls -lh data/PS_20174392719_1491204439457_log.csv
```

Expected size: ~470MB

If missing, download from Kaggle:
```bash
# Install kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d ealaxi/paysim1 -p data/

# Extract
unzip data/paysim1.zip -d data/
```

#### Sample Data (Optional)

For faster training, use a sample:
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/PS_20174392719_1491204439457_log.csv')
sample = df.sample(n=100000, random_state=42)
sample.to_csv('data/PS_20174392719_1491204439457_log_sample.csv', index=False)
print(f'Sampled {len(sample)} rows')
"
```

### 6. Model Training

#### Train All Models

```bash
python src/models/trainer.py
```

This will:
1. Load and preprocess data
2. Train Isolation Forest with hyperparameter tuning
3. Train XGBoost with hyperparameter tuning
4. Train Ensemble model
5. Log all metrics to MLflow
6. Save best models

Expected output:
```
INFO: Preparing data...
INFO: Loaded 100000 transactions
INFO: Cleaning data...
INFO: Training Isolation Forest with hyperparameter tuning...
INFO: Best Isolation Forest metrics: {'precision': 0.96, 'recall': 0.91, 'f1_score': 0.93}
INFO: Training XGBoost with hyperparameter tuning...
INFO: Best XGBoost metrics: {'precision': 0.97, 'recall': 0.92, 'f1_score': 0.95}
INFO: Training ensemble model...
INFO: Ensemble metrics: {'precision': 0.98, 'recall': 0.93, 'f1_score': 0.95}
INFO: Training pipeline completed!
```

#### View Training Results in MLflow

Open browser at http://localhost:5000

Navigate to:
- Experiments → fraud_detection
- View runs and metrics
- Compare model performance

### 7. Start API

#### Development Mode

```bash
python src/api/main.py
```

API will be available at:
- http://localhost:8000
- Interactive docs: http://localhost:8000/docs

#### Production Mode

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Test API

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Test prediction
curl -X POST http://localhost:8000/api/v1/predict \
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
```

### 8. Start Dashboard

```bash
streamlit run src/dashboard/app.py
```

Dashboard will be available at http://localhost:8501

### 9. Start Streaming Pipeline (Optional)

#### Start Kafka Producer

```bash
python src/streaming/kafka_producer.py
```

This will stream sample transactions to Kafka.

#### Start PySpark Pipeline

```bash
python src/streaming/pipeline.py
```

This will:
1. Read from Kafka
2. Process transactions
3. Apply fraud detection
4. Write predictions to Kafka

### 10. Run Tests

#### Unit Tests

```bash
pytest tests/ -v
```

#### With Coverage

```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term
```

View coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

#### Load Tests

```bash
locust -f tests/locustfile.py --host=http://localhost:8000
```

Open browser at http://localhost:8089 to view load test results.

## Configuration

### Environment Variables

Edit `.env` file:

```bash
# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_FRAUD_INPUT=transactions_input
KAFKA_TOPIC_FRAUD_OUTPUT=fraud_predictions

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=fraud_detection

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Spark Configuration
SPARK_MASTER=local[*]
SPARK_APP_NAME=FraudDetection

# Model Configuration
MODEL_PATH=./models
ISOLATION_FOREST_CONTAMINATION=0.01
XGBOOST_SCALE_POS_WEIGHT=10

# Monitoring Configuration
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

### Custom Configuration

Create custom configuration files:

**Spark Configuration** (`spark.conf`):
```properties
spark.executor.memory=4g
spark.driver.memory=2g
spark.executor.cores=2
spark.sql.shuffle.partitions=100
```

**Model Configuration** (`model_config.yaml`):
```yaml
isolation_forest:
  contamination: 0.01
  n_estimators: 100
  max_samples: auto

xgboost:
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.1
  scale_pos_weight: 10

ensemble:
  iso_weight: 0.3
  xgb_weight: 0.7
  threshold: 0.5
```

## Troubleshooting

### Common Issues

#### Port Already in Use

```bash
# Find process using port
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows
```

#### Docker Container Won't Start

```bash
# Check logs
docker-compose logs <service-name>

# Restart service
docker-compose restart <service-name>

# Rebuild
docker-compose up -d --build <service-name>
```

#### Python Import Errors

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### Memory Issues

```bash
# Increase Spark memory
export PYSPARK_DRIVER_MEMORY=4g
export PYSPARK_EXECUTOR_MEMORY=8g

# Use smaller sample
python src/models/trainer.py --sample-size 50000
```

#### Kafka Connection Issues

```bash
# Check Kafka is running
docker-compose ps kafka

# Test Kafka connection
docker-compose exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092

# Restart Kafka
docker-compose restart kafka zookeeper
```

#### Redis Connection Issues

```bash
# Check Redis is running
docker-compose ps redis

# Test Redis connection
docker-compose exec redis redis-cli ping

# Check Redis logs
docker-compose logs redis
```

## Verification Checklist

After setup, verify all components:

- [ ] Docker and Docker Compose installed
- [ ] Python 3.9+ installed
- [ ] Java 11+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Docker services running (6 services)
- [ ] Kafka accessible on port 9092
- [ ] Redis accessible on port 6379
- [ ] MLflow accessible on port 5000
- [ ] Prometheus accessible on port 9090
- [ ] Grafana accessible on port 3000
- [ ] Dataset downloaded and verified
- [ ] Models trained successfully
- [ ] API running on port 8000
- [ ] API health check returns healthy
- [ ] Dashboard running on port 8501
- [ ] Unit tests passing
- [ ] Load tests successful

## Next Steps

1. **Explore the API**: Try different transactions in the interactive docs
2. **Monitor Performance**: Check Grafana dashboard
3. **Review Model Metrics**: View MLflow experiment results
4. **Customize Models**: Adjust hyperparameters in `src/models/trainer.py`
5. **Scale Up**: Deploy to production using deployment guide
6. **Integrate**: Connect to your transaction processing system

## Getting Help

- **Documentation**: Check `README.md`, `ARCHITECTURE.md`, `API_DOCUMENTATION.md`
- **Issues**: Report bugs on GitHub
- **Community**: Join our Discord/Slack channel
- **Support**: Contact support@example.com
