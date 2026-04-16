"""Configuration for streaming pipeline."""

import os
from dotenv import load_dotenv

load_dotenv()

class StreamingConfig:
    """Configuration settings for streaming pipeline."""
    
    # Kafka
    KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    KAFKA_TOPIC_INPUT = os.getenv('KAFKA_TOPIC_FRAUD_INPUT', 'transactions_input')
    KAFKA_TOPIC_OUTPUT = os.getenv('KAFKA_TOPIC_FRAUD_OUTPUT', 'fraud_predictions')
    
    # Spark
    SPARK_APP_NAME = os.getenv('SPARK_APP_NAME', 'FraudDetection')
    SPARK_MASTER = os.getenv('SPARK_MASTER', 'local[*]')
    CHECKPOINT_LOCATION = './checkpoints'
    
    # Processing
    BATCH_INTERVAL_SECONDS = 5
    MAX_EVENTS_PER_BATCH = 1000
    
    # Model
    MODEL_PATH = os.getenv('MODEL_PATH', './models')
