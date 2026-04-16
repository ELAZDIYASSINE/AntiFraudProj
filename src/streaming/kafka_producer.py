"""
Kafka Producer for Streaming Transactions
Sends transaction data to Kafka for real-time processing.
"""

import json
import time
import pandas as pd
from kafka import KafkaProducer
from typing import Dict, List
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionProducer:
    """Kafka producer for streaming transaction data."""
    
    def __init__(self, bootstrap_servers: str = None, topic: str = None):
        """
        Initialize Kafka producer.
        
        Args:
            bootstrap_servers: Kafka broker addresses
            topic: Kafka topic to publish to
        """
        self.bootstrap_servers = bootstrap_servers or os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.topic = topic or os.getenv('KAFKA_TOPIC_FRAUD_INPUT', 'transactions_input')
        self.producer = None
        
    def connect(self):
        """Connect to Kafka broker."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3
            )
            logger.info(f"Connected to Kafka at {self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    def send_transaction(self, transaction: Dict):
        """
        Send a single transaction to Kafka.
        
        Args:
            transaction: Transaction data dictionary
        """
        if self.producer is None:
            self.connect()
        
        try:
            self.producer.send(self.topic, value=transaction)
            self.producer.flush(timeout=1.0)
        except Exception as e:
            logger.error(f"Failed to send transaction: {e}")
            raise
    
    def send_batch(self, transactions: List[Dict], delay: float = 0.01):
        """
        Send a batch of transactions to Kafka.
        
        Args:
            transactions: List of transaction dictionaries
            delay: Delay between sends in seconds
        """
        if self.producer is None:
            self.connect()
        
        logger.info(f"Sending {len(transactions)} transactions to Kafka...")
        
        for i, transaction in enumerate(transactions):
            try:
                self.producer.send(self.topic, value=transaction)
                if i % 100 == 0:
                    self.producer.flush()
                time.sleep(delay)
            except Exception as e:
                logger.error(f"Failed to send transaction {i}: {e}")
        
        self.producer.flush()
        logger.info("Batch sent successfully")
    
    def stream_from_csv(self, csv_path: str, batch_size: int = 100, 
                        delay: float = 0.01, max_rows: int = None):
        """
        Stream transactions from CSV file to Kafka.
        
        Args:
            csv_path: Path to CSV file
            batch_size: Number of transactions per batch
            delay: Delay between batches in seconds
            max_rows: Maximum number of rows to stream
        """
        logger.info(f"Streaming transactions from {csv_path}...")
        
        df = pd.read_csv(csv_path)
        if max_rows:
            df = df.head(max_rows)
        
        transactions = df.to_dict('records')
        
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i + batch_size]
            self.send_batch(batch, delay)
            logger.info(f"Streamed {min(i + batch_size, len(transactions))}/{len(transactions)} transactions")
            time.sleep(delay)
        
        logger.info("Streaming completed")
    
    def close(self):
        """Close Kafka producer connection."""
        if self.producer:
            self.producer.close()
            logger.info("Kafka producer closed")


if __name__ == "__main__":
    producer = TransactionProducer()
    producer.connect()
    
    # Example: Stream sample transactions
    sample_transactions = [
        {
            "step": 1,
            "type": "PAYMENT",
            "amount": 9839.64,
            "nameOrig": "C1231006815",
            "oldbalanceOrg": 170136.0,
            "newbalanceOrig": 160296.36,
            "nameDest": "M1979787155",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0
        },
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
    ]
    
    producer.send_batch(sample_transactions)
    producer.close()
