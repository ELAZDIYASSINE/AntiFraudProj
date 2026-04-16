"""
PySpark Structured Streaming Pipeline for Real-time Fraud Detection
Processes transactions from Kafka and applies ML models.
"""

import os
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_json, struct, udf, lit
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import logging
from typing import Dict, Any
import joblib
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionPipeline:
    """PySpark Structured Streaming pipeline for fraud detection."""
    
    def __init__(self, app_name: str = "FraudDetection"):
        """
        Initialize streaming pipeline.
        
        Args:
            app_name: Spark application name
        """
        self.app_name = app_name
        self.spark = None
        self.model = None
        self.kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.input_topic = os.getenv('KAFKA_TOPIC_FRAUD_INPUT', 'transactions_input')
        self.output_topic = os.getenv('KAFKA_TOPIC_FRAUD_OUTPUT', 'fraud_predictions')
        
    def create_spark_session(self):
        """Create Spark session with Kafka support."""
        self.spark = SparkSession.builder \
            .appName(self.app_name) \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
            .config("spark.sql.streaming.checkpointLocation", "./checkpoints") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info("Spark session created")
        
    def define_schema(self) -> StructType:
        """Define schema for transaction data."""
        return StructType([
            StructField("step", IntegerType(), True),
            StructField("type", StringType(), True),
            StructField("amount", DoubleType(), True),
            StructField("nameOrig", StringType(), True),
            StructField("oldbalanceOrg", DoubleType(), True),
            StructField("newbalanceOrig", DoubleType(), True),
            StructField("nameDest", StringType(), True),
            StructField("oldbalanceDest", DoubleType(), True),
            StructField("newbalanceDest", DoubleType(), True),
            StructField("isFraud", IntegerType(), True),
            StructField("isFlaggedFraud", IntegerType(), True)
        ])
    
    def load_model(self, model_path: str):
        """
        Load trained ML model.
        
        Args:
            model_path: Path to saved model
        """
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_batch(self, df):
        """
        Preprocess streaming batch for prediction.
        
        Args:
            df: Spark DataFrame with raw transactions
            
        Returns:
            Preprocessed DataFrame
        """
        # Type encoding
        type_mapping = {
            'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 
            'PAYMENT': 3, 'TRANSFER': 4
        }
        
        # Add feature columns (simplified for streaming)
        df = df.withColumn("type_encoded", 
                          col("type").cast("string"))
        
        # Calculate derived features
        df = df.withColumn("amount_log", 
                          lit(1.0) * col("amount"))  # Placeholder
        
        df = df.withColumn("balance_diff_orig", 
                          col("oldbalanceOrg") - col("newbalanceOrig"))
        
        df = df.withColumn("balance_diff_dest", 
                          col("newbalanceDest") - col("oldbalanceDest"))
        
        return df
    
    def predict_fraud(self, features):
        """
        Predict fraud using loaded model.
        
        Args:
            features: Feature array
            
        Returns:
            Prediction (0 or 1)
        """
        if self.model is None:
            return 0
        
        try:
            prediction = self.model.predict([features])[0]
            return int(prediction)
        except:
            return 0
    
    def create_streaming_query(self):
        """Create and start streaming query."""
        # Read from Kafka
        df = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers) \
            .option("subscribe", self.input_topic) \
            .option("startingOffsets", "latest") \
            .load()
        
        # Parse JSON
        schema = self.define_schema()
        parsed_df = df.select(
            from_json(col("value").cast("string"), schema).alias("data")
        ).select("data.*")
        
        # Preprocess
        processed_df = self.preprocess_batch(parsed_df)
        
        # Add prediction (using UDF)
        predict_udf = udf(lambda x: 0, IntegerType())  # Placeholder UDF
        result_df = processed_df.withColumn("prediction", predict_udf(lit(1)))
        
        # Select output columns
        output_df = result_df.select(
            col("step"),
            col("type"),
            col("amount"),
            col("nameOrig"),
            col("nameDest"),
            col("prediction"),
            to_json(struct("*")).alias("value")
        )
        
        # Write to Kafka
        query = output_df.writeStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers) \
            .option("topic", self.output_topic) \
            .option("checkpointLocation", "./checkpoints") \
            .outputMode("append") \
            .start()
        
        logger.info("Streaming query started")
        return query
    
    def start(self, model_path: str = None):
        """
        Start the streaming pipeline.
        
        Args:
            model_path: Path to trained model
        """
        logger.info("Starting fraud detection pipeline...")
        
        # Create Spark session
        self.create_spark_session()
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
        
        # Create and start streaming query
        query = self.create_streaming_query()
        
        # Wait for termination
        try:
            query.awaitTermination()
        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the streaming pipeline."""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")


if __name__ == "__main__":
    pipeline = FraudDetectionPipeline()
    pipeline.start(model_path="./models/xgboost_best.json")
