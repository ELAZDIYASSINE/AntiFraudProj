"""Configuration for preprocessing pipeline."""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings."""
    
    # Data paths
    DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data')
    RAW_DATA_FILE = os.path.join(DATA_PATH, 'PS_20174392719_1491204439457_log.csv')
    PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed')
    
    # Preprocessing settings
    SAMPLE_SIZE = int(os.getenv('SAMPLE_SIZE', 100000))  # Use 100K for faster training
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    
    # Feature engineering
    TIME_WINDOW_HOURS = 24
    AMOUNT_THRESHOLD = 1000000
    
    @classmethod
    def ensure_directories(cls):
        """Ensure required directories exist."""
        os.makedirs(cls.PROCESSED_DATA_PATH, exist_ok=True)
