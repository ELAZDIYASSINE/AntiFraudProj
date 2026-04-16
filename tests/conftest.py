"""
Pytest configuration and fixtures.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="session")
def sample_data():
    """Provide sample data for testing."""
    import pandas as pd
    import numpy as np
    
    return pd.DataFrame({
        'step': range(100),
        'type': ['PAYMENT'] * 50 + ['TRANSFER'] * 50,
        'amount': np.random.uniform(100, 10000, 100),
        'nameOrig': [f'C{i}' for i in range(100)],
        'oldbalanceOrg': np.random.uniform(0, 50000, 100),
        'newbalanceOrig': np.random.uniform(0, 50000, 100),
        'nameDest': [f'M{i}' for i in range(100)],
        'oldbalanceDest': np.random.uniform(0, 50000, 100),
        'newbalanceDest': np.random.uniform(0, 50000, 100),
        'isFraud': [0] * 95 + [1] * 5,
        'isFlaggedFraud': [0] * 100
    })
