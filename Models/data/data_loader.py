import numpy as np
import pandas as pd
import logging
from typing import Tuple
from ..config.model_config import ModelConfig

class DataLoader:
    """Handles data loading and preprocessing."""
    
    @staticmethod
    def load_data(config: ModelConfig) -> Tuple[np.ndarray, ...]:
        """Load and preprocess training data."""
        try:
            logging.info("Loading training data...")
            data = pd.read_csv('Data/train.csv')
            X, y = DataLoader._preprocess_data(data)
            
            # Split into train/dev sets
            dev_size = int(config.dev_set_size * len(y))
            X_train, X_dev = X[:, dev_size:], X[:, :dev_size]
            y_train, y_dev = y[dev_size:], y[:dev_size]
            
            logging.info(f"Train set: X={X_train.shape}, Y={y_train.shape}")
            logging.info(f"Dev set: X={X_dev.shape}, Y={y_dev.shape}")
            
            return X_train, y_train, X_dev, y_dev
            
        except Exception as e:
            logging.error(f"Data loading failed: {str(e)}")
            raise

    @staticmethod
    def _preprocess_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess raw data."""
        X = data.iloc[:, 1:].values.T / 255.0  # Normalize pixel values
        y = data.iloc[:, 0].values
        return X, y 