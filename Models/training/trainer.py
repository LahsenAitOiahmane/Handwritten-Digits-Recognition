import numpy as np
import logging
import time
import os
from typing import Tuple, List, Dict
from ..neural_network.model import NeuralNetwork
from ..config.model_config import ModelConfig
from .memory_monitor import MemoryMonitor
from ..utils.helpers import ensure_directory

class ModelTrainer:
    """Handles model training and checkpointing."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = NeuralNetwork(config)
        self.memory_monitor = MemoryMonitor()
        
        # Create directories for checkpoints and results
        self.checkpoint_dir = ensure_directory(os.path.join('Results', 'checkpoints'))
        self.results_dir = ensure_directory(os.path.join('Results', 'training'))
    
    def train(self, X_train: np.ndarray, Y_train: np.ndarray, 
             X_dev: np.ndarray, Y_dev: np.ndarray) -> Tuple[Dict, List[float], List[float], List[float]]:
        """Train the model."""
        logging.info("Starting training...")
        train_costs = []
        train_accuracies = []
        dev_accuracies = []
        best_accuracy = 0
        patience_counter = 0
        
        # Ensure Y_train is properly shaped
        Y_train = Y_train.reshape(-1, 1) if len(Y_train.shape) == 1 else Y_train
        
        for epoch in range(self.config.max_epochs):
            # Forward propagation
            A, cache = self.model.forward_propagation(X_train)
            
            # Compute cost
            m = Y_train.shape[0]
            Y_hot = self.model._one_hot_encode(Y_train)
            cost = -np.sum(Y_hot.T * np.log(A + 1e-15)) / m
            train_costs.append(cost)
            
            # Backward propagation
            self.model.backward_propagation(A, Y_train, cache)
            
            # Update parameters
            self._update_parameters()
            
            # Calculate accuracies
            train_accuracy, _, _ = self._compute_accuracy(X_train, Y_train)
            dev_accuracy, _, _ = self._compute_accuracy(X_dev, Y_dev)
            
            train_accuracies.append(train_accuracy)
            dev_accuracies.append(dev_accuracy)
            
            # Log progress
            logging.info(f"Epoch {epoch + 1}/{self.config.max_epochs}: "
                        f"cost={cost:.4f}, "
                        f"train_acc={train_accuracy:.2f}%, "
                        f"dev_acc={dev_accuracy:.2f}%")
            
            # Save checkpoint if improved
            if dev_accuracy > best_accuracy:
                best_accuracy = dev_accuracy
                self._save_checkpoint(epoch, dev_accuracy)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= self.config.early_stopping_patience:
                logging.info("Early stopping triggered")
                break
            
            # Log memory usage every few epochs
            if epoch % 10 == 0:
                self.memory_monitor.log_memory_usage(epoch)
        
        return self.model.get_parameters(), train_costs, train_accuracies, dev_accuracies
    
    def _update_parameters(self):
        """Update model parameters using optimizer."""
        for layer in self.model.layers:
            layer.update_parameters(self.config.learning_rate)
    
    def _compute_accuracy(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute prediction accuracy and return probabilities."""
        # Forward pass
        probabilities, _ = self.model.forward_propagation(X, training=False)
        predictions = np.argmax(probabilities, axis=0)
        
        if len(Y.shape) > 1:
            Y = Y.ravel()
        
        accuracy = np.mean(predictions == Y) * 100
        return accuracy, predictions, probabilities
    
    def _save_checkpoint(self, epoch: int, accuracy: float):
        """Save model checkpoint."""
        # Create scenario-specific directory
        scenario_dir = ensure_directory(os.path.join(
            self.checkpoint_dir,
            self.config.scenario_name
        ))
        
        # Create checkpoint filename
        checkpoint_path = os.path.join(
            scenario_dir,
            f"model_acc{accuracy:.2f}_epoch{epoch}.npz"
        )
        
        # Save checkpoint
        np.savez(checkpoint_path, **self.model.get_parameters())
        logging.info(f"Model checkpoint saved: {checkpoint_path}") 