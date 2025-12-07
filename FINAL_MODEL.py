import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
import logging
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import time
import argparse
import gc
import psutil
import torch
import json
from collections import Counter
from datetime import datetime
import urllib.request
from sklearn.metrics import confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class ModelConfig:
    """Configuration for model training and architecture."""
    
    # Architecture parameters
    input_size: int = 784  # 28x28 pixels
    output_size: int = 10  # 10 digits
    initial_hidden_size: int = 128
    final_hidden_size: int = 512
    growth_rate: int = 64
    growth_epochs: int = 10
    
    # Training parameters
    learning_rate: float = 0.001  # This replaces initial_learning_rate
    batch_size: int = 128
    batch_accumulation: int = 4
    max_epochs: int = 2
    warmup_epochs: int = 5
    decay_patience: int = 10
    learning_rate_decay: float = 0.95
    min_learning_rate: float = 1e-6
    target_accuracy: float = 98.0
    early_stopping_patience: int = 20
    
    # Optimizer parameters
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    gradient_clip_threshold: float = 5.0
    
    # Regularization
    dropout_rate: float = 0.2
    dev_set_size: float = 0.2
    
    # Data augmentation
    use_augmentation: bool = True
    rotation_range: float = 15.0
    noise_factor: float = 0.1
    
    # Memory management
    monitor_memory: bool = True
    max_memory_percent: float = 80.0
    gc_frequency: int = 5
    
    # Results management
    save_path: str = "Results"
    scenario_name: str = "default"

    def create_scenario_folders(self):
        """Create folders for scenario results."""
        scenario_path = os.path.join(self.save_path, self.scenario_name)
        os.makedirs(scenario_path, exist_ok=True)
        os.makedirs(os.path.join(scenario_path, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(scenario_path, "models"), exist_ok=True)
        return scenario_path

# Create global configuration instance
config = ModelConfig()

# Predefined scenarios
SCENARIOS = {
    "fast": ModelConfig(
        scenario_name="fast",
        batch_size=512,
        batch_accumulation=2,
        max_epochs=500,
        warmup_epochs=3,
        learning_rate=0.002,
        dropout_rate=0.1,
        use_augmentation=False
    ),
    "memory_optimized": ModelConfig(
        scenario_name="memory_optimized",
        batch_size=64,
        batch_accumulation=8,
        max_memory_percent=60.0,
        gc_frequency=3,
        initial_hidden_size=96,
        growth_rate=32
    ),
    "high_accuracy": ModelConfig(
        scenario_name="high_accuracy",
        initial_hidden_size=256,
        final_hidden_size=1024,
        growth_rate=128,
        dropout_rate=0.15,
        batch_accumulation=6,
        warmup_epochs=10,
        max_epochs=1500,
        use_augmentation=True,
        noise_factor=0.15
    ),
    "balanced": ModelConfig(
        scenario_name="balanced",
        batch_size=128,
        batch_accumulation=4,
        initial_hidden_size=128,
        final_hidden_size=512,
        warmup_epochs=5,
        dropout_rate=0.2
    )
}

class VisualizationTools:
    """Enhanced visualization tools."""
    
    @staticmethod
    def plot_scenario_results(train_accuracies: list, dev_accuracies: list,
                            training_time: float, memory_used: float,
                            save_path: str, scenario_name: str) -> None:
        """Creates comprehensive visualization for a training scenario."""
        plt.figure(figsize=(20, 10))
        
        # Learning curves
        plt.subplot(1, 2, 1)
        epochs = range(1, len(train_accuracies) + 1)
        plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
        plt.plot(epochs, dev_accuracies, 'r-', label='Dev Accuracy')
        plt.axhline(y=98, color='g', linestyle='--', label='Target Accuracy (98%)')
        plt.title(f'{scenario_name} - Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Resource usage
        plt.subplot(1, 2, 2)
        metrics = ['Training Time (s)', 'Memory Usage (%)']
        values = [training_time, memory_used]
        colors = ['#2ecc71', '#3498db']
        bars = plt.bar(metrics, values, color=colors)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')
        
        plt.title(f'{scenario_name} - Resource Usage')
        plt.ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{scenario_name}_results.png"), dpi=300, bbox_inches='tight')
        plt.close()

class DataPreprocessor:
    """Enhanced data preprocessing with memory optimization and augmentation."""
    
    @staticmethod
    def load_and_preprocess_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Loads and preprocesses training data with memory optimization."""
        try:
            logging.info("Loading training data...")
            chunks = pd.read_csv('Data/train.csv', chunksize=1000)
            data_list = []
            
            for chunk in chunks:
                chunk = chunk.astype(np.float32)  # Memory efficiency
                data_list.append(chunk.values)
                MemoryMonitor.check_memory_usage(config)
            
            data = np.vstack(data_list)
            del data_list
            gc.collect()
            
            m, n = data.shape
            indices = np.random.permutation(m)
            data = data[indices]
            
            dev_size = int(config.dev_set_size * m)
            
            # Split and normalize data
            data_dev = data[:dev_size].T
            Y_dev = data_dev[0].astype(np.int32)
            X_dev = data_dev[1:n] / 255.
            
            data_train = data[dev_size:].T
            Y_train = data_train[0].astype(np.int32)
            X_train = data_train[1:n] / 255.
            
            del data
            gc.collect()
            
            # Apply data augmentation if enabled
            if config.use_augmentation:
                X_train, Y_train = DataPreprocessor.augment_data(X_train, Y_train)
            
            logging.info(f"Train set: X={X_train.shape}, Y={Y_train.shape}")
            logging.info(f"Dev set: X={X_dev.shape}, Y={Y_dev.shape}")
            
            return X_train, Y_train, X_dev, Y_dev
            
        except Exception as e:
            logging.error(f"Data loading failed: {str(e)}")
            raise

    @staticmethod
    def augment_data(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies data augmentation techniques."""
        if not config.use_augmentation:
            return X, Y
            
        logging.info("Applying data augmentation...")
        X_augmented = []
        Y_augmented = []
        
        # Original data
        X_augmented.append(X)
        Y_augmented.append(Y)
        
        # Rotated versions
        if config.rotation_range > 0:
            for angle in [-config.rotation_range, config.rotation_range]:
                X_rot = DataPreprocessor.rotate_images(X, angle)
                X_augmented.append(X_rot)
                Y_augmented.append(Y)
        
        # Noisy versions
        if config.noise_factor > 0:
            X_noise = X + np.random.normal(0, config.noise_factor, X.shape)
            X_noise = np.clip(X_noise, 0., 1.)
            X_augmented.append(X_noise)
            Y_augmented.append(Y)
        
        return np.hstack(X_augmented), np.hstack(Y_augmented)

    @staticmethod
    def rotate_images(X: np.ndarray, angle: float) -> np.ndarray:
        """Rotates images by given angle."""
        from scipy.ndimage import rotate
        X_rotated = np.zeros_like(X)
        
        for i in range(X.shape[1]):
            img = X[:, i].reshape(28, 28)
            rotated = rotate(img, angle, reshape=False)
            X_rotated[:, i] = rotated.reshape(-1)
        
        return X_rotated

    @staticmethod
    def get_batch(X: np.ndarray, Y: np.ndarray, batch_size: int):
        """Memory-efficient batch generator."""
        m = X.shape[1]
        permutation = np.random.permutation(m)
        
        for i in range(0, m, batch_size):
            batch_idx = permutation[i:min(i + batch_size, m)]
            yield X[:, batch_idx], Y[batch_idx]

    @staticmethod
    def load_test_data():
        """Load and preprocess test data."""
        try:
            data = pd.read_csv('Data/test.csv')
            X_test = data['x_test'].reshape(data['x_test'].shape[0], -1).T / 255.0
            Y_test = data['y_test']
            logging.info(f"Test set: X={X_test.shape}, Y={Y_test.shape}")
            return X_test, Y_test
        except Exception as e:
            logging.error(f"Data loading failed: {str(e)}")
            raise

class NeuralNetwork:
    """Enhanced neural network with dynamic architecture."""
    
    @staticmethod
    def ReLU(Z: np.ndarray) -> np.ndarray:
        """Memory-efficient ReLU activation."""
        return np.maximum(0, Z)
    
    @staticmethod
    def ReLU_derivative(Z: np.ndarray) -> np.ndarray:
        """Memory-efficient ReLU derivative."""
        return (Z > 0).astype(np.float32)
    
    @staticmethod
    def softmax(Z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    @staticmethod
    def initialize_parameters(current_size: Optional[int] = None) -> dict:
        """Initialize network parameters."""
        hidden_size = current_size or config.initial_hidden_size
        
        parameters = {
            'W1': np.random.randn(hidden_size, config.input_size).astype(np.float32) * np.sqrt(1. / config.input_size),
            'b1': np.zeros((hidden_size, 1), dtype=np.float32),
            'W2': np.random.randn(hidden_size, hidden_size).astype(np.float32) * np.sqrt(1. / hidden_size),
            'b2': np.zeros((hidden_size, 1), dtype=np.float32),
            'W3': np.random.randn(config.output_size, hidden_size).astype(np.float32) * np.sqrt(1. / hidden_size),
            'b3': np.zeros((config.output_size, 1), dtype=np.float32)
        }
        return parameters

    @staticmethod
    def forward_propagation(parameters: dict, X: np.ndarray, training: bool = True) -> Tuple[np.ndarray, dict]:
        """Forward propagation with dropout."""
        W1, b1 = parameters['W1'], parameters['b1']
        W2, b2 = parameters['W2'], parameters['b2']
        W3, b3 = parameters['W3'], parameters['b3']
        
        Z1 = np.dot(W1, X) + b1
        A1 = NeuralNetwork.ReLU(Z1)
        
        if training:
            A1 *= np.random.binomial(1, 1-config.dropout_rate, A1.shape) / (1-config.dropout_rate)
        
        Z2 = np.dot(W2, A1) + b2
        A2 = NeuralNetwork.ReLU(Z2)
        
        if training:
            A2 *= np.random.binomial(1, 1-config.dropout_rate, A2.shape) / (1-config.dropout_rate)
        
        Z3 = np.dot(W3, A2) + b3
        A3 = NeuralNetwork.softmax(Z3)
        
        cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3}
        return A3, cache

    @staticmethod
    def backward_propagation(Z1: np.ndarray, A1: np.ndarray,
                           Z2: np.ndarray, A2: np.ndarray,
                           W2: np.ndarray, W3: np.ndarray,
                           Z3: np.ndarray, A3: np.ndarray,
                           X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Backward propagation with dropout."""
        m = X.shape[1]
        
        # Ensure Y is properly shaped (one-hot encoded)
        if Y.shape[0] != A3.shape[0]:
            Y = Y.T  # Transpose Y if shapes don't match
        
        dZ3 = A3 - Y
        dW3 = np.dot(dZ3, A2.T) / m
        db3 = np.sum(dZ3, axis=1, keepdims=True) / m
        
        dA2 = np.dot(W3.T, dZ3)
        dA2 *= np.random.binomial(1, 1-config.dropout_rate, dA2.shape) / (1-config.dropout_rate)
        dZ2 = dA2 * NeuralNetwork.ReLU_derivative(Z2)
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        
        dA1 = np.dot(W2.T, dZ2)
        dA1 *= np.random.binomial(1, 1-config.dropout_rate, dA1.shape) / (1-config.dropout_rate)
        dZ1 = dA1 * NeuralNetwork.ReLU_derivative(Z1)
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        
        return dW1, db1, dW2, db2, dW3, db3

    @staticmethod
    def grow_network(parameters: dict, v: dict, s: dict, epoch: int) -> Tuple[dict, dict, dict]:
        """Grow network if needed and adjust optimizer state."""
        if epoch % config.growth_epochs == 0 and epoch > 0:
            current_size = parameters['W1'].shape[0]
            target_size = min(current_size + config.growth_rate, config.final_hidden_size)
            
            if current_size < target_size:
                logging.info(f"Growing network from {current_size} to {target_size} units")
                new_parameters = NeuralNetwork.initialize_parameters(target_size)
                
                # Initialize new optimizer state dictionaries
                new_v = {key: np.zeros_like(value) for key, value in new_parameters.items()}
                new_s = {key: np.zeros_like(value) for key, value in new_parameters.items()}
                
                # Copy existing weights and optimizer states
                new_parameters['W1'][:current_size] = parameters['W1']
                new_parameters['b1'][:current_size] = parameters['b1']
                new_parameters['W2'][:current_size, :current_size] = parameters['W2']
                new_parameters['b2'][:current_size] = parameters['b2']
                new_parameters['W3'][:, :current_size] = parameters['W3']
                
                # Copy optimizer states
                for key in parameters.keys():
                    if key.startswith('W1'):
                        new_v[key][:current_size] = v[key]
                        new_s[key][:current_size] = s[key]
                    elif key.startswith('b1'):
                        new_v[key][:current_size] = v[key]
                        new_s[key][:current_size] = s[key]
                    elif key.startswith('W2'):
                        new_v[key][:current_size, :current_size] = v[key]
                        new_s[key][:current_size, :current_size] = s[key]
                    elif key.startswith('b2'):
                        new_v[key][:current_size] = v[key]
                        new_s[key][:current_size] = s[key]
                    elif key.startswith('W3'):
                        new_v[key][:, :current_size] = v[key]
                        new_s[key][:, :current_size] = s[key]
                    else:
                        new_v[key] = v[key]
                        new_s[key] = s[key]
                
                return new_parameters, new_v, new_s
        
        return parameters, v, s

class ModelTrainer:
    """Handles model training with comprehensive monitoring and optimization."""
    
    @staticmethod
    def relu(Z: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU) activation function."""
        return np.maximum(0, Z)
    
    @staticmethod
    def train_model(X_train: np.ndarray, Y_train: np.ndarray, 
                   X_dev: np.ndarray, Y_dev: np.ndarray,
                   X_test: np.ndarray, Y_test: np.ndarray,
                   save_path: str, initial_parameters: Optional[dict] = None) -> Tuple[dict, list, list]:
        """Trains the neural network with advanced features and monitoring."""
        
        logging.info("Initializing training...")
        parameters = initial_parameters or NeuralNetwork.initialize_parameters()
        
        # Initialize optimizer state
        v = {key: np.zeros_like(value) for key, value in parameters.items()}
        s = {key: np.zeros_like(value) for key, value in parameters.items()}
        t = 0  # Time step for Adam
        
        # Training history
        train_accuracies = []
        dev_accuracies = []
        best_accuracy = 0
        patience_counter = 0
        
        # Create one-hot encoded labels
        Y_train_one_hot = np.eye(config.output_size)[Y_train]
        
        for epoch in range(config.max_epochs):
            epoch_cost = 0
            batches = DataPreprocessor.get_batch(X_train, Y_train_one_hot, config.batch_size)
            
            # Training loop
            for batch_X, batch_Y in batches:
                # Forward propagation
                A3, cache = NeuralNetwork.forward_propagation(parameters, batch_X)
                
                # Backward propagation
                gradients = NeuralNetwork.backward_propagation(
                    cache['Z1'], cache['A1'],
                    cache['Z2'], cache['A2'],
                    parameters['W2'], parameters['W3'],
                    cache['Z3'], A3,
                    batch_X, batch_Y
                )
                
                # Update parameters with Adam optimization
                parameters, v, s = ModelTrainer.update_parameters_with_adam(
                    parameters, gradients, v, s, t,
                    config.batch_accumulation,
                    ModelTrainer.get_learning_rate(epoch)
                )
                t += 1
                
                # Memory management
                if epoch % config.gc_frequency == 0:
                    gc.collect()
                    MemoryMonitor.check_memory_usage(config)
            
            # Evaluate performance
            train_accuracy = ModelEvaluator.compute_accuracy(parameters, X_train, Y_train)
            dev_accuracy = ModelEvaluator.compute_accuracy(parameters, X_dev, Y_dev)
            
            train_accuracies.append(train_accuracy)
            dev_accuracies.append(dev_accuracy)
            
            # Log progress
            logging.info(
                f"Epoch {epoch + 1}/{config.max_epochs}: "
                f"train_acc={train_accuracy:.2f}%, "
                f"dev_acc={dev_accuracy:.2f}%"
            )
            
            # Grow network if needed
            parameters, v, s = NeuralNetwork.grow_network(parameters, v, s, epoch)
            
            # Save best model
            if dev_accuracy > best_accuracy:
                best_accuracy = dev_accuracy
                patience_counter = 0
                ModelTrainer.save_model(parameters, epoch, dev_accuracy)
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= config.early_stopping_patience:
                logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            # Check for target accuracy
            if train_accuracy >= config.target_accuracy:
                logging.info(f"Target accuracy {config.target_accuracy}% reached!")
                ModelTrainer.save_model(parameters, epoch, train_accuracy)
                break
        
        # After training, evaluate on test set
        test_accuracy, test_predictions = ModelTrainer.evaluate_model(parameters, X_test, Y_test, save_path)
        logging.info(f"Final test accuracy: {test_accuracy:.2f}%")
        
        return parameters, train_accuracies, dev_accuracies

    @staticmethod
    def update_parameters_with_adam(parameters: dict, gradients: tuple, v: dict, s: dict, 
                                  t: int, batch_accumulation: int, learning_rate: float) -> Tuple[dict, dict, dict]:
        """Updates parameters using Adam optimization with gradient accumulation."""
        v_corrected = {}
        s_corrected = {}
        t = t + 1
        
        for i, (key, value) in enumerate(parameters.items()):
            grad = gradients[i]
            
            # Gradient accumulation
            grad = grad / batch_accumulation
            
            # Adam optimization
            v[key] = config.beta1 * v[key] + (1 - config.beta1) * grad
            s[key] = config.beta2 * s[key] + (1 - config.beta2) * np.square(grad)
            
            v_corrected[key] = v[key] / (1 - np.power(config.beta1, t))
            s_corrected[key] = s[key] / (1 - np.power(config.beta2, t))
            
            # Update with gradient clipping
            update = learning_rate * v_corrected[key] / (np.sqrt(s_corrected[key]) + config.epsilon)
            update = np.clip(update, -config.gradient_clip_threshold, config.gradient_clip_threshold)
            parameters[key] -= update
        
        return parameters, v, s

    @staticmethod
    def get_learning_rate(epoch: int) -> float:
        """Dynamic learning rate scheduling with warmup."""
        if epoch < config.warmup_epochs:
            return config.learning_rate * (epoch + 1) / config.warmup_epochs
        
        decay = config.learning_rate_decay ** ((epoch - config.warmup_epochs) // config.decay_patience)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / config.max_epochs))
        
        return max(config.learning_rate * decay * cosine_decay, config.min_learning_rate)

    @staticmethod
    def save_model(parameters: dict, epoch: int, accuracy: float) -> None:
        """Saves model with metadata."""
        save_path = os.path.join(config.save_path, config.scenario_name)
        os.makedirs(save_path, exist_ok=True)
        
        model_data = {
            'parameters': parameters,
            'epoch': epoch,
            'accuracy': accuracy,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'config': config.__dict__
        }
        
        filename = f"model_acc{accuracy:.2f}_epoch{epoch}.npz"
        np.savez_compressed(os.path.join(save_path, filename), **model_data)
        logging.info(f"Model saved: {filename}")

    @staticmethod
    def load_model_parameters(filepath: str = 'Results/model_best.npz') -> dict:
        """Load model parameters from file."""
        try:
            data = np.load(filepath, allow_pickle=True)
            parameters = data['parameters'].item()
            logging.info(f"Model loaded from {filepath}")
            return parameters
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    @staticmethod
    def save_checkpoint(parameters: dict, epoch: int, metrics: dict, filepath: str) -> None:
        """Save training checkpoint for potential recovery."""
        checkpoint = {
            'parameters': parameters,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        np.savez_compressed(filepath, **checkpoint)
        logging.info(f"Checkpoint saved: {filepath}")

    @staticmethod
    def evaluate_model(parameters, X_test, Y_test, save_path=None):
        """Evaluate model on test data and save predictions."""
        # Forward pass to get predictions
        A1 = NeuralNetwork.ReLU(np.dot(parameters['W1'], X_test) + parameters['b1'])
        A2 = NeuralNetwork.ReLU(np.dot(parameters['W2'], A1) + parameters['b2'])
        Z3 = np.dot(parameters['W3'], A2) + parameters['b3']
        predictions = np.argmax(Z3, axis=0)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == Y_test) * 100
        
        if save_path:
            # Save test results
            test_results_path = os.path.join(save_path, "test_results.npz")
            np.savez(test_results_path,
                    predictions=predictions,
                    true_labels=Y_test)
            logging.info(f"Test results saved to {test_results_path}")
        
        return accuracy, predictions

class ModelEvaluator:
    """Enhanced evaluation and visualization with scenario comparison."""
    
    @staticmethod
    def compute_accuracy(parameters: dict, X: np.ndarray, Y: np.ndarray) -> float:
        """Computes prediction accuracy."""
        predictions = ModelEvaluator.predict(parameters, X)
        return 100 * np.mean(predictions == Y)

    @staticmethod
    def predict(parameters: dict, X: np.ndarray) -> np.ndarray:
        """Makes predictions for given input."""
        A3, _ = NeuralNetwork.forward_propagation(parameters, X, training=False)
        return np.argmax(A3, axis=0)

    @staticmethod
    def evaluate_and_visualize(parameters: dict, scenario_results: dict) -> dict:
        """Comprehensive evaluation and visualization of all scenarios."""
        results_path = os.path.join(config.save_path, "scenario_comparison")
        os.makedirs(results_path, exist_ok=True)
        
        scenario_metrics = {}
        
        for scenario_name, result in scenario_results.items():
            metrics = {
                'accuracy': result['accuracy'],
                'training_time': result['training_time'],
                'memory_usage': result['memory_usage'],
                'convergence_speed': result['convergence_epoch'],
                'final_model_size': ModelEvaluator.get_model_size(result['parameters'])
            }
            
            metrics['efficiency_score'] = ModelEvaluator.calculate_efficiency_score(metrics)
            scenario_metrics[scenario_name] = metrics
        
        # Generate visualizations
        ModelEvaluator.plot_scenario_comparison(scenario_metrics, results_path)
        ModelEvaluator.plot_learning_curves(scenario_results, results_path)
        ModelEvaluator.plot_resource_usage(scenario_metrics, results_path)
        
        # Save detailed report
        report_path = os.path.join(results_path, "scenario_comparison_report.json")
        with open(report_path, 'w') as f:
            json.dump(scenario_metrics, f, indent=4)
        
        return scenario_metrics

    @staticmethod
    def get_model_size(parameters: dict) -> float:
        """Calculate model size in KB."""
        return sum(arr.nbytes for arr in parameters.values()) / 1024

    @staticmethod
    def find_convergence_epoch(accuracies: list) -> int:
        """Find epoch where model converges."""
        window = 5
        for i in range(len(accuracies) - window):
            if np.std(accuracies[i:i+window]) < 0.1:
                return i
        return len(accuracies)

    @staticmethod
    def calculate_efficiency_score(metrics: dict) -> float:
        """Calculate overall efficiency score (0-100) based on multiple metrics."""
        weights = {
            'accuracy': 0.4,
            'training_time': 0.2,
            'memory_usage': 0.2,
            'convergence_speed': 0.1,
            'final_model_size': 0.1
        }
        
        # Normalize metrics to 0-100 scale
        normalized_metrics = {
            'accuracy': metrics['accuracy'],  # Already 0-100
            'training_time': 100 * (1 - metrics['training_time'] / 3600),  # Normalize to 1 hour
            'memory_usage': 100 * (1 - metrics['memory_usage'] / 100),
            'convergence_speed': 100 * (1 - metrics['convergence_speed'] / 100),
            'final_model_size': 100 * (1 - metrics['final_model_size'] / (10 * 1024))  # Normalize to 10MB
        }
        
        # Calculate weighted sum
        score = sum(weights[key] * normalized_metrics[key] for key in weights)
        return max(0, min(100, score))  # Clamp to 0-100 range

    @staticmethod
    def plot_scenario_comparison(metrics: dict, save_path: str) -> None:
        """Create radar chart comparing all scenarios."""
        categories = ['Accuracy', 'Training Speed', 'Memory Efficiency', 
                     'Convergence', 'Model Size']
        num_vars = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]  # Complete the circle
        
        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))
        
        # Plot data
        for scenario, metric in metrics.items():
            values = [
                metric['accuracy'],
                100 * (1 - metric['training_time'] / 3600),  # Normalize to 1 hour
                100 * (1 - metric['memory_usage'] / 100),
                100 * (1 - metric['convergence_speed'] / 100),
                100 * (1 - metric['final_model_size'] / (10 * 1024))  # Normalize to 10MB
            ]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=scenario)
            ax.fill(angles, values, alpha=0.1)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80, 100], 
                  ["20", "40", "60", "80", "100"], 
                  color="grey", size=7)
        plt.ylim(0, 100)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title("Scenario Comparison")
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(save_path, "scenario_comparison_radar.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_learning_curves(scenario_results: dict, save_path: str) -> None:
        """Plot learning curves for all scenarios."""
        plt.figure(figsize=(15, 8))
        
        for scenario_name, result in scenario_results.items():
            plt.plot(result['train_accuracies'], 
                    label=f"{scenario_name} (Train)", 
                    linestyle='--', alpha=0.7)
            plt.plot(result['dev_accuracies'], 
                    label=f"{scenario_name} (Dev)", 
                    linewidth=2)
        
        plt.title("Learning Curves Comparison")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(save_path, "learning_curves_comparison.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_resource_usage(metrics: dict, save_path: str) -> None:
        """Plot resource usage comparison for all scenarios."""
        scenarios = list(metrics.keys())
        x = np.arange(len(scenarios))
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Memory usage
        ax1.bar(x, [metrics[s]['memory_usage'] for s in scenarios], color='#3498db')
        ax1.set_title("Memory Usage (%)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenarios, rotation=45)
        
        # Training time
        ax2.bar(x, [metrics[s]['training_time'] for s in scenarios], color='#2ecc71')
        ax2.set_title("Training Time (s)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenarios, rotation=45)
        
        # Model size
        ax3.bar(x, [metrics[s]['final_model_size'] for s in scenarios], color='#e74c3c')
        ax3.set_title("Model Size (KB)")
        ax3.set_xticks(x)
        ax3.set_xticklabels(scenarios, rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "resource_usage_comparison.png"), dpi=300)
        plt.close()

    @staticmethod
    def plot_confusion_matrix(parameters: dict, X: np.ndarray, Y: np.ndarray, save_path: str) -> None:
        """Plot confusion matrix for model predictions."""
        predictions = ModelEvaluator.predict(parameters, X)
        cm = confusion_matrix(Y, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.savefig(os.path.join(save_path, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def analyze_errors(parameters: dict, X: np.ndarray, Y: np.ndarray) -> Dict:
        """Analyze prediction errors to identify patterns."""
        predictions = ModelEvaluator.predict(parameters, X)
        errors = predictions != Y
        
        error_analysis = {
            'total_errors': np.sum(errors),
            'error_rate': np.mean(errors) * 100,
            'error_indices': np.where(errors)[0],
            'confusion_pairs': Counter(zip(Y[errors], predictions[errors]))
        }
        
        # Find most common error patterns
        error_analysis['common_errors'] = {
            f"{true}->{pred}": count 
            for (true, pred), count in error_analysis['confusion_pairs'].most_common(5)
        }
        
        return error_analysis

class MemoryMonitor:
    """Enhanced memory monitoring and management."""
    
    @staticmethod
    def get_memory_usage() -> float:
        """Returns current memory usage as percentage."""
        process = psutil.Process(os.getpid())
        return process.memory_percent()
    
    @staticmethod
    def check_memory_usage(config: ModelConfig) -> None:
        """Checks if memory usage exceeds threshold."""
        if config.monitor_memory:
            usage = MemoryMonitor.get_memory_usage()
            if usage > config.max_memory_percent:
                logging.warning(f"High memory usage: {usage:.1f}%")
                gc.collect()

def train_with_different_scenarios(X_train, Y_train, X_dev, Y_dev, X_test, Y_test) -> dict:
    """Train models with different scenarios and compare results."""
    scenario_results = {}
    
    for scenario_name, config in SCENARIOS.items():
        try:
            logging.info(f"\nStarting scenario: {scenario_name}")
            logging.info("=" * 50)
            
            # Create scenario directory
            scenario_path = os.path.join(config.save_path, scenario_name)
            os.makedirs(os.path.join(scenario_path, "visualizations"), exist_ok=True)
            
            # Train model and track metrics
            start_time = time.time()
            parameters, train_accuracies, dev_accuracies = ModelTrainer.train_model(
                X_train, Y_train, X_dev, Y_dev, X_test, Y_test, scenario_path, **config.__dict__
            )
            training_time = time.time() - start_time
            
            # Calculate final metrics
            final_accuracy = dev_accuracies[-1]
            memory_used = psutil.Process().memory_percent()
            convergence_epoch = ModelEvaluator.find_convergence_epoch(dev_accuracies)
            
            # Get test predictions
            test_accuracy, test_predictions = ModelTrainer.evaluate_model(
                parameters, X_test, Y_test, scenario_path
            )
            
            # Store results
            scenario_results[scenario_name] = {
                'parameters': parameters,
                'accuracy': final_accuracy,
                'training_time': training_time,
                'memory_usage': memory_used,
                'convergence_epoch': convergence_epoch,
                'train_accuracies': train_accuracies,
                'dev_accuracies': dev_accuracies,
                'predictions': test_predictions,
                'true_labels': Y_test
            }
            
            # Rest of the visualization code...
            
        except Exception as e:
            logging.error(f"Scenario {scenario_name} failed: {str(e)}")
            scenario_results[scenario_name] = None
            continue
        
        finally:
            # Clean up memory
            gc.collect()
    
    # Generate final comparison
    if any(scenario_results.values()):
        comparison_path = os.path.join("Results", "comparison")
        os.makedirs(comparison_path, exist_ok=True)
        ModelEvaluator.evaluate_and_visualize(None, scenario_results)
        
        # Rank scenarios
        scenario_rankings = rank_scenarios(scenario_results)
        logging.info("\nScenario Rankings:")
        for rank, (name, score) in enumerate(scenario_rankings, 1):
            logging.info(f"{rank}. {name}: {score:.2f}")
    
    return scenario_results

def rank_scenarios(scenario_results: dict) -> List[Tuple[str, float]]:
    """
    Ranks scenarios based on multiple criteria.
    Returns list of (scenario_name, score) tuples sorted by score.
    """
    rankings = []
    
    for name, results in scenario_results.items():
        if results is None:
            continue
            
        # Calculate composite score
        accuracy_score = results['accuracy']
        time_score = 100 * (1 - results['training_time'] / 3600)  # Normalize to 1 hour
        memory_score = 100 * (1 - results['memory_usage'] / 100)
        convergence_score = 100 * (1 - results['convergence_epoch'] / 100)
        
        # Weighted sum of scores
        total_score = (
            0.4 * accuracy_score +    # Accuracy is most important
            0.25 * time_score +       # Training time is second
            0.2 * memory_score +      # Memory efficiency is third
            0.15 * convergence_score  # Convergence speed is fourth
        )
        
        rankings.append((name, total_score))
    
    return sorted(rankings, key=lambda x: x[1], reverse=True)

def main():
    """Main execution function with enhanced error handling and logging."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Train and evaluate MNIST classifier.')
        parser.add_argument('--scenario', action='store_true', help='Run all training scenarios')
        parser.add_argument('--evaluate', action='store_true', help='Evaluate existing model')
        parser.add_argument('--continue_training', action='store_true', help='Continue training from checkpoint')
        args = parser.parse_args()
        
        # Regular training pipeline
        logging.info("Starting model pipeline...")
        X_train, Y_train, X_dev, Y_dev = DataPreprocessor.load_and_preprocess_data()
        X_test, Y_test = DataPreprocessor.load_test_data()
        
        if args.scenario:
            scenario_results = train_with_different_scenarios(X_train, Y_train, X_dev, Y_dev, X_test, Y_test)
            return
        
        if args.evaluate or (args.continue_training and os.path.exists('Results/model_best.npz')):
            parameters = ModelTrainer.load_model_parameters()
            
            if args.continue_training:
                parameters, train_accuracies, dev_accuracies = ModelTrainer.train_model(
                    X_train, Y_train, X_dev, Y_dev, X_test, Y_test, "Results", 
                    initial_parameters=parameters
                )
        else:
            parameters, train_accuracies, dev_accuracies = ModelTrainer.train_model(
                X_train, Y_train, X_dev, Y_dev, X_test, Y_test, "Results"
            )
        
        # Final evaluation
        final_metrics = ModelEvaluator.evaluate_and_visualize(
            parameters, 
            {'default': {
                'parameters': parameters,
                'train_accuracies': train_accuracies,
                'dev_accuracies': dev_accuracies,
                'accuracy': dev_accuracies[-1],
                'training_time': time.time() - start_time,
                'memory_usage': MemoryMonitor.get_memory_usage(),
                'convergence_epoch': ModelEvaluator.find_convergence_epoch(dev_accuracies)
            }}
        )
        
        logging.info("Model pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    start_time = time.time()
    main()

