import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from typing import Dict, List, Tuple
from ..config.model_config import ModelConfig

class ModelEvaluator:
    """Handles model evaluation and visualization."""
    
    @staticmethod
    def evaluate_model(model, X_test: np.ndarray, Y_test: np.ndarray) -> Tuple[float, np.ndarray]:
        """Evaluate model on test data."""
        A, _ = model.forward_propagation(X_test, training=False)
        predictions = np.argmax(A, axis=0)
        accuracy = np.mean(predictions == Y_test) * 100
        return accuracy, predictions
    
    @staticmethod
    def plot_confusion_matrix(Y_true: np.ndarray, Y_pred: np.ndarray, save_path: str):
        """Plot and save confusion matrix."""
        cm = np.zeros((10, 10), dtype=int)
        for i in range(len(Y_true)):
            cm[Y_true[i]][Y_pred[i]] += 1
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        classes = range(10)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Add text annotations
        thresh = cm.max() / 2
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    @staticmethod
    def plot_learning_curves(train_acc: List[float], dev_acc: List[float], save_path: str):
        """Plot learning curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_acc, label='Training Accuracy')
        plt.plot(dev_acc, label='Dev Accuracy')
        plt.title('Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        plt.close() 