import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, List

class VisualizationTools:
    """Tools for creating visualizations of model performance."""
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], save_path: str):
        """Plot training metrics history."""
        plt.figure(figsize=(12, 6))
        
        # Plot training curves
        for metric, values in history.items():
            plt.plot(values, label=metric)
        
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(save_path)
        plt.close()
    
    @staticmethod
    def plot_sample_predictions(X: np.ndarray, y_true: np.ndarray, 
                              y_pred: np.ndarray, probabilities: np.ndarray,
                              save_path: str, n_samples: int = 5):
        """
        Plot sample predictions with their confidence scores.
        
        Args:
            X: Input images
            y_true: True labels
            y_pred: Predicted labels
            probabilities: Model's confidence scores for each class (shape: [n_classes, n_samples])
            save_path: Where to save the visualization
            n_samples: Number of samples to display
        """
        # Reshape features to images
        img_size = int(np.sqrt(X.shape[0]))
        
        # Select random samples
        indices = np.random.choice(X.shape[1], n_samples, replace=False)
        
        # Create figure with 2 rows: images and confidence plots
        fig, axes = plt.subplots(2, n_samples, figsize=(20, 8))
        
        for i, idx in enumerate(indices):
            # Top row: Display image
            img = X[:, idx].reshape(img_size, img_size)
            if img.max() > 1:
                img = img / 255.0
            axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[0, i].axis('off')
            
            # Add title with true and predicted labels
            color = 'green' if y_true[idx] == y_pred[idx] else 'red'
            axes[0, i].set_title(f'True: {y_true[idx]}\nPred: {y_pred[idx]}',
                               color=color)
            
            # Bottom row: Plot confidence scores
            confidence_scores = probabilities[:, idx]
            axes[1, i].bar(range(10), confidence_scores, color='skyblue')
            axes[1, i].set_ylim(0, 1)
            axes[1, i].set_xticks(range(10))
            axes[1, i].set_xlabel('Digit Class')
            if i == 0:  # Only add y-label for first plot
                axes[1, i].set_ylabel('Confidence')
            
            # Highlight predicted and true classes
            pred_color = 'green' if y_true[idx] == y_pred[idx] else 'red'
            axes[1, i].bar(y_pred[idx], confidence_scores[y_pred[idx]], 
                         color=pred_color)
            if y_true[idx] != y_pred[idx]:
                axes[1, i].bar(y_true[idx], confidence_scores[y_true[idx]], 
                             color='blue', alpha=0.5)
        
        plt.suptitle('Sample Predictions with Confidence Scores\n' + 
                    'Top: Images, Bottom: Class Probabilities\n' +
                    'Green: Correct Prediction, Red: Wrong Prediction, Blue: True Class (if different)',
                    y=1.05)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close() 
        
    @staticmethod
    def plot_training_metrics(train_costs: List[float], train_accuracies: List[float], 
                            dev_accuracies: List[float], save_path: str):
        """
        Plot training metrics (cost and accuracies) over time.
        
        Args:
            train_costs: List of training costs per epoch
            train_accuracies: List of training accuracies per epoch
            dev_accuracies: List of development set accuracies per epoch
            save_path: Where to save the visualization
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot training cost
        ax1.plot(train_costs, 'b-', label='Training Cost')
        ax1.set_title('Training Cost over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cost')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot accuracies
        ax2.plot(train_accuracies, 'g-', label='Training Accuracy')
        ax2.plot(dev_accuracies, 'r-', label='Dev Accuracy')
        ax2.set_title('Model Accuracy over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()