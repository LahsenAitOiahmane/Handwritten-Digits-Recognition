import argparse
import logging
import os
from typing import Dict, Any, Tuple
import numpy as np

from Models.config.model_config import ModelConfig, SCENARIOS
from Models.config.logging_config import setup_logging
from Models.data.data_loader import DataLoader
from Models.data.data_augmentation import DataAugmentation
from Models.neural_network.model import NeuralNetwork
from Models.neural_network.optimizer import Optimizer
from Models.training.trainer import ModelTrainer
from Models.training.memory_monitor import MemoryMonitor
from Models.evaluation.evaluator import ModelEvaluator
from Models.evaluation.visualizer import VisualizationTools
from Models.utils.helpers import save_results, ensure_directory

def train_scenario(config: ModelConfig, data: Tuple[np.ndarray, ...]) -> Dict[str, Any]:
    """Train model with specific scenario configuration."""
    X_train, Y_train, X_dev, Y_dev = data
    
    # Initialize components
    trainer = ModelTrainer(config)
    optimizer = Optimizer(
        learning_rate=config.learning_rate,
        beta1=config.beta1,
        beta2=config.beta2
    )
    memory_monitor = MemoryMonitor()
    
    # Apply data augmentation if configured
    if config.use_augmentation:
        X_train_aug, Y_train_aug = DataAugmentation.augment_data(
            X_train, Y_train,
            rotation_range=config.rotation_range,
            noise_factor=config.noise_factor
        )
    else:
        X_train_aug, Y_train_aug = X_train, Y_train
    
    # Train model
    parameters, train_costs, train_acc, dev_acc = trainer.train(
        X_train_aug, Y_train_aug,
        X_dev, Y_dev
    )
    
    # Get memory stats
    memory_stats = memory_monitor.get_memory_stats()
    
    # Evaluate on test set
    test_accuracy, predictions, probabilities = trainer._compute_accuracy(X_dev, Y_dev)
    
    # Create visualization directory
    vis_path = ensure_directory(os.path.join(config.save_path, config.scenario_name, "visualizations"))
    
    # Generate visualizations
    VisualizationTools.plot_training_metrics(
        train_costs, train_acc, dev_acc,
        os.path.join(vis_path, "training_metrics.png")
    )
    
    VisualizationTools.plot_sample_predictions(
        X_dev, Y_dev, predictions, probabilities,
        os.path.join(vis_path, "sample_predictions.png")
    )
    
    # Save results
    results = {
        'parameters': parameters,
        'train_costs': train_costs,
        'train_accuracies': train_acc,
        'dev_accuracies': dev_acc,
        'test_accuracy': test_accuracy,
        'predictions': predictions,
        'memory_stats': memory_stats,
        'config': config.__dict__
    }
    
    save_results(results, os.path.join(config.save_path, config.scenario_name, "results.json"))
    
    return results

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Train neural network model')
    parser.add_argument('--scenario', type=str, choices=SCENARIOS.keys(),
                      help='Training scenario to run')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logging.info("Starting model pipeline...")
    
    # Create visualization directories
    vis_dir = ensure_directory('Results')
    comparison_dir = ensure_directory(os.path.join(vis_dir, 'comparison'))
    
    # Load data
    data_loader = DataLoader()
    data = data_loader.load_data(ModelConfig())
    
    if args.scenario:
        # Run specific scenario
        config = SCENARIOS[args.scenario]
        results = train_scenario(config, data)
        logging.info(f"Scenario {args.scenario} completed with "
                    f"test accuracy: {results['test_accuracy']:.2f}%")
    else:
        # Run all scenarios
        all_results = {}
        for scenario_name, config in SCENARIOS.items():
            logging.info(f"\nRunning scenario: {scenario_name}")
            results = train_scenario(config, data)
            all_results[scenario_name] = results
            logging.info(f"Scenario {scenario_name} completed with "
                        f"test accuracy: {results['test_accuracy']:.2f}%")
        
        # Generate comparison visualizations
        VisualizationTools.plot_training_history(
            {name: results['train_accuracies'] 
             for name, results in all_results.items()},
            os.path.join(comparison_dir, "training_history.png")
        )
        
        # Save comparison results
        save_results(all_results, os.path.join(comparison_dir, "scenario_comparison.json"))
    
    logging.info("Model pipeline completed successfully")

if __name__ == "__main__":
    main() 