import os
import numpy as np
import logging
import json
from Models.FINAL_MODEL import (
    ModelEvaluator, 
    VisualizationTools, 
    rank_scenarios, 
    SCENARIOS
)
import matplotlib.pyplot as plt

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def load_scenario_data(scenario_name):
    """Load training data for a scenario from its results directory."""
    scenario_path = os.path.join("Results", scenario_name)
    
    try:
        # Find the latest model file
        model_files = [f for f in os.listdir(scenario_path) if f.endswith('.npz')]
        if not model_files:
            return None
        
        # Sort by accuracy and epoch number
        latest_model = sorted(model_files, key=lambda x: (
            float(x.split('_acc')[1].split('_')[0]), 
            int(x.split('epoch')[1].split('.')[0])
        ))[-1]
        
        # Load model data with allow_pickle=True
        model_data = np.load(os.path.join(scenario_path, latest_model), allow_pickle=True)
        
        # Extract metrics from filename
        accuracy = float(latest_model.split('_acc')[1].split('_')[0])
        epoch = int(latest_model.split('epoch')[1].split('.')[0])
        
        # Load training history if available
        history_path = os.path.join(scenario_path, "training_history.npz")
        training_history = {}
        if os.path.exists(history_path):
            history_data = np.load(history_path, allow_pickle=True)
            training_history = {
                'train_accuracies': history_data['train_accuracies'],
                'dev_accuracies': history_data['dev_accuracies']
            }
        
        # Load test results if available
        test_results_path = os.path.join(scenario_path, "test_results.npz")
        test_results = {}
        if os.path.exists(test_results_path):
            test_data = np.load(test_results_path, allow_pickle=True)
            test_results = {
                'predictions': test_data['predictions'],
                'true_labels': test_data['true_labels']
            }
            logging.info(f"Loaded test results for {scenario_name}")
        else:
            logging.warning(f"No test results found for {scenario_name}")
        
        return {
            'parameters': dict(model_data),
            'accuracy': accuracy,
            'epoch': epoch,
            'model_file': latest_model,
            'train_accuracies': training_history.get('train_accuracies', []),
            'dev_accuracies': training_history.get('dev_accuracies', []),
            'predictions': test_results.get('predictions', []),
            'true_labels': test_results.get('true_labels', [])
        }
        
    except Exception as e:
        logging.error(f"Error loading data for {scenario_name}: {str(e)}")
        return None

def generate_additional_visualizations(scenario_results, comparison_path):
    """Generate additional visualizations for each scenario."""
    for scenario_name, result in scenario_results.items():
        scenario_path = os.path.join(comparison_path, scenario_name)
        os.makedirs(scenario_path, exist_ok=True)
        
        # Plot individual scenario learning curves and resource usage
        VisualizationTools.plot_scenario_results(
            train_accuracies=result['train_accuracies'],
            dev_accuracies=result['dev_accuracies'],
            training_time=result['training_time'],
            memory_used=result['memory_usage'],
            save_path=scenario_path,
            scenario_name=scenario_name
        )
        
        # Generate confusion matrix if predictions are available
        if 'predictions' in result and 'true_labels' in result:
            try:
                confusion_save_path = os.path.join(scenario_path, f"{scenario_name}_confusion_matrix.png")
                
                # Debug prints
                print(f"\nDebug info for {scenario_name}:")
                print(f"Predictions shape: {np.array(result['predictions']).shape}")
                print(f"True labels shape: {np.array(result['true_labels']).shape}")
                print(f"Sample of predictions: {np.array(result['predictions'])[:10]}")
                print(f"Sample of true labels: {np.array(result['true_labels'])[:10]}")
                
                # Create confusion matrix manually
                cm = np.zeros((10, 10), dtype=int)
                true_labels = np.array(result['true_labels'])
                predictions = np.array(result['predictions'])
                
                if len(true_labels) == 0 or len(predictions) == 0:
                    logging.warning(f"Empty predictions or true labels for {scenario_name}")
                    continue
                
                # Debug print before creating confusion matrix
                print(f"Number of samples: {len(true_labels)}")
                print(f"Unique values in predictions: {np.unique(predictions)}")
                print(f"Unique values in true_labels: {np.unique(true_labels)}")
                
                for i in range(len(true_labels)):
                    try:
                        cm[true_labels[i]][predictions[i]] += 1
                    except IndexError as e:
                        print(f"Error at index {i}: true_label={true_labels[i]}, prediction={predictions[i]}")
                        raise e
                
                # Debug print confusion matrix
                print(f"Confusion matrix sum: {np.sum(cm)}")
                print(f"Confusion matrix:\n{cm}")
                
                # Plot confusion matrix
                plt.figure(figsize=(10, 8))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title(f'Confusion Matrix - {scenario_name}')
                plt.colorbar()
                
                # Add labels
                classes = range(10)
                tick_marks = np.arange(len(classes))
                plt.xticks(tick_marks, classes)
                plt.yticks(tick_marks, classes)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                
                # Add numbers to cells
                thresh = cm.max() / 2.
                for i, j in np.ndindex(cm.shape):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
                
                plt.tight_layout()
                plt.savefig(confusion_save_path)
                plt.close()
                
                logging.info(f"Successfully plotted confusion matrix for {scenario_name}")
                
            except Exception as e:
                logging.error(f"Error plotting confusion matrix for {scenario_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # Plot learning curve for individual scenario
        if result['train_accuracies'] and result['dev_accuracies']:
            plt.figure(figsize=(10, 6))
            plt.plot(result['train_accuracies'], label='Training Accuracy', linestyle='--')
            plt.plot(result['dev_accuracies'], label='Validation Accuracy')
            plt.title(f'{scenario_name} Learning Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(scenario_path, f"{scenario_name}_learning_curve.png"))
            plt.close()

        # Plot resource usage for individual scenario
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.bar(['Memory Usage'], [result['memory_usage']], color='blue')
        plt.title('Memory Usage (%)')
        
        plt.subplot(1, 2, 2)
        plt.bar(['Training Time'], [result['training_time']], color='green')
        plt.title('Training Time (s)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(scenario_path, f"{scenario_name}_resource_usage.png"))
        plt.close()

def resume_scenario_comparison():
    """Resume and complete scenario comparison from existing results."""
    logging.basicConfig(level=logging.INFO)
    
    comparison_path = os.path.join("Results", "comparison")
    os.makedirs(comparison_path, exist_ok=True)
    
    scenario_results = {}
    json_safe_results = {}
    
    for scenario_name in SCENARIOS.keys():
        logging.info(f"\nProcessing scenario: {scenario_name}")
        
        try:
            scenario_data = load_scenario_data(scenario_name)
            if scenario_data:
                logging.info(f"Found existing results for {scenario_name}")
                logging.info(f"Best accuracy: {scenario_data['accuracy']:.2f}%")
                logging.info(f"Last epoch: {scenario_data['epoch']}")
                
                model_size = ModelEvaluator.get_model_size(scenario_data['parameters'])
                
                scenario_results[scenario_name] = {
                    'parameters': scenario_data['parameters'],
                    'accuracy': scenario_data['accuracy'],
                    'final_model_size': model_size,
                    'convergence_epoch': scenario_data['epoch'],
                    'training_time': scenario_data['epoch'] * 60,
                    'memory_usage': model_size / (10 * 1024) * 100,
                    'train_accuracies': scenario_data.get('train_accuracies', []),
                    'dev_accuracies': scenario_data.get('dev_accuracies', []),
                    'predictions': scenario_data.get('predictions', []),
                    'true_labels': scenario_data.get('true_labels', [])
                }
                
                json_safe_results[scenario_name] = scenario_results[scenario_name]
        except Exception as e:
            logging.error(f"Error processing {scenario_name}: {str(e)}")
            continue
    
    if scenario_results:
        try:
            # Generate all visualizations
            ModelEvaluator.evaluate_and_visualize(None, scenario_results)
            generate_additional_visualizations(scenario_results, comparison_path)
            
            # Create summary plots
            plt.figure(figsize=(15, 10))
            
            # Accuracy comparison
            plt.subplot(2, 2, 1)
            accuracies = [result['accuracy'] for result in scenario_results.values()]
            scenarios = list(scenario_results.keys())
            plt.bar(scenarios, accuracies)
            plt.title('Final Accuracy Comparison')
            plt.xticks(rotation=45)
            
            # Convergence speed
            plt.subplot(2, 2, 2)
            convergence = [result['convergence_epoch'] for result in scenario_results.values()]
            plt.bar(scenarios, convergence)
            plt.title('Convergence Speed (epochs)')
            plt.xticks(rotation=45)
            
            # Memory usage
            plt.subplot(2, 2, 3)
            memory = [result['memory_usage'] for result in scenario_results.values()]
            plt.bar(scenarios, memory)
            plt.title('Memory Usage (%)')
            plt.xticks(rotation=45)
            
            # Training time
            plt.subplot(2, 2, 4)
            time = [result['training_time'] for result in scenario_results.values()]
            plt.bar(scenarios, time)
            plt.title('Training Time (s)')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_path, 'summary_metrics.png'))
            plt.close()
            
            # Rankings and reports
            scenario_rankings = rank_scenarios(scenario_results)
            logging.info("\nScenario Rankings:")
            for rank, (name, score) in enumerate(scenario_rankings, 1):
                logging.info(f"{rank}. {name}: {score:.2f}")
            
            # Save reports
            report_path = os.path.join(comparison_path, "final_comparison_report.json")
            with open(report_path, 'w') as f:
                json.dump(json_safe_results, f, indent=4, cls=NumpyEncoder)
            
            rankings_path = os.path.join(comparison_path, "scenario_rankings.json")
            rankings_dict = {name: float(score) for name, score in scenario_rankings}
            with open(rankings_path, 'w') as f:
                json.dump(rankings_dict, f, indent=4)
            
            logging.info(f"\nAll visualizations and reports saved in {comparison_path}")
            
        except Exception as e:
            logging.error(f"Error during visualization and reporting: {str(e)}")
            raise
    else:
        logging.error("No valid scenario results found to compare")

def fix_plot_resource_usage():
    """Update the plot_resource_usage method to fix matplotlib warnings."""
    @staticmethod
    def plot_resource_usage(metrics: dict, save_path: str) -> None:
        """Plot resource usage comparison for all scenarios."""
        scenarios = list(metrics.keys())
        x = np.arange(len(scenarios))
        
        memory_usage = [metrics[s]['memory_usage'] for s in scenarios]
        training_time = [metrics[s]['training_time'] for s in scenarios]
        model_size = [metrics[s]['final_model_size'] for s in scenarios]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Memory usage
        ax1.bar(x, memory_usage, color='#3498db')
        ax1.set_title("Memory Usage (%)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenarios, rotation=45)
        
        # Training time
        ax2.bar(x, training_time, color='#2ecc71')
        ax2.set_title("Training Time (s)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenarios, rotation=45)
        
        # Model size
        ax3.bar(x, model_size, color='#e74c3c')
        ax3.set_title("Model Size (KB)")
        ax3.set_xticks(x)
        ax3.set_xticklabels(scenarios, rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "resource_usage_comparison.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Monkey patch the method
    ModelEvaluator.plot_resource_usage = plot_resource_usage

if __name__ == "__main__":
    # Fix matplotlib warnings
    fix_plot_resource_usage()
    resume_scenario_comparison() 