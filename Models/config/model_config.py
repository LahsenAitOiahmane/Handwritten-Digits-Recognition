from dataclasses import dataclass
import os

@dataclass
class ModelConfig:
    """Configuration for model training and architecture."""
    
    # Architecture parameters
    input_size: int = 784
    output_size: int = 10
    initial_hidden_size: int = 128
    final_hidden_size: int = 512
    growth_rate: int = 64
    growth_epochs: int = 10
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 128
    batch_accumulation: int = 4
    max_epochs: int = 2
    warmup_epochs: int = 5
    early_stopping_patience: int = 20
    
    # Optimizer parameters
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    
    # Regularization
    dropout_rate: float = 0.2
    dev_set_size: float = 0.2
    
    # Data augmentation
    use_augmentation: bool = True
    rotation_range: float = 15.0
    noise_factor: float = 0.1
    
    # Results management
    save_path: str = "Results"
    scenario_name: str = "default"

    def create_scenario_folders(self):
        """Create folders for scenario results."""
        scenario_path = os.path.join(self.save_path, self.scenario_name)
        os.makedirs(scenario_path, exist_ok=True)
        os.makedirs(os.path.join(scenario_path, "visualizations"), exist_ok=True)
        return scenario_path

# Predefined training scenarios
SCENARIOS = {
    "fast": ModelConfig(
        scenario_name="fast",
        batch_size=512,
        max_epochs=21,
        learning_rate=0.002,
        use_augmentation=False
    ),
    "memory_optimized": ModelConfig(
        scenario_name="memory_optimized",
        learning_rate=0.0005,  # Lower learning rate
        max_epochs=21,          # Fewer epochs
        batch_size=32,         # Smaller batch size
        use_augmentation=True, # Use data augmentation
        initial_hidden_size=256,  # Smaller network
        final_hidden_size=128
    )
    # Add other scenarios as needed
} 