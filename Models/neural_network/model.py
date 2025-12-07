import numpy as np
from typing import Dict, List, Tuple
from .layers import Layer, Activations
from ..config.model_config import ModelConfig

class NeuralNetwork:
    """Neural network model implementation."""
    
    def __init__(self, config: ModelConfig):
        """Initialize neural network."""
        self.config = config
        self.layers: List[Layer] = []
        self._build_architecture()
    
    def _build_architecture(self):
        """Build neural network architecture."""
        layer_sizes = [
            self.config.input_size,
            self.config.initial_hidden_size,
            self.config.final_hidden_size,
            self.config.output_size
        ]
        
        # Create layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))
    
    def forward_propagation(self, X: np.ndarray, training: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """Forward propagation through the network."""
        A = X
        caches = []
        
        # Forward through all layers
        for layer in self.layers[:-1]:
            A, cache = layer.forward(A, training)
            caches.append(cache)
        
        # Output layer (with softmax)
        final_layer = self.layers[-1]
        Z = np.dot(final_layer.W, A) + final_layer.b
        A = Activations.softmax(Z)
        
        if training:
            final_cache = {
                'A_prev': caches[-1]['A'],  # Use the output of previous layer
                'Z': Z,
                'A': A
            }
            caches.append(final_cache)
        
        return A, caches
    
    def backward_propagation(self, AL: np.ndarray, Y: np.ndarray, caches: List[Dict]):
        """Backward propagation through the network."""
        m = AL.shape[1]  # number of examples
        Y_hot = self._one_hot_encode(Y)
        Y_hot = Y_hot.T  # Transpose to match AL shape
        
        # Initialize gradients for output layer
        dZ = AL - Y_hot
        
        # Get gradients for output layer
        final_layer = self.layers[-1]
        final_cache = caches[-1]
        
        final_layer.dW = (1/m) * np.dot(dZ, final_cache['A_prev'].T)
        final_layer.db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(final_layer.W.T, dZ)
        
        # Backward through hidden layers
        for layer, cache in zip(reversed(self.layers[:-1]), reversed(caches[:-1])):
            layer.cache = cache
            dA = layer.backward(dA)
    
    def _one_hot_encode(self, Y: np.ndarray) -> np.ndarray:
        """Convert labels to one-hot encoded format."""
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
        m = Y.shape[0]
        Y_hot = np.zeros((m, self.config.output_size))
        Y_hot[np.arange(m), Y.ravel()] = 1
        return Y_hot
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get all network parameters."""
        params = {}
        for i, layer in enumerate(self.layers):
            params[f'W{i+1}'] = layer.W
            params[f'b{i+1}'] = layer.b
        return params
    
    def set_parameters(self, parameters: Dict[str, np.ndarray]):
        """Set network parameters."""
        for i, layer in enumerate(self.layers):
            layer.W = parameters[f'W{i+1}']
            layer.b = parameters[f'b{i+1}'] 