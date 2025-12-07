import numpy as np
from typing import Tuple, Dict

class Activations:
    """Neural network activation functions."""
    
    @staticmethod
    def relu(Z: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, Z)
    
    @staticmethod
    def relu_derivative(Z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU activation."""
        return np.where(Z > 0, 1, 0)
    
    @staticmethod
    def softmax(Z: np.ndarray) -> np.ndarray:
        """Softmax activation function."""
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

class Layer:
    """Base neural network layer."""
    
    def __init__(self, input_size: int, output_size: int):
        """Initialize layer parameters."""
        # He initialization
        self.W = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.b = np.zeros((output_size, 1))
        self.dW = None
        self.db = None
        self.cache = {}
        
    def forward(self, A_prev: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict]:
        """Forward propagation through layer."""
        cache = {}
        cache['A_prev'] = A_prev
        
        Z = np.dot(self.W, A_prev) + self.b
        A = Activations.relu(Z)
        
        if training:
            cache['Z'] = Z
            cache['A'] = A
        
        return A, cache
    
    def backward(self, dA: np.ndarray) -> np.ndarray:
        """Backward propagation through layer."""
        m = self.cache['A_prev'].shape[1]
        
        dZ = dA * Activations.relu_derivative(self.cache['Z'])
        self.dW = (1/m) * np.dot(dZ, self.cache['A_prev'].T)
        self.db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)
        
        return dA_prev
    
    def update_parameters(self, learning_rate: float):
        """Update layer parameters using gradients."""
        if self.dW is not None and self.db is not None:
            self.W -= learning_rate * self.dW
            self.b -= learning_rate * self.db 