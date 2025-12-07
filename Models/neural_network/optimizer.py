import numpy as np
from typing import Dict, List
import logging

class Optimizer:
    """Implementation of optimization algorithms."""
    
    def __init__(self, learning_rate: float = 0.001, 
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8):
        """Initialize optimizer parameters."""
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step
    
    def adam(self, parameters: Dict[str, np.ndarray], 
             gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Adam optimization algorithm."""
        if not self.m:  # Initialize momentum if empty
            for key in parameters:
                self.m[key] = np.zeros_like(parameters[key])
                self.v[key] = np.zeros_like(parameters[key])
        
        self.t += 1
        updates = {}
        
        for key in parameters:
            # Update momentum
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.square(gradients[key])
            
            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Parameter update
            updates[key] = parameters[key] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updates 