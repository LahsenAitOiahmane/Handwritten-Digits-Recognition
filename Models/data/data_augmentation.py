import numpy as np
from typing import Tuple
from scipy.ndimage import rotate
import logging

class DataAugmentation:
    """Implements various data augmentation techniques."""
    
    @staticmethod
    def augment_data(X: np.ndarray, Y: np.ndarray, 
                    rotation_range: float = 15.0,
                    noise_factor: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation techniques to the dataset."""
        logging.info("Applying data augmentation...")
        
        # Get original dimensions
        n_features, n_samples = X.shape
        img_size = int(np.sqrt(n_features))
        
        # Reshape for image operations
        X_img = X.T.reshape(-1, img_size, img_size)
        
        # Initialize augmented arrays
        X_aug = []
        Y_aug = []
        
        for i in range(n_samples):
            # Original sample
            X_aug.append(X_img[i])
            Y_aug.append(Y[i])
            
            # Rotated version
            angle = np.random.uniform(-rotation_range, rotation_range)
            rotated = rotate(X_img[i], angle, reshape=False)
            X_aug.append(rotated)
            Y_aug.append(Y[i])
            
            # Noisy version
            noise = np.random.normal(0, noise_factor, X_img[i].shape)
            noisy = np.clip(X_img[i] + noise, 0, 1)
            X_aug.append(noisy)
            Y_aug.append(Y[i])
        
        # Convert back to original format
        X_augmented = np.array(X_aug).reshape(-1, n_features).T
        Y_augmented = np.array(Y_aug)
        
        logging.info(f"Augmented data shape: X={X_augmented.shape}, Y={Y_augmented.shape}")
        return X_augmented, Y_augmented 