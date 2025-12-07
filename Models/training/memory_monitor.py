import psutil
import logging
from typing import Dict
import time

class MemoryMonitor:
    """Monitors memory usage during training."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        self.start_time = time.time()
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current_memory)
        
        return {
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': current_memory - self.initial_memory,
            'training_time_s': time.time() - self.start_time
        }
    
    def log_memory_usage(self, epoch: int):
        """Log current memory usage."""
        stats = self.get_memory_stats()
        logging.info(
            f"Epoch {epoch} - Memory Usage: "
            f"Current: {stats['current_memory_mb']:.1f}MB, "
            f"Peak: {stats['peak_memory_mb']:.1f}MB, "
            f"Increase: {stats['memory_increase_mb']:.1f}MB"
        ) 