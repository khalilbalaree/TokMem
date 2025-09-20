"""
Simple replay buffer implementation for continual learning.
Uses reservoir sampling to maintain a fixed-size buffer of previous examples.
"""

import random
from typing import List, Dict, Any


class SimpleReplayBuffer:
    """
    A simple replay buffer using reservoir sampling algorithm.
    Maintains a fixed-size buffer of samples from all previous rounds.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the replay buffer.
        
        Args:
            max_size: Maximum number of samples to store in the buffer
        """
        self.buffer = []
        self.max_size = max_size
        self.total_seen = 0  # Track total samples seen for reservoir sampling
    
    def add(self, samples: List[Dict[str, Any]]):
        """
        Add samples to the buffer using reservoir sampling.
        This ensures uniform probability of keeping any sample regardless of order.
        
        Args:
            samples: List of samples to add (each sample is a dict with data)
        """
        for sample in samples:
            self.total_seen += 1
            
            if len(self.buffer) < self.max_size:
                # Buffer not full yet, just append
                self.buffer.append(sample)
            else:
                # Buffer is full, use reservoir sampling
                # Probability of keeping new sample: max_size / total_seen
                j = random.randint(1, self.total_seen)
                if j <= self.max_size:
                    # Replace a random existing sample
                    idx = random.randint(0, self.max_size - 1)
                    self.buffer[idx] = sample
    
    def sample(self, n: int) -> List[Dict[str, Any]]:
        """
        Sample n items from the buffer.
        
        Args:
            n: Number of samples to retrieve
            
        Returns:
            List of sampled items (may be less than n if buffer is small)
        """
        if len(self.buffer) == 0:
            return []
        
        # Sample without replacement for this batch
        n_samples = min(n, len(self.buffer))
        return random.sample(self.buffer, n_samples)
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all samples in the buffer.
        
        Returns:
            List of all samples currently in the buffer
        """
        return self.buffer.copy()
    
    def size(self) -> int:
        """
        Get current size of the buffer.
        
        Returns:
            Number of samples currently stored
        """
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []
        self.total_seen = 0
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def __repr__(self):
        """String representation of the buffer."""
        return f"SimpleReplayBuffer(size={len(self.buffer)}/{self.max_size}, total_seen={self.total_seen})"