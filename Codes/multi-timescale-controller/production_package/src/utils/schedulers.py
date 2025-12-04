"""
Learning rate and hyperparameter schedulers.

Provides:
- Linear scheduler
- Exponential scheduler
- Cosine annealing
- Warmup schedulers
"""

import numpy as np
from typing import Optional


class LinearScheduler:
    """
    Linear interpolation between start and end values.
    """
    
    def __init__(
        self,
        start_value: float,
        end_value: float,
        total_steps: int,
        warmup_steps: int = 0
    ):
        """
        Initialize linear scheduler.
        
        Args:
            start_value: Starting value
            end_value: Ending value
            total_steps: Total number of steps
            warmup_steps: Number of warmup steps (value stays at start)
        """
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def step(self) -> float:
        """Advance one step and return current value."""
        value = self.get_value()
        self.current_step += 1
        return value
    
    def get_value(self) -> float:
        """Get current value without advancing."""
        if self.current_step < self.warmup_steps:
            return self.start_value
        
        progress = min(
            (self.current_step - self.warmup_steps) / 
            max(self.total_steps - self.warmup_steps, 1),
            1.0
        )
        
        return self.start_value + progress * (self.end_value - self.start_value)
    
    def reset(self):
        """Reset scheduler."""
        self.current_step = 0


class ExponentialScheduler:
    """
    Exponential decay scheduler.
    """
    
    def __init__(
        self,
        start_value: float,
        decay_rate: float,
        min_value: float = 0.0,
        warmup_steps: int = 0
    ):
        """
        Initialize exponential scheduler.
        
        Args:
            start_value: Starting value
            decay_rate: Decay rate per step (e.g., 0.99)
            min_value: Minimum value
            warmup_steps: Number of warmup steps
        """
        self.start_value = start_value
        self.decay_rate = decay_rate
        self.min_value = min_value
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.current_value = start_value
    
    def step(self) -> float:
        """Advance one step and return current value."""
        value = self.get_value()
        self.current_step += 1
        
        if self.current_step >= self.warmup_steps:
            self.current_value = max(
                self.min_value,
                self.current_value * self.decay_rate
            )
        
        return value
    
    def get_value(self) -> float:
        """Get current value without advancing."""
        return self.current_value
    
    def reset(self):
        """Reset scheduler."""
        self.current_step = 0
        self.current_value = self.start_value


class CosineAnnealingScheduler:
    """
    Cosine annealing scheduler with optional warmup.
    """
    
    def __init__(
        self,
        start_value: float,
        end_value: float,
        total_steps: int,
        warmup_steps: int = 0
    ):
        """
        Initialize cosine annealing scheduler.
        
        Args:
            start_value: Starting value
            end_value: Ending value
            total_steps: Total number of steps
            warmup_steps: Number of warmup steps
        """
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def step(self) -> float:
        """Advance one step and return current value."""
        value = self.get_value()
        self.current_step += 1
        return value
    
    def get_value(self) -> float:
        """Get current value without advancing."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            progress = self.current_step / max(self.warmup_steps, 1)
            return self.end_value + progress * (self.start_value - self.end_value)
        
        # Cosine annealing
        progress = min(
            (self.current_step - self.warmup_steps) / 
            max(self.total_steps - self.warmup_steps, 1),
            1.0
        )
        
        cosine_value = 0.5 * (1 + np.cos(np.pi * progress))
        return self.end_value + cosine_value * (self.start_value - self.end_value)
    
    def reset(self):
        """Reset scheduler."""
        self.current_step = 0


class WarmupScheduler:
    """
    Warmup wrapper for any base scheduler.
    """
    
    def __init__(
        self,
        base_scheduler,
        warmup_steps: int,
        warmup_start_value: float
    ):
        """
        Initialize warmup scheduler.
        
        Args:
            base_scheduler: Base scheduler to wrap
            warmup_steps: Number of warmup steps
            warmup_start_value: Starting value during warmup
        """
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.warmup_start_value = warmup_start_value
        self.current_step = 0
    
    def step(self) -> float:
        """Advance one step and return current value."""
        value = self.get_value()
        self.current_step += 1
        if self.current_step > self.warmup_steps:
            self.base_scheduler.step()
        return value
    
    def get_value(self) -> float:
        """Get current value without advancing."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            progress = self.current_step / max(self.warmup_steps, 1)
            target_value = self.base_scheduler.get_value()
            return self.warmup_start_value + progress * (target_value - self.warmup_start_value)
        
        return self.base_scheduler.get_value()
    
    def reset(self):
        """Reset scheduler."""
        self.current_step = 0
        self.base_scheduler.reset()

