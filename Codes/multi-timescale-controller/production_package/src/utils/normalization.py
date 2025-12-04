"""
Normalization utilities for Safe RL.

Provides:
- Reward normalization
- Running mean/std estimation
- Advantage normalization
"""

import numpy as np
import torch
from typing import Optional, Tuple


class RunningMeanStd:
    """
    Tracks running mean and standard deviation.
    
    Uses Welford's online algorithm for numerical stability.
    """
    
    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-8):
        """
        Initialize running statistics tracker.
        
        Args:
            shape: Shape of the values being tracked
            epsilon: Small constant for numerical stability
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon
    
    def update(self, x: np.ndarray):
        """
        Update running statistics with new batch of values.
        
        Args:
            x: Batch of values [batch_size, *shape]
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int
    ):
        """Update from batch statistics."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        
        new_var = m2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    @property
    def std(self) -> np.ndarray:
        """Get standard deviation."""
        return np.sqrt(self.var + self.epsilon)
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize values using running statistics."""
        return (x - self.mean) / self.std
    
    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize values."""
        return x * self.std + self.mean


class RewardNormalizer:
    """
    Normalizes rewards using running statistics.
    
    Options:
    - Running mean/std normalization
    - Reward clipping
    - Return normalization
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        clip_range: Optional[Tuple[float, float]] = (-10.0, 10.0),
        normalize_returns: bool = False,
        epsilon: float = 1e-8
    ):
        """
        Initialize reward normalizer.
        
        Args:
            gamma: Discount factor for return estimation
            clip_range: Range to clip normalized rewards (None to disable)
            normalize_returns: Whether to normalize returns instead of rewards
            epsilon: Small constant for numerical stability
        """
        self.gamma = gamma
        self.clip_range = clip_range
        self.normalize_returns = normalize_returns
        self.epsilon = epsilon
        
        self.reward_stats = RunningMeanStd(shape=())
        self.return_stats = RunningMeanStd(shape=())
        
        self._returns = None
    
    def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        """
        Normalize a batch of rewards.
        
        Args:
            reward: Rewards to normalize [batch_size]
            
        Returns:
            Normalized rewards
        """
        # Update statistics
        self.reward_stats.update(reward)
        
        # Normalize
        normalized = (reward - self.reward_stats.mean) / self.reward_stats.std
        
        # Clip if specified
        if self.clip_range is not None:
            normalized = np.clip(normalized, self.clip_range[0], self.clip_range[1])
        
        return normalized
    
    def normalize_returns(self, returns: np.ndarray) -> np.ndarray:
        """
        Normalize a batch of returns.
        
        Args:
            returns: Returns to normalize [batch_size]
            
        Returns:
            Normalized returns
        """
        self.return_stats.update(returns)
        normalized = returns / (self.return_stats.std + self.epsilon)
        
        if self.clip_range is not None:
            normalized = np.clip(normalized, self.clip_range[0], self.clip_range[1])
        
        return normalized
    
    def __call__(self, reward: np.ndarray) -> np.ndarray:
        """Normalize rewards (convenience method)."""
        return self.normalize_reward(reward)


class AdvantageNormalizer:
    """
    Normalizes advantages for policy gradient methods.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
    
    def normalize(
        self,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Normalize advantages.
        
        Args:
            advantages: Advantage values [batch_size]
            mask: Optional mask for valid entries
            
        Returns:
            Normalized advantages
        """
        if mask is not None:
            # Only use valid entries for statistics
            valid_advantages = advantages[mask]
            mean = valid_advantages.mean()
            std = valid_advantages.std()
        else:
            mean = advantages.mean()
            std = advantages.std()
        
        normalized = (advantages - mean) / (std + self.epsilon)
        return normalized


def normalize_batch(
    batch: dict,
    reward_normalizer: Optional[RewardNormalizer] = None,
    state_mean: Optional[np.ndarray] = None,
    state_std: Optional[np.ndarray] = None
) -> dict:
    """
    Normalize a batch of transitions.
    
    Args:
        batch: Dictionary with states, rewards, etc.
        reward_normalizer: Optional reward normalizer
        state_mean: Optional state mean for normalization
        state_std: Optional state std for normalization
        
    Returns:
        Normalized batch
    """
    normalized = batch.copy()
    
    # Normalize rewards
    if reward_normalizer is not None and 'rewards' in batch:
        rewards = batch['rewards']
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.numpy()
        normalized['rewards'] = torch.FloatTensor(reward_normalizer(rewards))
    
    # Normalize states
    if state_mean is not None and state_std is not None:
        if 'states' in batch:
            states = batch['states']
            if isinstance(states, torch.Tensor):
                states = states.numpy()
            normalized['states'] = torch.FloatTensor(
                (states - state_mean) / (state_std + 1e-8)
            )
        
        if 'next_states' in batch:
            next_states = batch['next_states']
            if isinstance(next_states, torch.Tensor):
                next_states = next_states.numpy()
            normalized['next_states'] = torch.FloatTensor(
                (next_states - state_mean) / (state_std + 1e-8)
            )
    
    return normalized

