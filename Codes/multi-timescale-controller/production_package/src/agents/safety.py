"""
Safety Module for Safe RL agents.

Implements:
- Safety cost calculation
- Hard constraint checking
- Safety shield for action filtering
- Soft constraint penalties
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SafetyConfig:
    """Safety constraint configuration."""
    
    # Hard constraints (violation = cost 1.0)
    tx_power_hard_min: float = 0.0
    tx_power_hard_max: float = 30.0
    obss_pd_hard_min: float = -82.0
    obss_pd_hard_max: float = -62.0
    channel_change_interval_hours: float = 4.0
    
    # Safety Shield (stricter limits, actions blocked)
    tx_power_shield_min: float = 10.0
    tx_power_shield_max: float = 20.0
    obss_pd_shield_min: float = -82.0
    obss_pd_shield_max: float = -68.0
    max_changes_per_window: int = 1
    
    # Soft constraints (penalties)
    p95_retry_threshold: float = 0.08
    min_client_rssi: float = -70.0
    tx_power_step_max: float = 6.0
    config_churn_threshold: float = 0.2
    
    # Penalty weights
    rssi_penalty_weight: float = 0.01
    retry_penalty_weight: float = 1.0
    churn_penalty_weight: float = 0.5


class SafetyModule:
    """
    Safety module for enforcing constraints in Safe RL.
    
    Provides:
    1. Cost calculation for constraint violations
    2. Safety shield for filtering unsafe actions
    3. Soft constraint penalties for reward shaping
    """
    
    # Feature indices (matching generator)
    IDX_CLIENT_COUNT = 0
    IDX_MEDIAN_RSSI = 1
    IDX_P95_RETRY = 2
    IDX_P95_PER = 3
    IDX_CHANNEL_UTIL = 4
    IDX_AVG_THROUGHPUT = 5
    IDX_EDGE_THROUGHPUT = 6
    IDX_NEIGHBOR_RSSI = 7
    IDX_OBSS_PD = 8
    IDX_TX_POWER = 9
    IDX_NOISE_FLOOR = 10
    IDX_CHANNEL_WIDTH = 11
    IDX_AIRTIME = 12
    IDX_CCA_BUSY = 13
    IDX_ROAMING_RATE = 14
    
    # Action indices
    ACTION_INCREASE_TX = 0
    ACTION_DECREASE_TX = 1
    ACTION_INCREASE_OBSS = 2
    ACTION_DECREASE_OBSS = 3
    ACTION_INCREASE_CHANNEL_WIDTH = 4
    ACTION_DECREASE_CHANNEL_WIDTH = 5
    ACTION_INCREASE_CHANNEL_NUMBER = 6
    ACTION_DECREASE_CHANNEL_NUMBER = 7
    ACTION_NOOP = 8
    
    NUM_ACTIONS = 9
    
    def __init__(self, config: Optional[SafetyConfig] = None):
        """Initialize safety module with configuration."""
        self.config = config or SafetyConfig()
    
    def calculate_cost(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray
    ) -> float:
        """
        Calculate safety cost for a transition.
        
        Cost = 1.0 if hard constraint violated
        Cost = sum of soft penalties otherwise (capped at 1.0)
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            Cost value in [0, 1]
        """
        # Check hard constraints first
        hard_cost = self._check_hard_constraints(state, action, next_state)
        if hard_cost >= 1.0:
            return 1.0
        
        # Calculate soft constraint penalties
        soft_cost = self._calculate_soft_penalties(state, action, next_state)
        
        return min(hard_cost + soft_cost, 1.0)
    
    def _check_hard_constraints(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray
    ) -> float:
        """Check hard constraints. Returns 1.0 if any violated."""
        
        # Tx Power out of hard bounds
        tx_power = next_state[self.IDX_TX_POWER]
        if tx_power < self.config.tx_power_hard_min or tx_power > self.config.tx_power_hard_max:
            return 1.0
        
        # OBSS-PD out of hard bounds
        obss_pd = next_state[self.IDX_OBSS_PD]
        if obss_pd < self.config.obss_pd_hard_min or obss_pd > self.config.obss_pd_hard_max:
            return 1.0
        
        # Channel Width out of hard bounds [20, 80] MHz
        channel_width = next_state[self.IDX_CHANNEL_WIDTH]
        if channel_width < 20 or channel_width > 80:
            return 1.0
        
        return 0.0
    
    def _calculate_soft_penalties(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray
    ) -> float:
        """Calculate sum of soft constraint penalties."""
        cost = 0.0
        
        # P95 Retry Rate > threshold
        p95_retry = next_state[self.IDX_P95_RETRY]
        if p95_retry > self.config.p95_retry_threshold:
            cost += (p95_retry - self.config.p95_retry_threshold) * self.config.retry_penalty_weight
        
        # Tx Power step > max
        tx_step = abs(next_state[self.IDX_TX_POWER] - state[self.IDX_TX_POWER])
        if tx_step > self.config.tx_power_step_max:
            cost += (tx_step - self.config.tx_power_step_max) / 6.0
        
        # Client RSSI below threshold
        median_rssi = next_state[self.IDX_MEDIAN_RSSI]
        if median_rssi < self.config.min_client_rssi:
            cost += (self.config.min_client_rssi - median_rssi) * self.config.rssi_penalty_weight
        
        # Increased retry rate (safety concern)
        delta_retry = next_state[self.IDX_P95_RETRY] - state[self.IDX_P95_RETRY]
        if delta_retry > 0.02:
            cost += delta_retry * self.config.retry_penalty_weight
        
        return cost
    
    def is_action_safe(
        self,
        state: np.ndarray,
        action: int,
        denormalize_fn: Optional[callable] = None
    ) -> bool:
        """
        Check if an action is safe given the current state.
        
        Uses Safety Shield limits (stricter than hard constraints).
        
        Args:
            state: Current state (may be normalized)
            action: Proposed action
            denormalize_fn: Optional function to denormalize state
            
        Returns:
            True if action is safe, False otherwise
        """
        # Denormalize if needed
        if denormalize_fn is not None:
            state = denormalize_fn(state)
        
        tx_power = state[self.IDX_TX_POWER]
        obss_pd = state[self.IDX_OBSS_PD]
        
        if action == self.ACTION_INCREASE_TX:
            new_tx = tx_power + 2.0
            return new_tx <= self.config.tx_power_shield_max
        
        elif action == self.ACTION_DECREASE_TX:
            new_tx = tx_power - 2.0
            return new_tx >= self.config.tx_power_shield_min
        
        elif action == self.ACTION_INCREASE_OBSS:
            new_obss = obss_pd + 4.0
            return new_obss <= self.config.obss_pd_shield_max
        
        elif action == self.ACTION_DECREASE_OBSS:
            new_obss = obss_pd - 4.0
            return new_obss >= self.config.obss_pd_shield_min
        
        # No-op is always safe
        return True
    
    def get_safe_actions(
        self,
        state: np.ndarray,
        denormalize_fn: Optional[callable] = None
    ) -> List[int]:
        """
        Get list of safe actions for the current state.
        
        Returns:
            List of safe action indices
        """
        safe_actions = []
        for action in range(self.NUM_ACTIONS):
            if self.is_action_safe(state, action, denormalize_fn):
                safe_actions.append(action)
        return safe_actions
    
    def mask_unsafe_actions(
        self,
        q_values: torch.Tensor,
        states: torch.Tensor,
        denormalize_fn: Optional[callable] = None,
        mask_value: float = -100.0  # CHANGED: Finite mask value (was -1e9)
    ) -> torch.Tensor:
        """
        Mask Q-values for unsafe actions.
        
        IMPORTANT: Uses finite mask value (-100) instead of -1e9 to prevent
        Q-value explosion during learning. The mask should only be applied
        during ACTION SELECTION, not during Q-learning updates.
        
        Args:
            q_values: Q-values [batch_size, num_actions]
            states: States [batch_size, state_dim]
            denormalize_fn: Optional denormalization function
            mask_value: Value to assign to unsafe actions (default: -100)
            
        Returns:
            Masked Q-values
        """
        batch_size = q_values.shape[0]
        masked_q = q_values.clone()
        
        for i in range(batch_size):
            state = states[i].cpu().numpy()
            for action in range(self.NUM_ACTIONS):
                if not self.is_action_safe(state, action, denormalize_fn):
                    masked_q[i, action] = mask_value
        
        return masked_q
    
    def get_safe_action_mask(
        self,
        state: np.ndarray,
        denormalize_fn: Optional[callable] = None
    ) -> np.ndarray:
        """
        Get boolean mask of safe actions.
        
        Returns:
            Boolean array [num_actions] where True = safe
        """
        mask = np.zeros(self.NUM_ACTIONS, dtype=bool)
        for action in range(self.NUM_ACTIONS):
            mask[action] = self.is_action_safe(state, action, denormalize_fn)
        return mask
    
    def filter_action(
        self,
        state: np.ndarray,
        proposed_action: int,
        denormalize_fn: Optional[callable] = None
    ) -> int:
        """
        Filter proposed action through safety shield.
        
        If proposed action is unsafe, returns No-op.
        
        Args:
            state: Current state
            proposed_action: Action proposed by policy
            denormalize_fn: Optional denormalization function
            
        Returns:
            Safe action (original or No-op)
        """
        if self.is_action_safe(state, proposed_action, denormalize_fn):
            return proposed_action
        return self.ACTION_NOOP
    
    def get_cost_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate costs for a batch of transitions.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size]
            next_states: Batch of next states [batch_size, state_dim]
            
        Returns:
            Costs [batch_size]
        """
        batch_size = states.shape[0]
        costs = torch.zeros(batch_size, device=states.device)
        
        states_np = states.cpu().numpy()
        actions_np = actions.cpu().numpy()
        next_states_np = next_states.cpu().numpy()
        
        for i in range(batch_size):
            costs[i] = self.calculate_cost(
                states_np[i],
                int(actions_np[i]),
                next_states_np[i]
            )
        
        return costs


class CostCritic(torch.nn.Module):
    """
    Neural network for predicting costs (for constrained RL).
    
    Can be used to learn a cost model from data.
    """
    
    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 5,
        hidden_dims: List[int] = [256, 128]
    ):
        super().__init__()
        
        # State-action cost predictor
        layers = []
        prev_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(torch.nn.Linear(prev_dim, 1))
        layers.append(torch.nn.Sigmoid())  # Cost in [0, 1]
        
        self.network = torch.nn.Sequential(*layers)
    
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict cost for state-action pairs.
        
        Args:
            states: [batch_size, state_dim]
            actions: [batch_size] (will be one-hot encoded)
            
        Returns:
            Predicted costs [batch_size, 1]
        """
        # One-hot encode actions
        action_onehot = torch.nn.functional.one_hot(
            actions.long(), num_classes=9
        ).float()
        
        # Concatenate state and action
        x = torch.cat([states, action_onehot], dim=-1)
        
        return self.network(x)

