"""
Implicit Q-Learning (IQL) Agent for Safe Offline RL.

IQL avoids querying out-of-distribution actions entirely by using
expectile regression for value estimation. This makes it simpler
and often more effective than CQL or BCQ.

Key Features:
1. No explicit policy constraint (simpler than BCQ)
2. Expectile regression for value function (less conservative than CQL)
3. Advantage-weighted policy extraction
4. Works well with limited data

Why IQL for RRM?
- Simpler than CQL/BCQ (no additional models)
- Less conservative (better action diversity)
- State-of-the-art offline RL performance
- Good balance of safety and exploration

Reference: Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning", ICLR 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List
from copy import deepcopy

from .networks import QNetwork
from .safety import SafetyModule, SafetyConfig


class ValueNetwork(nn.Module):
    """Value function V(s) for IQL."""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class IQLAgent:
    """
    Implicit Q-Learning Agent.
    
    IQL uses expectile regression to estimate V(s) as a lower expectile
    of Q(s,a), avoiding the need to evaluate Q for out-of-distribution actions.
    
    The policy is extracted by advantage-weighted behavior cloning.
    """
    
    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 5,
        hidden_dims: List[int] = [256, 256, 128],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.7,  # IQL expectile (0.5 = mean, higher = more optimistic)
        temperature: float = 3.0,  # Temperature for advantage-weighted policy
        target_update_freq: int = 100,
        device: str = 'auto',
        safety_config: Optional[SafetyConfig] = None
    ):
        """
        Initialize IQL Agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            expectile: IQL expectile for V-function (0.5-0.9, higher = less conservative)
            temperature: Temperature for advantage-weighted policy extraction
            target_update_freq: Steps between target updates
            device: Device to run on
            safety_config: Safety constraint configuration
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        self.target_update_freq = target_update_freq
        
        # Q-Networks (Double Q-learning)
        self.q_network1 = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        self.q_network2 = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        self.target_q1 = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        self.target_q2 = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        self.target_q1.load_state_dict(self.q_network1.state_dict())
        self.target_q2.load_state_dict(self.q_network2.state_dict())
        
        # Value Network V(s)
        self.value_network = ValueNetwork(
            state_dim=state_dim,
            hidden_dims=hidden_dims[:-1]  # Slightly smaller
        ).to(self.device)
        
        # Policy Network (for advantage-weighted extraction)
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim)
        ).to(self.device)
        
        # Optimizers
        self.q_optimizer = optim.Adam(
            list(self.q_network1.parameters()) + list(self.q_network2.parameters()),
            lr=learning_rate
        )
        self.v_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Safety module
        self.safety = SafetyModule(safety_config)
        
        # Training state
        self.train_step = 0
        self.training_history = {
            'q_loss': [],
            'v_loss': [],
            'policy_loss': [],
            'mean_q': [],
            'mean_v': [],
            'advantage': []
        }
        
        # Normalization
        self.state_mean = None
        self.state_std = None
    
    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        """Set state normalization parameters."""
        self.state_mean = torch.FloatTensor(mean).to(self.device)
        self.state_std = torch.FloatTensor(std).to(self.device)
    
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state."""
        if self.state_mean is None:
            return state
        return (state - self.state_mean) / (self.state_std + 1e-8)
    
    def denormalize_state(self, state: np.ndarray) -> np.ndarray:
        """Denormalize state for safety checking."""
        if self.state_mean is not None:
            mean = self.state_mean.cpu().numpy()
            std = self.state_std.cpu().numpy()
            return state * std + mean
        return state
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        use_safety_shield: bool = True
    ) -> int:
        """
        Select action using IQL policy.
        
        Uses advantage-weighted policy with optional stochastic sampling.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            state_normalized = self.normalize_state(state_tensor)
            
            # Get policy logits
            logits = self.policy_network(state_normalized).squeeze()
            
            # Apply safety mask
            if use_safety_shield:
                safe_mask = torch.tensor([
                    self.safety.is_action_safe(state, a, self.denormalize_state)
                    for a in range(self.action_dim)
                ], dtype=torch.bool, device=self.device)
                
                # Hard mask unsafe actions
                logits[~safe_mask] = float('-inf')
                
                # Fallback to No-op if all actions masked
                if not safe_mask.any():
                    return 4  # No-op
            
            if deterministic:
                action = logits.argmax().item()
            else:
                # Softmax sampling
                probs = F.softmax(logits, dim=-1)
                probs = probs + 1e-8
                probs = probs / probs.sum()
                action = torch.multinomial(probs, 1).item()
        
        return action
    
    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """
        Compute expectile loss.
        
        L_τ(u) = |τ - 1(u < 0)| * u^2
        
        This is the key innovation of IQL - it estimates V as a lower expectile of Q.
        """
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        return (weight * diff.pow(2)).mean()
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one IQL update step.
        
        Updates:
        1. Q-networks with standard TD learning
        2. V-network with expectile regression
        3. Policy with advantage-weighted BC
        """
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        costs = batch['costs'].to(self.device)
        
        # Normalize states
        states = self.normalize_state(states)
        next_states = self.normalize_state(next_states)
        
        # ========== Update V-Network (Expectile Regression) ==========
        with torch.no_grad():
            # Get Q-values for the actions taken
            q1 = self.target_q1(states).gather(1, actions.unsqueeze(1).long()).squeeze()
            q2 = self.target_q2(states).gather(1, actions.unsqueeze(1).long()).squeeze()
            target_q = torch.min(q1, q2)
        
        v = self.value_network(states).squeeze()
        
        # Expectile loss: V should be the τ-expectile of Q
        v_loss = self._expectile_loss(target_q - v)
        
        self.v_optimizer.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1.0)
        self.v_optimizer.step()
        
        # ========== Update Q-Networks ==========
        with torch.no_grad():
            next_v = self.value_network(next_states).squeeze()
            safety_penalty = costs * 0.3
            targets = rewards - safety_penalty + self.gamma * next_v * (1 - dones)
        
        current_q1 = self.q_network1(states).gather(1, actions.unsqueeze(1).long()).squeeze()
        current_q2 = self.q_network2(states).gather(1, actions.unsqueeze(1).long()).squeeze()
        
        q_loss = F.smooth_l1_loss(current_q1, targets) + F.smooth_l1_loss(current_q2, targets)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q_network1.parameters()) + list(self.q_network2.parameters()),
            1.0
        )
        self.q_optimizer.step()
        
        # ========== Update Policy (Advantage-Weighted BC) ==========
        with torch.no_grad():
            v_current = self.value_network(states).squeeze()
            q1_current = self.q_network1(states).gather(1, actions.unsqueeze(1).long()).squeeze()
            q2_current = self.q_network2(states).gather(1, actions.unsqueeze(1).long()).squeeze()
            q_current = torch.min(q1_current, q2_current)
            
            # Advantage = Q(s,a) - V(s)
            advantage = q_current - v_current
            
            # Advantage weights for BC (exponential, temperature-scaled)
            weights = torch.exp(advantage / self.temperature)
            weights = torch.clamp(weights, 0, 100)  # Clip for stability
        
        # Policy loss: weighted behavior cloning
        logits = self.policy_network(states)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1).long()).squeeze()
        
        policy_loss = -(weights * action_log_probs).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # Soft update targets
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self._soft_update_targets()
        
        metrics = {
            'q_loss': q_loss.item(),
            'v_loss': v_loss.item(),
            'policy_loss': policy_loss.item(),
            'mean_q': q_current.mean().item(),
            'mean_v': v_current.mean().item(),
            'mean_advantage': advantage.mean().item(),
            'cost_rate': (costs > 0).float().mean().item()
        }
        
        self.training_history['q_loss'].append(q_loss.item())
        self.training_history['v_loss'].append(v_loss.item())
        self.training_history['policy_loss'].append(policy_loss.item())
        self.training_history['mean_q'].append(metrics['mean_q'])
        self.training_history['mean_v'].append(metrics['mean_v'])
        self.training_history['advantage'].append(metrics['mean_advantage'])
        
        return metrics
    
    def _soft_update_targets(self):
        """Soft update target networks."""
        for target, source in [
            (self.target_q1, self.q_network1),
            (self.target_q2, self.q_network2)
        ]:
            for t_param, s_param in zip(target.parameters(), source.parameters()):
                t_param.data.copy_(self.tau * s_param.data + (1 - self.tau) * t_param.data)
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        use_safety_shield: bool = True
    ) -> Dict[str, float]:
        """Evaluate the agent on a dataset."""
        self.q_network1.eval()
        self.q_network2.eval()
        self.value_network.eval()
        self.policy_network.eval()
        
        all_q_values = []
        all_v_values = []
        all_actions = []
        constraint_violations = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                states = batch['state'].to(self.device)
                states_normalized = self.normalize_state(states)
                
                q1 = self.q_network1(states_normalized)
                q2 = self.q_network2(states_normalized)
                q_values = torch.min(q1, q2)
                v_values = self.value_network(states_normalized)
                
                all_q_values.append(q_values.cpu())
                all_v_values.append(v_values.cpu())
                
                for i in range(states.shape[0]):
                    state = states[i].cpu().numpy()
                    action = self.select_action(state, deterministic=True, use_safety_shield=use_safety_shield)
                    all_actions.append(action)
                    
                    if not self.safety.is_action_safe(state, action, self.denormalize_state):
                        constraint_violations += 1
                    total_samples += 1
        
        all_q_values = torch.cat(all_q_values, dim=0)
        all_v_values = torch.cat(all_v_values, dim=0)
        all_actions = torch.tensor(all_actions)
        
        action_counts = torch.bincount(all_actions, minlength=self.action_dim)
        action_dist = action_counts.float() / len(all_actions)
        
        # Calculate action diversity
        probs = action_dist + 1e-8
        diversity = -(probs * torch.log(probs)).sum().item()
        max_entropy = np.log(self.action_dim)
        diversity_score = diversity / max_entropy * 100
        
        metrics = {
            'mean_q': all_q_values.mean().item(),
            'max_q': all_q_values.max().item(),
            'min_q': all_q_values.min().item(),
            'mean_v': all_v_values.mean().item(),
            'constraint_violation_rate': constraint_violations / max(total_samples, 1),
            'action_distribution': action_dist.tolist(),
            'action_diversity_score': diversity_score
        }
        
        self.q_network1.train()
        self.q_network2.train()
        self.value_network.train()
        self.policy_network.train()
        
        return metrics
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'q_network1': self.q_network1.state_dict(),
            'q_network2': self.q_network2.state_dict(),
            'target_q1': self.target_q1.state_dict(),
            'target_q2': self.target_q2.state_dict(),
            'value_network': self.value_network.state_dict(),
            'policy_network': self.policy_network.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'v_optimizer': self.v_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'train_step': self.train_step,
            'training_history': self.training_history,
            'state_mean': self.state_mean,
            'state_std': self.state_std
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.q_network1.load_state_dict(checkpoint['q_network1'])
        self.q_network2.load_state_dict(checkpoint['q_network2'])
        self.target_q1.load_state_dict(checkpoint['target_q1'])
        self.target_q2.load_state_dict(checkpoint['target_q2'])
        self.value_network.load_state_dict(checkpoint['value_network'])
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.v_optimizer.load_state_dict(checkpoint['v_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.train_step = checkpoint['train_step']
        self.training_history = checkpoint['training_history']
        if checkpoint['state_mean'] is not None:
            self.state_mean = checkpoint['state_mean'].to(self.device)
            self.state_std = checkpoint['state_std'].to(self.device)

