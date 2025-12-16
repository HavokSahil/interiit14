"""
Conservative Q-Learning (CQL) Agent for Safe Offline RL.

CQL is the main algorithm for safe offline RL in RRM.
It learns a conservative Q-function that underestimates action values
for out-of-distribution actions, making it safe for offline learning.

Reference: Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning", NeurIPS 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List
from copy import deepcopy

import sys
from .networks import QNetwork, DuelingQNetwork
from .safety import SafetyModule, SafetyConfig


class CQLAgent:
    """
    Conservative Q-Learning Agent for Safe Offline RL.
    
    Key features:
    1. Conservative Q-value estimation (prevents overestimation)
    2. Offline learning from fixed dataset
    3. Safety constraint integration
    4. Safety shield for action filtering
    5. Balanced exploration vs safety trade-off
    """
    
    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 5,
        hidden_dims: List[int] = [256, 256, 128],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.5,  # Reduced from 1.0 for less conservatism
        min_q_weight: float = 1.0,  # Reduced from 5.0 for more action diversity
        num_random_actions: int = 10,
        target_update_freq: int = 100,
        use_dueling: bool = False,
        use_lagrange: bool = True,  # Automatic alpha tuning
        target_action_gap: float = 1.0,  # Target gap for lagrange
        temperature: float = 1.0,  # Temperature for action selection
        temperature_decay: float = 0.995,  # Temperature decay rate
        min_temperature: float = 0.1,  # Minimum temperature
        device: str = 'auto',
        safety_config: Optional[SafetyConfig] = None
    ):
        """
        Initialize CQL Agent.
        
        Args:
            state_dim: Dimension of state space (15 features)
            action_dim: Number of discrete actions (5)
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            tau: Soft update coefficient for target network
            alpha: CQL regularization weight (Lagrange multiplier)
            min_q_weight: Weight for conservative Q penalty
            num_random_actions: Number of random actions for CQL loss
            target_update_freq: Steps between target network updates
            use_dueling: Whether to use Dueling DQN architecture
            use_lagrange: Whether to use automatic alpha tuning
            target_action_gap: Target gap for lagrange multiplier
            temperature: Temperature for softmax action selection (higher = more exploration)
            temperature_decay: Rate at which temperature decays
            min_temperature: Minimum temperature value
            device: Device to run on ('auto', 'cpu', 'cuda')
            safety_config: Safety constraint configuration
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.min_q_weight = min_q_weight
        self.num_random_actions = num_random_actions
        self.target_update_freq = target_update_freq
        self.use_lagrange = use_lagrange
        self.target_action_gap = target_action_gap
        
        # Initialize networks
        NetworkClass = DuelingQNetwork if use_dueling else QNetwork
        self.q_network = NetworkClass(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        self.target_network = NetworkClass(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # Copy weights to target
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Lagrange multiplier for automatic alpha tuning
        if use_lagrange:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        
        # Safety module
        self.safety = SafetyModule(safety_config)
        
        # Training state
        self.train_step = 0
        self.training_history = {
            'q_loss': [],
            'cql_loss': [],
            'total_loss': [],
            'mean_q': [],
            'alpha': [],
            'constraint_violations': []
        }
        
        # Normalization parameters (set during training)
        self.state_mean = None
        self.state_std = None
        
        # UCB Exploration (NEW - Phase 3 improvement)
        self.action_counts = np.zeros(action_dim)
        self.total_actions = 0
        self.exploration_weight = 1.0  # UCB coefficient
    
    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        """Set state normalization parameters."""
        self.state_mean = torch.FloatTensor(mean).to(self.device)
        self.state_std = torch.FloatTensor(std).to(self.device)
    
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
        epsilon: float = 0.0,
        use_safety_shield: bool = True,
        use_temperature: bool = True,
        use_ucb: bool = True
    ) -> int:
        """
        Select action using UCB exploration + temperature-scaled softmax.
        
        UCB (Upper Confidence Bound) adds exploration bonus to less-tried actions:
        Q_explore(a) = Q(s,a) + c * sqrt(log(t) / N(a))
        
        This prevents CQL from always picking No-op.
        
        Args:
            state: Current state (normalized)
            epsilon: Exploration probability (for random exploration)
            use_safety_shield: Whether to filter unsafe actions
            use_temperature: Whether to use temperature-based selection
            use_ucb: Whether to add UCB exploration bonus
            
        Returns:
            Selected action index
        """
        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            if use_safety_shield:
                safe_actions = self.safety.get_safe_actions(
                    state, None
                )
                if safe_actions:
                    return np.random.choice(safe_actions)
            return np.random.randint(0, self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze()
            
            # === UCB EXPLORATION BONUS (Phase 3 improvement) ===
            if use_ucb and self.total_actions > 0:
                ucb_bonus = np.zeros(self.action_dim)
                for a in range(self.action_dim):
                    if self.action_counts[a] > 0:
                        # UCB formula: c * sqrt(log(t) / N(a))
                        ucb_bonus[a] = self.exploration_weight * np.sqrt(
                            2 * np.log(self.total_actions + 1) / self.action_counts[a]
                        )
                    else:
                        # High bonus for untried actions (encourages exploration)
                        ucb_bonus[a] = 5.0
                
                # Add UCB bonus to Q-values
                q_values = q_values + torch.FloatTensor(ucb_bonus).to(self.device)
            
            # === SAFETY MASKING ===
            if use_safety_shield:
                safe_mask = self.safety.get_safe_action_mask(state, None)
                for i in range(self.action_dim):
                    if not safe_mask[i]:
                        q_values[i] = -100  # Finite mask value
            
            # === ACTION SELECTION ===
            if use_temperature and self.temperature > self.min_temperature:
                # Temperature-scaled softmax for stochastic exploration
                scaled_q = q_values / self.temperature
                # Clamp to prevent overflow
                scaled_q = torch.clamp(scaled_q, -20, 20)
                probs = F.softmax(scaled_q, dim=-1)
                
                # Add small epsilon to prevent zero probabilities
                probs = probs + 1e-8
                probs = probs / probs.sum()
                
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
            else:
                # Greedy action
                action = q_values.argmax().item()
            
            # Track action for UCB
            self.action_counts[action] += 1
            self.total_actions += 1
        
        return action
    
    def decay_temperature(self):
        """Decay the temperature for exploration."""
        self.temperature = max(
            self.min_temperature,
            self.temperature * self.temperature_decay
        )
    
    def compute_cql_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        q_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CQL conservative regularization loss.
        
        CQL Loss = α * E[log(sum_a exp(Q(s,a))) - Q(s, a_data)]
        
        This penalizes high Q-values for actions not in the dataset.
        
        Returns:
            cql_loss: The CQL regularization loss
            action_gap: Gap between logsumexp and data Q-values (for Lagrange)
        """
        batch_size = states.shape[0]
        
        # Q-values for actions in dataset
        q_data = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Log-sum-exp of Q-values (approximates max)
        logsumexp_q = torch.logsumexp(q_values, dim=1)
        
        # Action gap for Lagrange tuning
        action_gap = (logsumexp_q - q_data).mean()
        
        # CQL loss: push down Q-values for all actions, push up for dataset actions
        cql_loss = self.min_q_weight * action_gap
        
        return cql_loss, action_gap
    
    def compute_td_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        costs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute TD loss for Q-learning.
        
        Returns:
            td_loss: Temporal difference loss
            q_values: Current Q-values (for CQL loss)
        """
        # Current Q-values
        q_values = self.q_network(states)
        q_current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (Double DQN style)
        with torch.no_grad():
            # Select actions using online network
            next_q_values = self.q_network(next_states)
            next_actions = next_q_values.argmax(dim=1)
            
            # Evaluate using target network
            next_q_target = self.target_network(next_states)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Compute target with safety penalty (reduced penalty)
            safety_penalty = costs * 0.3  # Reduced from 0.5
            target_q = rewards - safety_penalty + self.gamma * next_q * (1 - dones)
        
        # TD loss (Huber loss for stability)
        td_loss = F.smooth_l1_loss(q_current, target_q)
        
        return td_loss, q_values
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one update step on a batch of data.
        
        Args:
            batch: Dictionary with states, actions, rewards, next_states, dones, costs
            
        Returns:
            Dictionary of loss values
        """
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        costs = batch['costs'].to(self.device)
        
        # Compute TD loss
        td_loss, q_values = self.compute_td_loss(
            states, actions, rewards, next_states, dones, costs
        )
        
        # Compute CQL loss
        cql_loss, action_gap = self.compute_cql_loss(states, actions, q_values)
        
        # Get current alpha (from Lagrange or fixed)
        if self.use_lagrange:
            current_alpha = torch.clamp(self.log_alpha.exp(), min=0.01, max=10.0)
        else:
            current_alpha = self.alpha
        
        # Total loss
        total_loss = td_loss + current_alpha * cql_loss
        
        # Optimize Q-network
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update Lagrange multiplier (automatic alpha tuning)
        if self.use_lagrange:
            alpha_loss = -self.log_alpha * (action_gap - self.target_action_gap).detach()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # Soft update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self._soft_update_target()
        
        # Track metrics
        alpha_value = current_alpha.item() if isinstance(current_alpha, torch.Tensor) else current_alpha
        metrics = {
            'td_loss': td_loss.item(),
            'cql_loss': cql_loss.item(),
            'total_loss': total_loss.item(),
            'mean_q': q_values.mean().item(),
            'max_q': q_values.max().item(),
            'min_q': q_values.min().item(),
            'alpha': alpha_value,
            'action_gap': action_gap.item(),
            'cost_rate': (costs > 0).float().mean().item()
        }
        
        # Update history
        self.training_history['q_loss'].append(td_loss.item())
        self.training_history['cql_loss'].append(cql_loss.item())
        self.training_history['total_loss'].append(total_loss.item())
        self.training_history['mean_q'].append(q_values.mean().item())
        self.training_history['alpha'].append(alpha_value)
        
        # Decay temperature
        self.decay_temperature()
        
        return metrics
    
    def _soft_update_target(self):
        """Soft update target network parameters."""
        for target_param, param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def train_offline(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 100,
        eval_fn: Optional[callable] = None,
        eval_freq: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the agent offline on a fixed dataset.
        
        Args:
            dataloader: DataLoader for training data
            epochs: Number of training epochs
            eval_fn: Optional evaluation function
            eval_freq: Frequency of evaluation
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        self.q_network.train()
        
        for epoch in range(epochs):
            epoch_metrics = {
                'td_loss': [],
                'cql_loss': [],
                'total_loss': [],
                'mean_q': [],
                'alpha': []
            }
            
            for batch in dataloader:
                # Convert batch format
                batch_dict = {
                    'states': batch['state'],
                    'actions': batch['action'].squeeze(-1),
                    'rewards': batch['reward'].squeeze(-1),
                    'next_states': batch['next_state'],
                    'dones': batch['done'].squeeze(-1),
                    'costs': batch['cost'].squeeze(-1)
                }
                
                metrics = self.update(batch_dict)
                
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key].append(metrics[key])
            
            # Log progress
            if verbose and (epoch + 1) % eval_freq == 0:
                avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"TD Loss: {avg_metrics['td_loss']:.4f} | "
                      f"CQL Loss: {avg_metrics['cql_loss']:.4f} | "
                      f"Mean Q: {avg_metrics['mean_q']:.4f} | "
                      f"Alpha: {avg_metrics['alpha']:.4f}")
                
                # Run evaluation if provided
                if eval_fn is not None:
                    eval_metrics = eval_fn(self)
                    print(f"  Eval: {eval_metrics}")
        
        return self.training_history
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        use_safety_shield: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the agent on a dataset.
        
        Returns metrics including:
        - Average Q-value
        - Action distribution
        - Safety constraint violations
        """
        self.q_network.eval()
        
        all_q_values = []
        all_actions = []
        all_costs = []
        constraint_violations = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                states = batch['state'].to(self.device)
                
                q_values = self.q_network(states)
                
                if use_safety_shield:
                    q_values = self.safety.mask_unsafe_actions(
                        q_values, states,
                        denormalize_fn=self.denormalize_state,
                        mask_value=-100  # Reduced from -1e9
                    )
                
                actions = q_values.argmax(dim=1)
                
                all_q_values.append(q_values.cpu())
                all_actions.append(actions.cpu())
                
                # Check for constraint violations
                for i in range(states.shape[0]):
                    state = states[i].cpu().numpy()
                    action = actions[i].item()
                    if not self.safety.is_action_safe(state, action, self.denormalize_state):
                        constraint_violations += 1
                    total_samples += 1
        
        all_q_values = torch.cat(all_q_values, dim=0)
        all_actions = torch.cat(all_actions, dim=0)
        
        # Compute action distribution
        action_counts = torch.bincount(all_actions, minlength=self.action_dim)
        action_dist = action_counts.float() / len(all_actions)
        
        # Filter out masked values for mean calculation
        valid_q_mask = all_q_values > -50  # Exclude masked values
        valid_q_values = all_q_values[valid_q_mask]
        
        metrics = {
            'mean_q': valid_q_values.mean().item() if len(valid_q_values) > 0 else 0,
            'max_q': valid_q_values.max().item() if len(valid_q_values) > 0 else 0,
            'min_q': valid_q_values.min().item() if len(valid_q_values) > 0 else 0,
            'constraint_violation_rate': constraint_violations / max(total_samples, 1),
            'action_distribution': action_dist.tolist()
        }
        
        self.q_network.train()
        return metrics
    
    def save(self, filepath: str):
        """Save agent state."""
        save_dict = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_step': self.train_step,
            'training_history': self.training_history,
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'tau': self.tau,
                'alpha': self.alpha,
                'min_q_weight': self.min_q_weight
            }
        }
        if self.use_lagrange:
            save_dict['log_alpha'] = self.log_alpha
        
        torch.save(save_dict, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_step = checkpoint['train_step']
        self.training_history = checkpoint['training_history']
        if checkpoint['state_mean'] is not None:
            self.state_mean = checkpoint['state_mean'].to(self.device)
            self.state_std = checkpoint['state_std'].to(self.device)
        if 'log_alpha' in checkpoint and self.use_lagrange:
            self.log_alpha = checkpoint['log_alpha'].to(self.device)
