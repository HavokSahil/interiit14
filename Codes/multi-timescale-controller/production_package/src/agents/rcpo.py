"""
Reward-Constrained Policy Optimization (RCPO) Agent for Safe RL.

RCPO is an alternative to CQL mentioned in the problem statement.
It optimizes rewards while satisfying safety constraints using Lagrange multipliers.

Reference: Tessler et al., "Reward Constrained Policy Optimization", ICLR 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List

from .networks import ActorCritic
from .safety import SafetyModule, SafetyConfig


class RCPOAgent:
    """
    Reward-Constrained Policy Optimization Agent.
    
    Key features:
    1. Policy-based method (like PPO) with constraint handling
    2. Lagrange multiplier for automatic constraint balancing
    3. Offline learning from fixed dataset
    4. Safety constraint integration via Lagrangian
    5. Less conservative than CQL while maintaining safety
    """
    
    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 5,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.1,  # Higher entropy for exploration
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        num_minibatches: int = 4,
        # RCPO-specific parameters
        constraint_threshold: float = 0.1,  # Max expected cost
        lambda_lr: float = 1e-3,  # Lagrange multiplier learning rate
        lambda_init: float = 0.1,  # Initial Lagrange multiplier
        lambda_max: float = 10.0,  # Max Lagrange multiplier
        device: str = 'auto',
        safety_config: Optional[SafetyConfig] = None
    ):
        """
        Initialize RCPO Agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for policy
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_epsilon: PPO clipping parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO update epochs per batch
            num_minibatches: Number of minibatches per update
            constraint_threshold: Maximum allowed expected cost
            lambda_lr: Learning rate for Lagrange multiplier
            lambda_init: Initial value for Lagrange multiplier
            lambda_max: Maximum value for Lagrange multiplier
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
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches
        
        # RCPO-specific
        self.constraint_threshold = constraint_threshold
        self.lambda_lr = lambda_lr
        self.lambda_max = lambda_max
        
        # Lagrange multiplier (learnable parameter)
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))
        
        # Actor-Critic network
        self.actor_critic = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # Optimizer for policy (includes lambda)
        self.optimizer = optim.Adam(
            list(self.actor_critic.parameters()) + [self.lambda_param],
            lr=learning_rate,
            eps=1e-5,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        # Safety module
        if safety_config is None:
            safety_config = SafetyConfig()
        self.safety = SafetyModule(safety_config)
        
        # Normalization
        self.state_mean = None
        self.state_std = None
        
        # Training history
        self.training_history = {
            'policy_loss': [],
            'value_loss': [],
            'constraint_loss': [],
            'entropy': [],
            'lambda_value': [],
            'expected_cost': []
        }
        
        self.train_step = 0
    
    def set_normalization(self, mean, std):
        """Set state normalization parameters."""
        if mean is not None:
            if isinstance(mean, np.ndarray):
                self.state_mean = torch.FloatTensor(mean).to(self.device)
            else:
                self.state_mean = mean.to(self.device) if hasattr(mean, 'to') else torch.FloatTensor(mean).to(self.device)
        else:
            self.state_mean = None
        
        if std is not None:
            if isinstance(std, np.ndarray):
                self.state_std = torch.FloatTensor(std).to(self.device)
            else:
                self.state_std = std.to(self.device) if hasattr(std, 'to') else torch.FloatTensor(std).to(self.device)
        else:
            self.state_std = None
    
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state."""
        if self.state_mean is not None and self.state_std is not None:
            return (state - self.state_mean) / (self.state_std + 1e-8)
        return state
    
    def denormalize_state(self, state: np.ndarray) -> np.ndarray:
        """Denormalize state for safety checks."""
        if self.state_mean is not None and self.state_std is not None:
            mean_np = self.state_mean.cpu().numpy()
            std_np = self.state_std.cpu().numpy()
            return state * std_np + mean_np
        return state
    
    def _safe_log_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """Safe log softmax with clamping."""
        logits = torch.clamp(logits, min=-20, max=20)
        return F.log_softmax(logits, dim=-1)
    
    def _safe_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """Safe softmax with clamping."""
        logits = torch.clamp(logits, min=-20, max=20)
        return F.softmax(logits, dim=-1)
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        use_safety_shield: bool = True
    ) -> Tuple[int, float, float]:
        """
        Select action from policy.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
            use_safety_shield: Whether to filter unsafe actions
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            state_tensor = self.normalize_state(state_tensor)
            logits, value = self.actor_critic(state_tensor)
            
            # Apply safety mask to logits
            if use_safety_shield:
                for action_idx in range(self.action_dim):
                    if not self.safety.is_action_safe(state, action_idx, self.denormalize_state):
                        logits[0, action_idx] = -20  # Mask unsafe actions
            
            probs = self._safe_softmax(logits)
            
            if deterministic:
                action = probs.argmax(dim=1)
            else:
                probs = probs + 1e-8
                probs = probs / probs.sum(dim=-1, keepdim=True)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
            
            log_prob = self._safe_log_softmax(logits)
            action_log_prob = log_prob.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        return action.item(), action_log_prob.item(), value.squeeze().item()
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor,
        costs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE (Generalized Advantage Estimation) for rewards and costs.
        
        Returns:
            advantages: Advantage estimates
            returns: Return estimates
            cost_advantages: Cost advantage estimates
            cost_returns: Cost return estimates
        """
        batch_size = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        cost_advantages = torch.zeros_like(costs)
        cost_returns = torch.zeros_like(costs)
        
        last_gae = 0
        last_cost_gae = 0
        
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_value = next_values[t] if t < len(next_values) else 0
                next_cost_value = 0  # Cost value is 0 at terminal
            else:
                next_value = values[t + 1]
                next_cost_value = 0  # Simplified: cost is immediate
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            
            returns[t] = advantages[t] + values[t]
            
            # Cost advantage (simplified - costs are immediate)
            cost_delta = costs[t] - 0  # No cost value function
            cost_advantages[t] = last_cost_gae = cost_delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_cost_gae
            cost_returns[t] = cost_advantages[t]
        
        return advantages, returns, cost_advantages, cost_returns
    
    def update(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Update policy using RCPO.
        
        RCPO optimizes: max E[R] - λ * (E[C] - threshold)
        where λ is the Lagrange multiplier.
        """
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        costs = batch['costs'].to(self.device)
        old_log_probs = batch.get('old_log_probs', None)
        
        # Normalize states
        states = self.normalize_state(states)
        next_states = self.normalize_state(next_states)
        
        # Get current policy outputs
        logits, values = self.actor_critic(states)
        _, next_values = self.actor_critic(next_states)
        
        # Compute advantages and returns
        # Note: compute_gae returns (advantages, returns, cost_advantages, cost_returns)
        gae_results = self.compute_gae(
            rewards, values.squeeze(-1), dones, next_values.squeeze(-1), costs
        )
        advantages, returns, cost_advantages, cost_returns = gae_results
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)
        
        # Get current log probs
        log_probs = self._safe_log_softmax(logits)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Compute policy loss (PPO-style)
        if old_log_probs is not None:
            old_log_probs = old_log_probs.to(self.device)
            ratio = torch.exp(action_log_probs - old_log_probs)
        else:
            ratio = torch.ones_like(action_log_probs)
        
        # Clamp ratio for numerical stability
        ratio = torch.clamp(ratio, 0.01, 100)
        
        # RCPO objective: maximize reward - λ * (cost - threshold)
        lambda_val = torch.clamp(self.lambda_param, 0, self.lambda_max)
        
        # Policy loss: - (advantage - λ * cost_advantage)
        policy_loss1 = -ratio * (advantages - lambda_val * cost_advantages)
        policy_loss2 = -torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * (advantages - lambda_val * cost_advantages)
        policy_loss = torch.max(policy_loss1, policy_loss2).mean()
        
        # Value loss
        value_loss = F.smooth_l1_loss(values.squeeze(-1), returns)
        
        # Entropy bonus
        probs = self._safe_softmax(logits)
        log_probs_full = self._safe_log_softmax(logits)
        entropy = -(probs * log_probs_full).sum(dim=-1).mean()
        
        # Total loss
        total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        
        # Constraint loss (for tracking)
        expected_cost = costs.mean()
        constraint_loss = lambda_val * (expected_cost - self.constraint_threshold)
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Update Lagrange multiplier (dual gradient ascent)
        with torch.no_grad():
            constraint_violation = expected_cost - self.constraint_threshold
            lambda_update = self.lambda_param + self.lambda_lr * constraint_violation
            self.lambda_param.data = torch.clamp(lambda_update, 0, self.lambda_max)
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'constraint_loss': constraint_loss.item(),
            'entropy': entropy.item(),
            'lambda_value': lambda_val.item(),
            'expected_cost': expected_cost.item()
        }
        
        return metrics
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        use_safety_shield: bool = True
    ) -> Dict[str, float]:
        """Evaluate the agent on a dataset."""
        self.actor_critic.eval()
        
        all_actions = []
        constraint_violations = 0
        total_samples = 0
        total_entropy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                states = batch['state'].to(self.device)
                states = self.normalize_state(states)
                
                logits, values = self.actor_critic(states)
                
                # Apply safety shield
                if use_safety_shield:
                    for i in range(states.shape[0]):
                        state_np = batch['state'][i].numpy()
                        for action_idx in range(self.action_dim):
                            if not self.safety.is_action_safe(state_np, action_idx, self.denormalize_state):
                                logits[i, action_idx] = -20
                
                probs = self._safe_softmax(logits)
                log_probs = self._safe_log_softmax(logits)
                
                # Compute entropy
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                total_entropy += entropy.item()
                num_batches += 1
                
                # Sample actions
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
                all_actions.append(actions)
                
                # Check for violations
                for i in range(states.shape[0]):
                    state_np = batch['state'][i].numpy()
                    action = actions[i].item()
                    cost = batch['cost'][i].item()
                    
                    if cost > 0.5:  # Hard constraint violation
                        constraint_violations += 1
                    total_samples += 1
        
        all_actions = torch.cat(all_actions, dim=0)
        
        action_counts = torch.bincount(all_actions, minlength=self.action_dim)
        action_dist = action_counts.float() / len(all_actions)
        
        metrics = {
            'mean_entropy': total_entropy / max(num_batches, 1),
            'constraint_violation_rate': constraint_violations / max(total_samples, 1),
            'action_distribution': action_dist.tolist()
        }
        
        self.actor_critic.train()
        return metrics
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lambda_param': self.lambda_param.data,
            'train_step': self.train_step,
            'training_history': self.training_history,
            'state_mean': self.state_mean,
            'state_std': self.state_std
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        try:
            self.actor_critic.load_state_dict(checkpoint['actor_critic'], strict=True)
        except RuntimeError as e:
            print(f"Warning: Architecture mismatch when loading RCPO. Attempting partial load...")
            try:
                self.actor_critic.load_state_dict(checkpoint['actor_critic'], strict=False)
                print("  Partial load successful (some layers may not match)")
            except Exception as e2:
                raise RuntimeError(f"Failed to load RCPO checkpoint: {e2}")
        
        if 'optimizer' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print("  Warning: Could not load optimizer state")
        
        if 'lambda_param' in checkpoint:
            self.lambda_param.data = checkpoint['lambda_param'].to(self.device)
        
        self.train_step = checkpoint.get('train_step', 0)
        self.training_history = checkpoint.get('training_history', {
            'policy_loss': [],
            'value_loss': [],
            'constraint_loss': [],
            'entropy': [],
            'lambda_value': [],
            'expected_cost': []
        })
        
        if 'state_mean' in checkpoint and checkpoint['state_mean'] is not None:
            self.state_mean = checkpoint['state_mean'].to(self.device)
        if 'state_std' in checkpoint and checkpoint['state_std'] is not None:
            self.state_std = checkpoint['state_std'].to(self.device)

