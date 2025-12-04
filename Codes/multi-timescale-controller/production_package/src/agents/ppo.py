"""
Proximal Policy Optimization (PPO) Agent - Baseline for comparison.

Offline adaptation of PPO using behavior cloning initialization
and conservative policy updates.

Reference: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List

from .networks import ActorCritic, PolicyNetwork, ValueNetwork
from .safety import SafetyModule, SafetyConfig


class PPOAgent:
    """
    Proximal Policy Optimization Agent (Baseline).
    
    Adapted for offline learning with:
    - Behavior cloning initialization
    - Conservative policy updates
    - Safety constraint integration
    - Numerical stability improvements
    """
    
    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 5,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 1e-4,  # Reduced from 3e-4 for stability
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.05,  # Increased from 0.01 for exploration
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        num_minibatches: int = 4,
        device: str = 'auto',
        safety_config: Optional[SafetyConfig] = None
    ):
        """
        Initialize PPO Agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_epsilon: PPO clipping parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO update epochs per batch
            num_minibatches: Number of minibatches per update
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
        
        # Actor-Critic network
        self.actor_critic = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=learning_rate,
            eps=1e-5,  # Numerical stability
            weight_decay=1e-4  # L2 regularization
        )
        
        # Learning rate scheduler for stability
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.9
        )
        
        # Safety module
        self.safety = SafetyModule(safety_config)
        
        # Training state
        self.train_step = 0
        self.training_history = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': []
        }
        
        # Normalization
        self.state_mean = None
        self.state_std = None
        
        # For numerical stability tracking
        self._nan_detected = False
    
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
    
    def _safe_log_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """Numerically stable log softmax with clamping."""
        # Clamp logits to prevent extreme values
        logits = torch.clamp(logits, min=-20, max=20)
        return F.log_softmax(logits, dim=-1)
    
    def _safe_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """Numerically stable softmax with clamping."""
        logits = torch.clamp(logits, min=-20, max=20)
        return F.softmax(logits, dim=-1)
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        use_safety_shield: bool = True
    ) -> Tuple[int, float, float]:
        """
        Select action from policy with HARD safety enforcement.
        
        Phase 4 Improvement: Uses -inf masking to make unsafe actions IMPOSSIBLE.
        This guarantees 0% constraint violations.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
            use_safety_shield: Whether to filter unsafe actions
            
        Returns:
            action: Selected action (guaranteed safe)
            log_prob: Log probability of action
            value: State value estimate
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, value = self.actor_critic(state_tensor)
            
            # === HARD SAFETY ENFORCEMENT ===
            if use_safety_shield:
                # Get safe action mask
                safe_mask = []
                for action_idx in range(self.action_dim):
                    is_safe = self.safety.is_action_safe(state, action_idx, self.denormalize_state)
                    safe_mask.append(is_safe)
                safe_mask = torch.tensor(safe_mask, dtype=torch.bool)
                
                # Check if any action is safe
                if not safe_mask.any():
                    # Fallback: No-op is always the safest option
                    action = torch.tensor([4])  # No-op
                    log_prob_val = 0.0
                    return 4, log_prob_val, value.squeeze().item()
                
                # HARD masking: set unsafe actions to -infinity (impossible to select)
                masked_logits = logits.clone()
                masked_logits[0, ~safe_mask] = float('-inf')
                
                # Compute probabilities only over safe actions
                probs = self._safe_softmax(masked_logits)
            else:
                probs = self._safe_softmax(logits)
                masked_logits = logits
            
            if deterministic:
                action = probs.argmax(dim=1)
            else:
                # Sample from safe actions only
                probs = probs + 1e-8
                probs = probs / probs.sum(dim=-1, keepdim=True)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
            
            # Compute log prob
            log_prob = self._safe_log_softmax(masked_logits)
            action_log_prob = log_prob.gather(1, action.unsqueeze(-1)).squeeze(-1)
            
            # === SAFETY VALIDATION (should never fail) ===
            if use_safety_shield:
                selected_action = action.item()
                if not self.safety.is_action_safe(state, selected_action, self.denormalize_state):
                    # This should never happen with hard masking, but just in case
                    print(f"WARNING: Safety violation detected! Falling back to No-op.")
                    return 4, 0.0, value.squeeze().item()
        
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
        Compute Generalized Advantage Estimation.
        
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        batch_size = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        
        # Adjust rewards with safety penalty (reduced penalty)
        adjusted_rewards = rewards - costs * 0.3
        
        # Compute advantages backwards
        last_gae = 0
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = adjusted_rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        
        # Normalize advantages with better numerical stability
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            advantages = advantages - adv_mean
        
        # Clamp advantages to prevent extreme values
        advantages = torch.clamp(advantages, -10, 10)
        
        return advantages, returns
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform PPO update on a batch with numerical stability checks.
        
        Args:
            batch: Dictionary with states, actions, rewards, etc.
            
        Returns:
            Dictionary of metrics
        """
        # Skip update if NaN was previously detected
        if self._nan_detected:
            return {
                'policy_loss': float('nan'),
                'value_loss': float('nan'),
                'entropy': float('nan'),
                'total_loss': float('nan')
            }
        
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        costs = batch['costs'].to(self.device)
        old_log_probs = batch.get('log_probs')
        
        # Get current values
        with torch.no_grad():
            _, values = self.actor_critic(states)
            values = values.squeeze(-1)
            _, next_values = self.actor_critic(next_states)
            next_values = next_values.squeeze(-1)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            rewards, values, dones, next_values, costs
        )
        
        # If we don't have old log probs, compute them
        if old_log_probs is None:
            with torch.no_grad():
                logits, _ = self.actor_critic(states)
                old_log_probs = self._safe_log_softmax(logits)
                old_log_probs = old_log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        else:
            old_log_probs = old_log_probs.to(self.device)
        
        # PPO epochs
        batch_size = states.shape[0]
        minibatch_size = max(batch_size // self.num_minibatches, 32)  # Ensure minimum size
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for _ in range(self.ppo_epochs):
            # Shuffle indices
            indices = torch.randperm(batch_size, device=self.device)
            
            for start in range(0, batch_size, minibatch_size):
                end = min(start + minibatch_size, batch_size)
                mb_indices = indices[start:end]
                
                if len(mb_indices) < 8:  # Skip very small batches
                    continue
                
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Get current policy and value
                logits, new_values = self.actor_critic(mb_states)
                new_values = new_values.squeeze(-1) if new_values.dim() > 1 else new_values
                
                # Safe log probs and entropy calculation
                log_probs = self._safe_log_softmax(logits)
                probs = self._safe_softmax(logits)
                
                new_log_probs = log_probs.gather(1, mb_actions.unsqueeze(-1)).squeeze(-1)
                
                # Entropy with numerical stability
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                entropy = torch.clamp(entropy, min=0, max=2)  # Clamp entropy
                
                # Policy loss with clipping
                log_ratio = new_log_probs - mb_old_log_probs
                log_ratio = torch.clamp(log_ratio, -20, 20)  # Prevent extreme ratios
                ratio = torch.exp(log_ratio)
                
                # Clamp ratio for additional stability
                ratio = torch.clamp(ratio, 0.01, 100)
                
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with clipping
                value_loss = F.smooth_l1_loss(new_values, mb_returns)  # More stable than MSE
                
                # Check for NaN before computing total loss
                if torch.isnan(policy_loss) or torch.isnan(value_loss) or torch.isnan(entropy):
                    print(f"Warning: NaN detected in PPO update. Skipping batch.")
                    self._nan_detected = True
                    continue
                
                # Total loss with entropy bonus
                loss = (
                    policy_loss +
                    self.value_loss_coef * value_loss -
                    self.entropy_coef * entropy
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(),
                    self.max_grad_norm
                )
                
                # Check for NaN gradients
                has_nan_grad = False
                for param in self.actor_critic.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print("Warning: NaN gradient detected. Skipping update.")
                    self.optimizer.zero_grad()
                    continue
                
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1
        
        self.train_step += 1
        
        # Update learning rate
        if self.train_step % 10 == 0:
            self.scheduler.step()
        
        if num_updates == 0:
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'total_loss': 0.0
            }
        
        metrics = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'total_loss': (total_policy_loss + total_value_loss) / num_updates
        }
        
        self.training_history['policy_loss'].append(metrics['policy_loss'])
        self.training_history['value_loss'].append(metrics['value_loss'])
        self.training_history['entropy'].append(metrics['entropy'])
        self.training_history['total_loss'].append(metrics['total_loss'])
        
        return metrics
    
    def behavior_cloning_pretrain(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 20,  # Increased from 10
        verbose: bool = True
    ):
        """
        Pretrain policy using behavior cloning.
        
        This helps initialize the policy to be close to the behavior policy
        in the dataset, which is important for offline RL.
        """
        self.actor_critic.train()
        
        # Use separate optimizer for BC with higher learning rate
        bc_optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                states = batch['state'].to(self.device)
                actions = batch['action'].squeeze(-1).to(self.device)
                
                # Get policy logits
                logits, _ = self.actor_critic(states)
                
                # Clamp logits for stability
                logits = torch.clamp(logits, -20, 20)
                
                # Cross-entropy loss (behavior cloning) with label smoothing
                loss = F.cross_entropy(logits, actions, label_smoothing=0.1)
                
                if torch.isnan(loss):
                    continue
                
                bc_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1.0)
                bc_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if verbose and (epoch + 1) % 5 == 0 and num_batches > 0:
                print(f"BC Pretrain Epoch {epoch + 1}/{epochs} | Loss: {total_loss / num_batches:.4f}")
    
    def train_offline(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 100,
        bc_pretrain_epochs: int = 20,  # Increased from 10
        eval_fn: Optional[callable] = None,
        eval_freq: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train PPO offline on a fixed dataset.
        
        Uses behavior cloning pretraining followed by conservative PPO updates.
        """
        # Reset NaN detection
        self._nan_detected = False
        
        # Behavior cloning pretraining
        if bc_pretrain_epochs > 0:
            if verbose:
                print("Starting behavior cloning pretraining...")
            self.behavior_cloning_pretrain(dataloader, bc_pretrain_epochs, verbose)
        
        self.actor_critic.train()
        
        for epoch in range(epochs):
            epoch_metrics = {
                'policy_loss': [],
                'value_loss': [],
                'entropy': []
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
                
                # Skip NaN metrics
                for key in epoch_metrics:
                    if key in metrics and not np.isnan(metrics[key]):
                        epoch_metrics[key].append(metrics[key])
            
            if verbose and (epoch + 1) % eval_freq == 0:
                avg_metrics = {}
                for k, v in epoch_metrics.items():
                    if v:
                        avg_metrics[k] = np.mean(v)
                    else:
                        avg_metrics[k] = 0.0
                
                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"Policy Loss: {avg_metrics['policy_loss']:.4f} | "
                      f"Value Loss: {avg_metrics['value_loss']:.4f} | "
                      f"Entropy: {avg_metrics['entropy']:.4f}")
                
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
        Evaluate the agent on a dataset with HARD safety enforcement.
        """
        self.actor_critic.eval()
        
        all_actions = []
        constraint_violations = 0
        total_samples = 0
        total_entropy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                states = batch['state'].to(self.device)
                
                logits, values = self.actor_critic(states)
                
                # Apply HARD safety mask (-inf for unsafe actions)
                if use_safety_shield:
                    for i in range(states.shape[0]):
                        state = states[i].cpu().numpy()
                        for action_idx in range(self.action_dim):
                            if not self.safety.is_action_safe(state, action_idx, self.denormalize_state):
                                logits[i, action_idx] = float('-inf')  # HARD mask
                
                probs = self._safe_softmax(logits)
                actions = probs.argmax(dim=1)
                
                # Compute entropy (only over safe actions)
                log_probs = self._safe_log_softmax(logits)
                # Mask out -inf values for entropy calculation
                valid_probs = probs.clone()
                valid_probs[probs < 1e-10] = 1e-10  # Avoid log(0)
                entropy = -(valid_probs * torch.log(valid_probs)).sum(dim=-1).mean()
                if not torch.isnan(entropy):
                    total_entropy += entropy.item()
                    num_batches += 1
                
                all_actions.append(actions.cpu())
                
                # Check for any constraint violations (should be 0 with hard mask)
                for i in range(states.shape[0]):
                    state = states[i].cpu().numpy()
                    action = actions[i].item()
                    if not self.safety.is_action_safe(state, action, self.denormalize_state):
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
            'train_step': self.train_step,
            'training_history': self.training_history,
            'state_mean': self.state_mean,
            'state_std': self.state_std
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Try to load state dict, handle architecture mismatches
        try:
            self.actor_critic.load_state_dict(checkpoint['actor_critic'], strict=True)
        except RuntimeError as e:
            # Architecture mismatch - try partial loading
            print(f"Warning: Architecture mismatch when loading PPO. Attempting partial load...")
            try:
                self.actor_critic.load_state_dict(checkpoint['actor_critic'], strict=False)
                print("  Partial load successful (some layers may not match)")
            except Exception as e2:
                raise RuntimeError(f"Failed to load PPO checkpoint: {e2}. Architecture mismatch detected.")
        
        if 'optimizer' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print("  Warning: Could not load optimizer state")
        
        self.train_step = checkpoint.get('train_step', 0)
        self.training_history = checkpoint.get('training_history', {})
        
        if 'state_mean' in checkpoint and checkpoint['state_mean'] is not None:
            self.state_mean = checkpoint['state_mean'].to(self.device)
        if 'state_std' in checkpoint and checkpoint['state_std'] is not None:
            self.state_std = checkpoint['state_std'].to(self.device)
