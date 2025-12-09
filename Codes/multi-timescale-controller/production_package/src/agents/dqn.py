"""
Deep Q-Network (DQN) Agent - Baseline for comparison.

Standard DQN implementation with:
- Experience replay
- Target network
- Epsilon-greedy exploration
- Safety constraint integration

Reference: Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List
from copy import deepcopy

from .networks import QNetwork, DuelingQNetwork
from .dataset import ReplayBuffer
from .safety import SafetyModule, SafetyConfig


class DQNAgent:
    """
    Deep Q-Network Agent (Baseline).
    
    Standard DQN with experience replay and target network.
    Used as a baseline to compare against CQL.
    """
    
    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 5,
        hidden_dims: List[int] = [256, 256, 128],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        buffer_size: int = 10000,
        use_dueling: bool = False,
        use_double: bool = True,
        device: str = 'auto',
        safety_config: Optional[SafetyConfig] = None
    ):
        """
        Initialize DQN Agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate per step
            target_update_freq: Steps between target updates
            buffer_size: Replay buffer capacity
            use_dueling: Use Dueling DQN architecture
            use_double: Use Double DQN for target computation
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
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.use_double = use_double
        
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
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer with learning rate scheduling
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.95
        )
        self.initial_lr = learning_rate
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size,
            state_dim=state_dim,
            device=self.device
        )
        
        # Safety module
        self.safety = SafetyModule(safety_config)
        
        # Training state
        self.train_step = 0
        self.training_history = {
            'loss': [],
            'mean_q': [],
            'epsilon': [],
            'constraint_violations': []
        }
        
        # Normalization
        self.state_mean = None
        self.state_std = None
    
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
        use_epsilon: bool = True,
        use_safety_shield: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state (normalized)
            use_epsilon: Whether to use epsilon-greedy
            use_safety_shield: Whether to filter unsafe actions
            
        Returns:
            Selected action index
        """
        if use_epsilon and np.random.random() < self.epsilon:
            if use_safety_shield:
                safe_actions = self.safety.get_safe_actions(
                    state, None
                )
                if safe_actions:
                    return np.random.choice(safe_actions)
            return np.random.randint(0, self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
            if use_safety_shield:
                q_values = self.safety.mask_unsafe_actions(
                    q_values, state_tensor,
                    denormalize_fn=None
                )
            
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one update step.
        
        IMPORTANT FIX: Do NOT apply safety masking during Q-learning.
        Masking is only for action selection, not for learning true Q-values.
        This prevents Q-value explosion (the -1e9 mask was corrupting learning).
        
        Args:
            batch: Dictionary with transition data
            
        Returns:
            Dictionary of metrics
        """
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        costs = batch['costs'].to(self.device)
        
        # Current Q-values (NO masking during learning)
        q_values = self.q_network(states)
        q_current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values (NO masking - learn true values)
        with torch.no_grad():
            if self.use_double:
                # Double DQN: select action with online, evaluate with target
                # FIX: Don't mask next_actions - we want to learn true Q-values
                next_q_online = self.q_network(next_states)
                next_actions = next_q_online.argmax(dim=1)
                next_q = self.target_network(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                # Standard DQN
                next_q = self.target_network(next_states).max(dim=1)[0]
            
            # Include safety penalty in target (reduced from 0.5)
            safety_penalty = costs * 0.3
            target_q = rewards.squeeze() - safety_penalty + self.gamma * next_q * (1 - dones.squeeze())
        
        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(q_current, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self._soft_update_target()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Step learning rate scheduler
        if self.train_step % 100 == 0:
            self.scheduler.step()
        
        metrics = {
            'loss': loss.item(),
            'mean_q': q_values.mean().item(),
            'max_q': q_values.max().item(),
            'epsilon': self.epsilon,
            'cost_rate': (costs > 0).float().mean().item()
        }
        
        self.training_history['loss'].append(loss.item())
        self.training_history['mean_q'].append(q_values.mean().item())
        self.training_history['epsilon'].append(self.epsilon)
        
        return metrics
    
    def _soft_update_target(self):
        """Soft update target network."""
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
        
        For DQN, we populate the replay buffer from the dataset
        and sample from it during training.
        """
        self.q_network.train()
        
        # Populate replay buffer from dataset
        for batch in dataloader:
            for i in range(batch['state'].shape[0]):
                self.replay_buffer.add(
                    state=batch['state'][i].numpy(),
                    action=batch['action'][i].item(),
                    reward=batch['reward'][i].item(),
                    next_state=batch['next_state'][i].numpy(),
                    done=batch['done'][i].item() > 0.5,
                    cost=batch['cost'][i].item()
                )
        
        batch_size = dataloader.batch_size
        
        for epoch in range(epochs):
            epoch_metrics = {'loss': [], 'mean_q': []}
            
            # Number of updates per epoch
            num_updates = len(self.replay_buffer) // batch_size
            
            for _ in range(num_updates):
                batch = self.replay_buffer.sample(batch_size)
                metrics = self.update(batch)
                
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key].append(metrics[key])
            
            if verbose and (epoch + 1) % eval_freq == 0:
                avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"Loss: {avg_metrics['loss']:.4f} | "
                      f"Mean Q: {avg_metrics['mean_q']:.4f} | "
                      f"Epsilon: {self.epsilon:.4f}")
                
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
        
        FIX: Report UNMASKED Q-values for accurate metrics.
        Masking is only for action selection.
        """
        self.q_network.eval()
        
        all_q_values = []  # Store unmasked Q-values for metrics
        all_actions = []
        constraint_violations = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                states = batch['state'].to(self.device)
                
                # Get unmasked Q-values for metrics
                q_values_raw = self.q_network(states)
                all_q_values.append(q_values_raw.cpu())
                
                # Apply safety mask only for action selection
                if use_safety_shield:
                    q_values_masked = self.safety.mask_unsafe_actions(
                        q_values_raw, states,
                        denormalize_fn=self.denormalize_state,
                        mask_value=-100  # Finite mask value
                    )
                    actions = q_values_masked.argmax(dim=1)
                else:
                    actions = q_values_raw.argmax(dim=1)
                
                all_actions.append(actions.cpu())
                
                for i in range(states.shape[0]):
                    state = states[i].cpu().numpy()
                    action = actions[i].item()
                    if not self.safety.is_action_safe(state, action, self.denormalize_state):
                        constraint_violations += 1
                    total_samples += 1
        
        all_q_values = torch.cat(all_q_values, dim=0)
        all_actions = torch.cat(all_actions, dim=0)
        
        action_counts = torch.bincount(all_actions, minlength=self.action_dim)
        action_dist = action_counts.float() / len(all_actions)
        
        # Report unmasked Q-values (true learned values)
        metrics = {
            'mean_q': all_q_values.mean().item(),
            'max_q': all_q_values.max().item(),
            'min_q': all_q_values.min().item(),
            'constraint_violation_rate': constraint_violations / max(total_samples, 1),
            'action_distribution': action_dist.tolist()
        }
        
        self.q_network.train()
        return metrics
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_step': self.train_step,
            'epsilon': self.epsilon,
            'training_history': self.training_history,
            'state_mean': self.state_mean,
            'state_std': self.state_std
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_step = checkpoint['train_step']
        self.epsilon = checkpoint['epsilon']
        self.training_history = checkpoint['training_history']
        if checkpoint['state_mean'] is not None:
            self.state_mean = checkpoint['state_mean'].to(self.device)
            self.state_std = checkpoint['state_std'].to(self.device)

