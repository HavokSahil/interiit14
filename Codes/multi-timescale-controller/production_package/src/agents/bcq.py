"""
Batch-Constrained Q-Learning (BCQ) Agent for Safe Offline RL.

BCQ is an offline RL algorithm that constrains the policy to only select actions
similar to those in the dataset. This is less conservative than CQL but still safe.

Key Features:
1. Generative model to identify in-distribution actions
2. Perturbation model to refine actions
3. Only selects actions within data support

Why BCQ for RRM?
- Less conservative than CQL (more action diversity)
- Still safe for offline learning
- Good when dataset has reasonable coverage

Reference: Fujimoto et al., "Off-Policy Deep Reinforcement Learning without Exploration", ICML 2019
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


class VAE(nn.Module):
    """
    Variational Autoencoder for BCQ.
    
    Used to model the distribution of actions in the dataset.
    Only actions that the VAE can reconstruct well are considered in-distribution.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # Encoder: (state, action) -> latent
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: (state, latent) -> action
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def encode(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode state-action pair to latent distribution."""
        x = torch.cat([state, action], dim=-1)
        h = self.encoder(x)
        mean = self.mean_layer(h)
        log_var = self.log_var_layer(h)
        return mean, log_var
    
    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to action logits."""
        x = torch.cat([state, latent], dim=-1)
        return self.decoder(x)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode then decode."""
        mean, log_var = self.encode(state, action)
        latent = self.reparameterize(mean, log_var)
        reconstructed = self.decode(state, latent)
        return reconstructed, mean, log_var
    
    def sample(self, state: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
        """Sample actions for a given state."""
        batch_size = state.shape[0]
        
        # Sample from prior
        latent = torch.randn(batch_size, num_samples, self.latent_dim, device=state.device)
        
        # Expand state for sampling
        state_expanded = state.unsqueeze(1).expand(-1, num_samples, -1)
        
        # Decode to actions
        state_flat = state_expanded.reshape(-1, self.state_dim)
        latent_flat = latent.reshape(-1, self.latent_dim)
        
        action_logits = self.decode(state_flat, latent_flat)
        action_logits = action_logits.reshape(batch_size, num_samples, self.action_dim)
        
        return action_logits


class BCQAgent:
    """
    Batch-Constrained Q-Learning Agent.
    
    BCQ constrains the policy to only select actions that are
    similar to those in the dataset, using a VAE to model the
    action distribution.
    
    Less conservative than CQL but still safe for offline RL.
    """
    
    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 5,
        hidden_dims: List[int] = [256, 256, 128],
        latent_dim: int = 32,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        threshold: float = 0.3,  # BCQ threshold for filtering actions
        num_samples: int = 10,  # Number of action samples for selection
        target_update_freq: int = 100,
        device: str = 'auto',
        safety_config: Optional[SafetyConfig] = None
    ):
        """
        Initialize BCQ Agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dims: Hidden layer dimensions for Q-network
            latent_dim: Latent dimension for VAE
            learning_rate: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            threshold: BCQ threshold for filtering actions (0-1)
            num_samples: Number of action samples for selection
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
        self.threshold = threshold
        self.num_samples = num_samples
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
        
        # VAE for action generation
        self.vae = VAE(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=latent_dim
        ).to(self.device)
        
        # Optimizers
        self.q_optimizer = optim.Adam(
            list(self.q_network1.parameters()) + list(self.q_network2.parameters()),
            lr=learning_rate
        )
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=learning_rate)
        
        # Safety module
        self.safety = SafetyModule(safety_config)
        
        # Training state
        self.train_step = 0
        self.training_history = {
            'q_loss': [],
            'vae_loss': [],
            'mean_q': [],
            'action_diversity': []
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
        use_safety_shield: bool = True
    ) -> int:
        """
        Select action using BCQ policy.
        
        1. Generate candidate actions using VAE
        2. Filter by BCQ threshold (in-distribution check)
        3. Select action with highest Q-value among candidates
        4. Apply safety shield
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            state_normalized = self.normalize_state(state_tensor)
            
            # Get Q-values for all actions
            q1 = self.q_network1(state_normalized)
            q2 = self.q_network2(state_normalized)
            q_values = torch.min(q1, q2).squeeze()
            
            # Generate action probabilities from VAE
            # Sample actions and compute reconstruction probability
            action_probs = self._get_action_probabilities(state_normalized)
            
            # BCQ filtering: only consider actions with probability above threshold
            max_prob = action_probs.max()
            bcq_mask = action_probs >= (self.threshold * max_prob)
            
            # Apply safety mask
            if use_safety_shield:
                safe_mask = torch.tensor([
                    self.safety.is_action_safe(state, a, self.denormalize_state)
                    for a in range(self.action_dim)
                ], dtype=torch.bool, device=self.device)
                
                # Combine BCQ mask with safety mask
                combined_mask = bcq_mask & safe_mask
                
                # If no actions pass both masks, fall back to safest action
                if not combined_mask.any():
                    combined_mask = safe_mask
                    if not combined_mask.any():
                        return 4  # No-op as last resort
            else:
                combined_mask = bcq_mask
            
            # Select action with highest Q-value among valid actions
            masked_q = q_values.clone()
            masked_q[~combined_mask] = -float('inf')
            action = masked_q.argmax().item()
        
        return action
    
    def _get_action_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get probability of each action being in-distribution.
        
        Uses VAE reconstruction to estimate action likelihood.
        """
        batch_size = state.shape[0]
        
        # For discrete actions, compute probability for each
        probs = torch.zeros(batch_size, self.action_dim, device=self.device)
        
        for action_idx in range(self.action_dim):
            # One-hot encode action
            action = torch.zeros(batch_size, self.action_dim, device=self.device)
            action[:, action_idx] = 1.0
            
            # Get reconstruction
            reconstructed, mean, log_var = self.vae(state, action)
            
            # Compute reconstruction probability (softmax of logits)
            recon_probs = F.softmax(reconstructed, dim=-1)
            probs[:, action_idx] = recon_probs[:, action_idx]
        
        return probs.squeeze()
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one BCQ update step.
        
        Updates:
        1. VAE for action generation
        2. Q-networks for value estimation
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
        
        # Convert actions to one-hot
        actions_onehot = F.one_hot(actions.long(), num_classes=self.action_dim).float()
        
        # ========== Update VAE ==========
        reconstructed, mean, log_var = self.vae(states, actions_onehot)
        
        # VAE loss: reconstruction + KL divergence
        recon_loss = F.cross_entropy(reconstructed, actions.long())
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        vae_loss = recon_loss + 0.5 * kl_loss
        
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
        self.vae_optimizer.step()
        
        # ========== Update Q-Networks ==========
        with torch.no_grad():
            # Get action probabilities for next states
            next_probs = []
            for i in range(next_states.shape[0]):
                prob = self._get_action_probabilities(next_states[i:i+1])
                next_probs.append(prob)
            next_probs = torch.stack(next_probs)
            
            # BCQ filtering for next actions
            max_probs = next_probs.max(dim=1, keepdim=True)[0]
            bcq_mask = next_probs >= (self.threshold * max_probs)
            
            # Get Q-values from target networks
            next_q1 = self.target_q1(next_states)
            next_q2 = self.target_q2(next_states)
            next_q = torch.min(next_q1, next_q2)
            
            # Mask out-of-distribution actions
            next_q[~bcq_mask] = -float('inf')
            
            # Select best action
            next_q_max = next_q.max(dim=1)[0]
            
            # Handle -inf values (when all actions are masked)
            next_q_max = torch.where(
                next_q_max == -float('inf'),
                torch.zeros_like(next_q_max),
                next_q_max
            )
            
            # Compute targets with safety penalty
            safety_penalty = costs * 0.3
            targets = rewards - safety_penalty + self.gamma * next_q_max * (1 - dones)
        
        # Current Q-values
        q1 = self.q_network1(states).gather(1, actions.unsqueeze(1).long()).squeeze()
        q2 = self.q_network2(states).gather(1, actions.unsqueeze(1).long()).squeeze()
        
        # Q-loss
        q_loss = F.smooth_l1_loss(q1, targets) + F.smooth_l1_loss(q2, targets)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q_network1.parameters()) + list(self.q_network2.parameters()),
            1.0
        )
        self.q_optimizer.step()
        
        # Soft update targets
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self._soft_update_targets()
        
        metrics = {
            'q_loss': q_loss.item(),
            'vae_loss': vae_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'mean_q': ((q1 + q2) / 2).mean().item(),
            'cost_rate': (costs > 0).float().mean().item()
        }
        
        self.training_history['q_loss'].append(q_loss.item())
        self.training_history['vae_loss'].append(vae_loss.item())
        self.training_history['mean_q'].append(metrics['mean_q'])
        
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
        self.vae.eval()
        
        all_q_values = []
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
                
                all_q_values.append(q_values.cpu())
                
                for i in range(states.shape[0]):
                    state = states[i].cpu().numpy()
                    action = self.select_action(state, use_safety_shield)
                    all_actions.append(action)
                    
                    if not self.safety.is_action_safe(state, action, self.denormalize_state):
                        constraint_violations += 1
                    total_samples += 1
        
        all_q_values = torch.cat(all_q_values, dim=0)
        all_actions = torch.tensor(all_actions)
        
        action_counts = torch.bincount(all_actions, minlength=self.action_dim)
        action_dist = action_counts.float() / len(all_actions)
        
        # Calculate action diversity (entropy-based)
        probs = action_dist + 1e-8
        diversity = -(probs * torch.log(probs)).sum().item()
        max_entropy = np.log(self.action_dim)
        diversity_score = diversity / max_entropy * 100
        
        metrics = {
            'mean_q': all_q_values.mean().item(),
            'max_q': all_q_values.max().item(),
            'min_q': all_q_values.min().item(),
            'constraint_violation_rate': constraint_violations / max(total_samples, 1),
            'action_distribution': action_dist.tolist(),
            'action_diversity_score': diversity_score
        }
        
        self.q_network1.train()
        self.q_network2.train()
        self.vae.train()
        
        return metrics
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'q_network1': self.q_network1.state_dict(),
            'q_network2': self.q_network2.state_dict(),
            'target_q1': self.target_q1.state_dict(),
            'target_q2': self.target_q2.state_dict(),
            'vae': self.vae.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'vae_optimizer': self.vae_optimizer.state_dict(),
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
        self.vae.load_state_dict(checkpoint['vae'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.vae_optimizer.load_state_dict(checkpoint['vae_optimizer'])
        self.train_step = checkpoint['train_step']
        self.training_history = checkpoint['training_history']
        if checkpoint['state_mean'] is not None:
            self.state_mean = checkpoint['state_mean'].to(self.device)
            self.state_std = checkpoint['state_std'].to(self.device)

