"""
Neural Network architectures for Safe RL agents.

Provides Q-networks for DQN/CQL and Actor-Critic networks for PPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class QNetwork(nn.Module):
    """
    Q-Network for DQN and CQL.
    
    Maps states to Q-values for each action.
    Architecture: MLP with configurable hidden layers.
    """
    
    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 5,
        hidden_dims: List[int] = [256, 256, 128],
        activation: str = 'relu',
        dropout: float = 0.1,
        use_batch_norm: bool = False
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (before activation)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif activation == 'swish':
                # Swish activation: x * sigmoid(x)
                layers.append(nn.SiLU())  # SiLU is Swish
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            
        Returns:
            Q-values for each action [batch_size, action_dim]
        """
        return self.network(state)
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Single state [state_dim] or [1, state_dim]
            epsilon: Exploration probability
            
        Returns:
            Selected action index
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()
        
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network architecture.
    
    Separates state value and advantage estimation for better learning.
    """
    
    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 5,
        hidden_dims: List[int] = [256, 256],
        activation: str = 'relu',
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.1))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.feature_layer = nn.Sequential(*layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dueling architecture.
        
        Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        """
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()
        
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()


class PolicyNetwork(nn.Module):
    """
    Policy Network for PPO.
    
    Maps states to action probabilities.
    """
    
    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 5,
        hidden_dims: List[int] = [256, 256, 128],
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            
        Returns:
            Action logits [batch_size, action_dim]
        """
        return self.network(state)
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities using softmax."""
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """
        Sample action from policy.
        
        Returns:
            action: Selected action index
            log_prob: Log probability of the action
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = probs.argmax(dim=1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        log_prob = F.log_softmax(logits, dim=-1)
        action_log_prob = log_prob.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        return action.item(), action_log_prob
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given state-action pairs.
        
        Returns:
            log_probs: Log probabilities of actions
            entropy: Policy entropy
        """
        logits = self.forward(states)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        return action_log_probs, entropy


class ValueNetwork(nn.Module):
    """
    Value Network for PPO.
    
    Maps states to state values.
    """
    
    def __init__(
        self,
        state_dim: int = 15,
        hidden_dims: List[int] = [256, 256, 128],
        activation: str = 'relu'
    ):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            
        Returns:
            State values [batch_size, 1]
        """
        return self.network(state)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    
    Shares feature extraction between policy and value heads.
    """
    
    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 5,
        hidden_dims: List[int] = [256, 256],
        activation: str = 'relu',
        use_batch_norm: bool = False
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif activation == 'swish':
                layers.append(nn.SiLU())
            else:
                layers.append(nn.Tanh())
            
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Policy head
        policy_layers = [nn.Linear(prev_dim, 128)]
        if use_batch_norm:
            policy_layers.append(nn.BatchNorm1d(128))
        policy_layers.append(nn.ReLU())
        policy_layers.append(nn.Linear(128, action_dim))
        self.policy_head = nn.Sequential(*policy_layers)
        
        # Value head
        value_layers = [nn.Linear(prev_dim, 128)]
        if use_batch_norm:
            value_layers.append(nn.BatchNorm1d(128))
        value_layers.append(nn.ReLU())
        value_layers.append(nn.Linear(128, 1))
        self.value_head = nn.Sequential(*value_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            action_logits: [batch_size, action_dim]
            values: [batch_size, 1]
        """
        features = self.shared(state)
        action_logits = self.policy_head(features)
        values = self.value_head(features)
        return action_logits, values
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and value estimate.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = probs.argmax(dim=1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        log_prob = F.log_softmax(logits, dim=-1)
        action_log_prob = log_prob.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        return action.item(), action_log_prob, value.squeeze(-1)
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Returns:
            log_probs: Log probabilities of actions
            values: State values
            entropy: Policy entropy
        """
        logits, values = self.forward(states)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        return action_log_probs, values.squeeze(-1), entropy

