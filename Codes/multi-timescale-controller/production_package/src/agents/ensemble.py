"""
Ensemble Agent for Safe RL

Combines multiple IQL agents for improved accuracy and robustness.
Uses voting or Q-value averaging for action selection.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .iql import IQLAgent
from .safety import SafetyConfig


class EnsembleIQLAgent:
    """
    Ensemble of IQL agents for improved performance.
    
    Combines multiple IQL models trained with different:
    - Random seeds
    - Hyperparameters
    - Initializations
    
    Uses voting or Q-value averaging for action selection.
    """
    
    def __init__(
        self,
        agents: List[IQLAgent],
        voting_method: str = 'q_average',  # 'q_average', 'voting', 'weighted'
        weights: Optional[List[float]] = None
    ):
        """
        Initialize ensemble agent.
        
        Args:
            agents: List of IQL agents
            voting_method: 'q_average', 'voting', or 'weighted'
            weights: Weights for each agent (None = uniform)
        """
        self.agents = agents
        self.voting_method = voting_method
        self.num_agents = len(agents)
        
        if weights is None:
            self.weights = [1.0 / self.num_agents] * self.num_agents
        else:
            assert len(weights) == self.num_agents
            self.weights = np.array(weights) / np.sum(weights)  # Normalize
        
        # Use first agent's safety config and normalization
        self.safety = agents[0].safety if agents else None
        self.state_mean = agents[0].state_mean if agents else None
        self.state_std = agents[0].state_std if agents else None
        self.device = agents[0].device if agents else None
    
    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        """Set normalization for all agents."""
        for agent in self.agents:
            agent.set_normalization(mean, std)
        self.state_mean = mean
        self.state_std = std
    
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state using first agent's normalization."""
        if self.state_mean is None or self.state_std is None:
            return state
        mean = torch.FloatTensor(self.state_mean).to(state.device)
        std = torch.FloatTensor(self.state_std).to(state.device)
        return (state - mean) / (std + 1e-8)
    
    def denormalize_state(self, state: torch.Tensor) -> np.ndarray:
        """Denormalize state."""
        if self.state_mean is None or self.state_std is None:
            return state.cpu().numpy()
        mean = torch.FloatTensor(self.state_mean).to(state.device)
        std = torch.FloatTensor(self.state_std).to(state.device)
        return (state * std + mean).cpu().numpy()
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        use_safety_shield: bool = True
    ) -> int:
        """
        Select action using ensemble method.
        
        Args:
            state: State vector
            deterministic: Whether to use deterministic policy
            use_safety_shield: Whether to apply safety shield
            
        Returns:
            Selected action
        """
        if self.voting_method == 'q_average':
            return self._select_action_q_average(state, deterministic, use_safety_shield)
        elif self.voting_method == 'voting':
            return self._select_action_voting(state, deterministic, use_safety_shield)
        elif self.voting_method == 'weighted':
            return self._select_action_weighted(state, deterministic, use_safety_shield)
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")
    
    def _select_action_q_average(
        self,
        state: np.ndarray,
        deterministic: bool,
        use_safety_shield: bool
    ) -> int:
        """Select action by averaging Q-values from all agents."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            state_normalized = self.normalize_state(state_tensor)
            
            # Get Q-values from all agents
            all_q_values = []
            for agent in self.agents:
                q1 = agent.q_network1(state_normalized)
                q2 = agent.q_network2(state_normalized)
                q = torch.min(q1, q2)  # Conservative Q-value
                all_q_values.append(q)
            
            # Average Q-values (weighted)
            avg_q = sum(w * q for w, q in zip(self.weights, all_q_values))
            
            # Apply safety mask
            if use_safety_shield:
                safe_mask = torch.tensor([
                    self.safety.is_action_safe(state, a, self.denormalize_state)
                    for a in range(9)
                ], dtype=torch.bool, device=self.device)
                
                avg_q[~safe_mask] = float('-inf')
                
                if not safe_mask.any():
                    return 8  # No-op
            
            if deterministic:
                return avg_q.argmax().item()
            else:
                # Softmax sampling
                probs = torch.softmax(avg_q, dim=-1)
                return torch.multinomial(probs, 1).item()
    
    def _select_action_voting(
        self,
        state: np.ndarray,
        deterministic: bool,
        use_safety_shield: bool
    ) -> int:
        """Select action by voting from all agents."""
        votes = []
        for agent in self.agents:
            action = agent.select_action(state, deterministic, use_safety_shield)
            votes.append(action)
        
        # Count votes (weighted)
        vote_counts = np.zeros(9)
        for action, weight in zip(votes, self.weights):
            vote_counts[action] += weight
        
        return int(np.argmax(vote_counts))
    
    def _select_action_weighted(
        self,
        state: np.ndarray,
        deterministic: bool,
        use_safety_shield: bool
    ) -> int:
        """Select action using weighted combination of Q-values and votes."""
        # Combine Q-average and voting
        q_action = self._select_action_q_average(state, deterministic, use_safety_shield)
        vote_action = self._select_action_voting(state, deterministic, use_safety_shield)
        
        # If they agree, use that action
        if q_action == vote_action:
            return q_action
        
        # Otherwise, use Q-average (more reliable)
        return q_action
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        use_safety_shield: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate ensemble agent.
        
        Args:
            dataloader: Data loader
            use_safety_shield: Whether to use safety shield
            
        Returns:
            Evaluation metrics
        """
        all_actions = []
        constraint_violations = 0
        total_samples = 0
        
        for batch in dataloader:
            states = batch['state']
            costs = batch['cost']
            
            for i in range(states.shape[0]):
                state = states[i].numpy()
                action = self.select_action(state, deterministic=True, use_safety_shield=use_safety_shield)
                all_actions.append(action)
                
                if costs[i].item() > 0.5:
                    constraint_violations += 1
                total_samples += 1
        
        # Calculate metrics
        action_dist = np.bincount(all_actions, minlength=9) / len(all_actions)
        entropy = -sum(p * np.log(p + 1e-10) for p in action_dist if p > 0)
        max_entropy = np.log(5)
        diversity = entropy / max_entropy
        
        return {
            'constraint_violation_rate': constraint_violations / total_samples if total_samples > 0 else 0.0,
            'action_distribution': action_dist.tolist(),
            'action_diversity': diversity,
            'num_samples': total_samples
        }
    
    def save(self, path: str):
        """Save all agents in ensemble."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            agent.save(str(path / f'agent_{i}.pt'))
        
        # Save ensemble metadata
        import json
        metadata = {
            'num_agents': self.num_agents,
            'voting_method': self.voting_method,
            'weights': self.weights.tolist()
        }
        with open(path / 'ensemble_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(
        cls,
        path: str,
        state_dim: int = 15,
        action_dim: int = 5,
        safety_config: Optional[SafetyConfig] = None
    ) -> 'EnsembleIQLAgent':
        """
        Load ensemble from directory.
        
        Args:
            path: Path to ensemble directory
            state_dim: State dimension
            action_dim: Action dimension
            safety_config: Safety configuration
            
        Returns:
            Loaded ensemble agent
        """
        path = Path(path)
        
        # Load metadata
        import json
        with open(path / 'ensemble_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load all agents
        agents = []
        for i in range(metadata['num_agents']):
            agent = IQLAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                safety_config=safety_config
            )
            agent.load(str(path / f'agent_{i}.pt'))
            agents.append(agent)
        
        return cls(
            agents=agents,
            voting_method=metadata['voting_method'],
            weights=metadata.get('weights')
        )

