"""
General Ensemble Agent for Safe RL

Combines multiple agents of different types for improved accuracy and robustness.
Uses voting or Q-value averaging for action selection.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from .safety import SafetyConfig


class GeneralEnsembleAgent:
    """
    General ensemble of RL agents for improved performance.
    
    Combines multiple models (IQL, CQL, BCQ, etc.) trained with different:
    - Random seeds
    - Hyperparameters
    - Algorithms
    
    Uses voting or Q-value averaging for action selection.
    """
    
    def __init__(
        self,
        agents: List,
        voting_method: str = 'weighted_voting',  # 'q_average', 'voting', 'weighted_voting'
        weights: Optional[List[float]] = None,
        agent_names: Optional[List[str]] = None
    ):
        """
        Initialize ensemble agent.
        
        Args:
            agents: List of agent objects (can be different types)
            voting_method: 'q_average', 'voting', or 'weighted_voting'
            weights: Weights for each agent (None = uniform)
            agent_names: Names of agents for logging
        """
        self.agents = agents
        self.voting_method = voting_method
        self.num_agents = len(agents)
        self.agent_names = agent_names or [f"Agent_{i}" for i in range(self.num_agents)]
        
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
            if hasattr(agent, 'set_normalization'):
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
    
    def denormalize_state(self, state) -> np.ndarray:
        """Denormalize state."""
        # Handle both numpy arrays and tensors
        if isinstance(state, np.ndarray):
            if self.state_mean is None or self.state_std is None:
                return state
            return state * self.state_std + self.state_mean
        
        # Handle torch tensors
        if self.state_mean is None or self.state_std is None:
            return state.cpu().numpy() if isinstance(state, torch.Tensor) else state
        mean = torch.FloatTensor(self.state_mean).to(state.device)
        std = torch.FloatTensor(self.state_std).to(state.device)
        return (state * std + mean).cpu().numpy()
    
    def _get_q_values(self, agent, state: np.ndarray) -> Optional[torch.Tensor]:
        """Get Q-values from an agent (supports multiple agent types)."""
        try:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # Normalize state
                if hasattr(agent, 'state_mean') and agent.state_mean is not None:
                    mean = torch.FloatTensor(agent.state_mean).to(state_tensor.device)
                    std = torch.FloatTensor(agent.state_std).to(state_tensor.device)
                    state_tensor = (state_tensor - mean) / (std + 1e-8)
                
                state_tensor = state_tensor.to(agent.device)
                
                # Try different Q-network attributes
                if hasattr(agent, 'q_network1') and hasattr(agent, 'q_network2'):
                    # IQL, CQL style
                    q1 = agent.q_network1(state_tensor)
                    q2 = agent.q_network2(state_tensor)
                    q = torch.min(q1, q2)  # Conservative Q-value
                    return q
                elif hasattr(agent, 'q_network'):
                    # DQN style
                    q = agent.q_network(state_tensor)
                    return q
                elif hasattr(agent, 'critic'):
                    # PPO, RCPO style - use critic as Q-value proxy
                    q = agent.critic(state_tensor)
                    return q
                else:
                    return None
        except Exception:
            return None
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = True,
        use_safety_shield: bool = True
    ) -> Tuple[int, Dict]:
        """
        Select action using ensemble method.
        
        Args:
            state: State vector
            deterministic: Whether to use deterministic policy
            use_safety_shield: Whether to apply safety shield
            
        Returns:
            Selected action, metadata dict
        """
        if self.voting_method == 'q_average':
            return self._select_action_q_average(state, deterministic, use_safety_shield)
        elif self.voting_method == 'voting':
            return self._select_action_voting(state, deterministic, use_safety_shield)
        elif self.voting_method == 'weighted_voting':
            return self._select_action_weighted_voting(state, deterministic, use_safety_shield)
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")
    
    def _select_action_q_average(
        self,
        state: np.ndarray,
        deterministic: bool,
        use_safety_shield: bool
    ) -> Tuple[int, Dict]:
        """Select action by averaging Q-values from all agents."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            state_normalized = self.normalize_state(state_tensor)
            
            # Get Q-values from all agents
            all_q_values = []
            q_available = []
            for i, agent in enumerate(self.agents):
                q = self._get_q_values(agent, state)
                if q is not None:
                    all_q_values.append(q)
                    q_available.append(i)
            
            if not all_q_values:
                # Fallback to voting
                return self._select_action_voting(state, deterministic, use_safety_shield)
            
            # Average Q-values (weighted)
            weights_available = [self.weights[i] for i in q_available]
            weights_available = np.array(weights_available) / np.sum(weights_available)
            
            avg_q = sum(w * q for w, q in zip(weights_available, all_q_values))
            
            # Apply safety mask
            if use_safety_shield and self.safety:
                # Convert state to numpy for safety check
                state_np = state if isinstance(state, np.ndarray) else state.cpu().numpy()
                safe_mask = torch.tensor([
                    self.safety.is_action_safe(state_np, a, None)
                    for a in range(9)
                ], dtype=torch.bool, device=self.device).unsqueeze(0)
                
                avg_q[~safe_mask] = float('-inf')
                
                if not safe_mask.any():
                    return 8, {'method': 'q_average', 'fallback': 'no_safe_actions'}  # No-op
            
            if deterministic:
                action = avg_q.argmax().item()
            else:
                # Softmax sampling
                probs = torch.softmax(avg_q, dim=-1)
                action = torch.multinomial(probs, 1).item()
            
            return action, {
                'method': 'q_average',
                'q_values': avg_q.cpu().numpy().flatten().tolist(),
                'agents_used': len(q_available)
            }
    
    def _select_action_voting(
        self,
        state: np.ndarray,
        deterministic: bool,
        use_safety_shield: bool
    ) -> Tuple[int, Dict]:
        """Select action by voting from all agents."""
        votes = []
        vote_details = []
        
        for i, agent in enumerate(self.agents):
            try:
                action = agent.select_action(state, deterministic, use_safety_shield)
                if isinstance(action, tuple):
                    action = action[0]
                votes.append(action)
                vote_details.append({
                    'agent': self.agent_names[i],
                    'action': action,
                    'weight': self.weights[i]
                })
            except Exception as e:
                # Skip failed agents
                continue
        
        if not votes:
            return 8, {'method': 'voting', 'fallback': 'no_agents_available'}  # No-op
        
        # Count votes (weighted)
        vote_counts = np.zeros(9)
        for action, weight in zip(votes, self.weights[:len(votes)]):
            vote_counts[action] += weight
        
        action = int(np.argmax(vote_counts))
        
        return action, {
            'method': 'voting',
            'vote_counts': vote_counts.tolist(),
            'votes': vote_details
        }
    
    def _select_action_weighted_voting(
        self,
        state: np.ndarray,
        deterministic: bool,
        use_safety_shield: bool
    ) -> Tuple[int, Dict]:
        """Select action using weighted combination of Q-values and votes."""
        # Try Q-average first
        try:
            q_action, q_meta = self._select_action_q_average(state, deterministic, use_safety_shield)
            vote_action, vote_meta = self._select_action_voting(state, deterministic, use_safety_shield)
            
            # If they agree, use that action
            if q_action == vote_action:
                return q_action, {
                    'method': 'weighted_voting',
                    'q_action': q_action,
                    'vote_action': vote_action,
                    'agreement': True
                }
            
            # Otherwise, prefer Q-average (more reliable)
            return q_action, {
                'method': 'weighted_voting',
                'q_action': q_action,
                'vote_action': vote_action,
                'agreement': False,
                'selected': 'q_average'
            }
        except Exception:
            # Fallback to voting
            return self._select_action_voting(state, deterministic, use_safety_shield)
    
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
            costs = batch.get('cost', None)
            
            for i in range(states.shape[0]):
                state = states[i].numpy()
                action, _ = self.select_action(state, deterministic=True, use_safety_shield=use_safety_shield)
                all_actions.append(action)
                
                if costs is not None and costs[i].item() > 0.5:
                    constraint_violations += 1
                total_samples += 1
        
        # Calculate metrics
        action_dist = np.bincount(all_actions, minlength=9) / len(all_actions) if all_actions else np.zeros(9)
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
            if hasattr(agent, 'save'):
                agent.save(str(path / f'agent_{i}.pt'))
            else:
                # Save state dict if available
                if hasattr(agent, 'state_dict'):
                    torch.save(agent.state_dict(), str(path / f'agent_{i}_state_dict.pt'))
        
        # Save ensemble metadata
        import json
        metadata = {
            'num_agents': self.num_agents,
            'voting_method': self.voting_method,
            'weights': self.weights.tolist(),
            'agent_names': self.agent_names
        }
        with open(path / 'ensemble_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(
        cls,
        path: str,
        agents: List,
        state_dim: int = 15,
        action_dim: int = 5,
        safety_config: Optional[SafetyConfig] = None
    ) -> 'GeneralEnsembleAgent':
        """
        Load ensemble from directory.
        
        Args:
            path: Path to ensemble directory
            agents: List of agent instances (already loaded)
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
        
        return cls(
            agents=agents,
            voting_method=metadata['voting_method'],
            weights=metadata.get('weights'),
            agent_names=metadata.get('agent_names')
        )

