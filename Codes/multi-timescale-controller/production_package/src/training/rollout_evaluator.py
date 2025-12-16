"""
Rollout-based evaluation for Safe RL agents.

Simulates episodes using the learned policy to evaluate
performance beyond just dataset metrics.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from ..agents.cql import CQLAgent
from ..agents.dqn import DQNAgent
from ..agents.ppo import PPOAgent
from ..dataset.generator import RRMDatasetGenerator


@dataclass
class RolloutMetrics:
    """Metrics from a rollout evaluation."""
    total_reward: float
    mean_reward: float
    total_cost: float
    constraint_violations: int
    episode_length: int
    action_counts: np.ndarray
    throughput_improvement: float
    retry_reduction: float


class RolloutEvaluator:
    """
    Evaluates agents using simulated rollouts.
    
    Uses the dataset generator as a simulator to run episodes
    and collect performance metrics.
    """
    
    # Feature indices
    IDX_EDGE_THROUGHPUT = 6
    IDX_AVG_THROUGHPUT = 5
    IDX_P95_RETRY = 2
    IDX_P95_PER = 3
    IDX_TX_POWER = 9
    IDX_OBSS_PD = 8
    
    def __init__(
        self,
        generator: Optional[RRMDatasetGenerator] = None,
        seed: int = 42,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None
    ):
        """
        Initialize rollout evaluator.
        
        Args:
            generator: Dataset generator to use as simulator
            seed: Random seed
            state_mean: State mean for normalization
            state_std: State std for normalization
        """
        self.generator = generator or RRMDatasetGenerator(seed=seed)
        self.state_mean = state_mean
        self.state_std = state_std
    
    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        """Set normalization parameters."""
        self.state_mean = mean
        self.state_std = std
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state for agent input."""
        if self.state_mean is not None:
            return (state - self.state_mean) / (self.state_std + 1e-8)
        return state
    
    def run_episode(
        self,
        agent: Union[CQLAgent, DQNAgent, PPOAgent],
        max_steps: int = 100,
        use_safety_shield: bool = True
    ) -> RolloutMetrics:
        """
        Run a single episode with the agent.
        
        Args:
            agent: Agent to evaluate
            max_steps: Maximum steps per episode
            use_safety_shield: Whether to use safety shield
            
        Returns:
            RolloutMetrics with episode statistics
        """
        # Initialize episode
        state = self.generator._generate_initial_state()
        initial_state = state.copy()
        
        total_reward = 0.0
        total_cost = 0.0
        constraint_violations = 0
        action_counts = np.zeros(5, dtype=np.int32)
        
        for step in range(max_steps):
            # Normalize state for agent
            normalized_state = self.normalize_state(state)
            
            # Get action from agent
            if isinstance(agent, (CQLAgent, DQNAgent)):
                action = agent.select_action(
                    normalized_state,
                    epsilon=0.0,
                    use_safety_shield=use_safety_shield
                )
            else:  # PPO
                action, _, _ = agent.select_action(
                    normalized_state,
                    deterministic=True,
                    use_safety_shield=use_safety_shield
                )
            
            action_counts[action] += 1
            
            # Apply action
            next_state = self.generator._apply_action(state, action)
            reward = self.generator._calculate_reward(state, action, next_state)
            cost = self.generator._calculate_cost(state, action, next_state)
            
            total_reward += reward
            total_cost += cost
            
            if cost >= 1.0:
                constraint_violations += 1
            
            # Check for episode termination
            done = np.random.random() < 0.01 or cost >= 1.0
            
            if done:
                break
            
            state = next_state
        
        # Calculate performance improvements
        throughput_improvement = (
            (state[self.IDX_EDGE_THROUGHPUT] - initial_state[self.IDX_EDGE_THROUGHPUT]) /
            (initial_state[self.IDX_EDGE_THROUGHPUT] + 1e-8)
        )
        
        retry_reduction = (
            (initial_state[self.IDX_P95_RETRY] - state[self.IDX_P95_RETRY]) /
            (initial_state[self.IDX_P95_RETRY] + 1e-8)
        )
        
        episode_length = step + 1
        
        return RolloutMetrics(
            total_reward=total_reward,
            mean_reward=total_reward / episode_length,
            total_cost=total_cost,
            constraint_violations=constraint_violations,
            episode_length=episode_length,
            action_counts=action_counts,
            throughput_improvement=throughput_improvement,
            retry_reduction=retry_reduction
        )
    
    def evaluate(
        self,
        agent: Union[CQLAgent, DQNAgent, PPOAgent],
        num_episodes: int = 100,
        max_steps: int = 100,
        use_safety_shield: bool = True,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate agent over multiple episodes.
        
        Args:
            agent: Agent to evaluate
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            use_safety_shield: Whether to use safety shield
            verbose: Whether to print progress
            
        Returns:
            Dictionary of aggregated metrics
        """
        all_metrics = []
        
        for ep in range(num_episodes):
            metrics = self.run_episode(agent, max_steps, use_safety_shield)
            all_metrics.append(metrics)
            
            if verbose and (ep + 1) % 20 == 0:
                print(f"  Episode {ep + 1}/{num_episodes}")
        
        # Aggregate metrics
        total_rewards = [m.total_reward for m in all_metrics]
        mean_rewards = [m.mean_reward for m in all_metrics]
        total_costs = [m.total_cost for m in all_metrics]
        violations = [m.constraint_violations for m in all_metrics]
        lengths = [m.episode_length for m in all_metrics]
        throughput_imps = [m.throughput_improvement for m in all_metrics]
        retry_reds = [m.retry_reduction for m in all_metrics]
        
        # Aggregate action counts
        total_action_counts = np.sum([m.action_counts for m in all_metrics], axis=0)
        action_distribution = total_action_counts / total_action_counts.sum()
        
        return {
            'mean_episode_reward': np.mean(total_rewards),
            'std_episode_reward': np.std(total_rewards),
            'mean_step_reward': np.mean(mean_rewards),
            'mean_episode_cost': np.mean(total_costs),
            'total_violations': sum(violations),
            'violation_rate': sum(violations) / (sum(lengths)),
            'mean_episode_length': np.mean(lengths),
            'mean_throughput_improvement': np.mean(throughput_imps),
            'mean_retry_reduction': np.mean(retry_reds),
            'action_distribution': action_distribution.tolist(),
            'num_episodes': num_episodes
        }
    
    def compare_agents(
        self,
        agents: Dict[str, Union[CQLAgent, DQNAgent, PPOAgent]],
        num_episodes: int = 100,
        max_steps: int = 100,
        verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple agents using rollout evaluation.
        
        Args:
            agents: Dictionary of agent name to agent
            num_episodes: Number of episodes per agent
            max_steps: Maximum steps per episode
            verbose: Whether to print progress
            
        Returns:
            Dictionary of agent name to metrics
        """
        results = {}
        
        for name, agent in agents.items():
            if verbose:
                print(f"\nEvaluating {name}...")
            
            results[name] = self.evaluate(
                agent, num_episodes, max_steps,
                use_safety_shield=True, verbose=verbose
            )
            
            if verbose:
                print(f"  Mean Episode Reward: {results[name]['mean_episode_reward']:.4f}")
                print(f"  Violation Rate: {results[name]['violation_rate']:.2%}")
                print(f"  Throughput Improvement: {results[name]['mean_throughput_improvement']:.2%}")
        
        return results
    
    def generate_comparison_report(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> str:
        """Generate text report comparing agents."""
        lines = []
        lines.append("=" * 60)
        lines.append("ROLLOUT EVALUATION COMPARISON")
        lines.append("=" * 60)
        lines.append("")
        
        # Performance ranking
        lines.append("PERFORMANCE RANKING (by Mean Episode Reward):")
        lines.append("-" * 40)
        
        sorted_agents = sorted(
            results.items(),
            key=lambda x: x[1]['mean_episode_reward'],
            reverse=True
        )
        
        for i, (name, metrics) in enumerate(sorted_agents, 1):
            reward = metrics['mean_episode_reward']
            std = metrics['std_episode_reward']
            lines.append(f"  {i}. {name}: {reward:.4f} (±{std:.4f})")
        
        lines.append("")
        
        # Safety ranking
        lines.append("SAFETY RANKING (by Violation Rate):")
        lines.append("-" * 40)
        
        sorted_by_safety = sorted(
            results.items(),
            key=lambda x: x[1]['violation_rate']
        )
        
        for i, (name, metrics) in enumerate(sorted_by_safety, 1):
            rate = metrics['violation_rate']
            lines.append(f"  {i}. {name}: {rate:.2%}")
        
        lines.append("")
        
        # Detailed metrics
        lines.append("DETAILED METRICS:")
        lines.append("-" * 40)
        
        for name, metrics in results.items():
            lines.append(f"\n{name}:")
            lines.append(f"  Episodes: {metrics['num_episodes']}")
            lines.append(f"  Mean Episode Reward: {metrics['mean_episode_reward']:.4f}")
            lines.append(f"  Mean Step Reward: {metrics['mean_step_reward']:.6f}")
            lines.append(f"  Violation Rate: {metrics['violation_rate']:.2%}")
            lines.append(f"  Mean Episode Length: {metrics['mean_episode_length']:.1f}")
            lines.append(f"  Throughput Improvement: {metrics['mean_throughput_improvement']:.2%}")
            lines.append(f"  Retry Reduction: {metrics['mean_retry_reduction']:.2%}")
            
            action_names = ['Inc Tx', 'Dec Tx', 'Inc OBSS', 'Dec OBSS', 'No-op']
            lines.append(f"  Action Distribution:")
            for action_name, prob in zip(action_names, metrics['action_distribution']):
                lines.append(f"    {action_name}: {prob:.1%}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)

