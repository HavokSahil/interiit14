"""
Evaluation and Comparison Framework for Safe RL Agents.

Provides comprehensive evaluation metrics and comparison tools.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

from ..agents.cql import CQLAgent
from ..agents.dqn import DQNAgent
from ..agents.ppo import PPOAgent
from ..agents.rcpo import RCPOAgent
from ..agents.iql import IQLAgent
from ..agents.bcq import BCQAgent
from ..agents.safety import SafetyModule


class Evaluator:
    """
    Comprehensive evaluator for Safe RL agents.
    
    Evaluates:
    - Performance metrics (rewards, Q-values)
    - Safety metrics (constraint violations)
    - Action distributions
    - Comparison across algorithms
    """
    
    ACTION_NAMES = [
        'Inc Tx Power (+2dB)',
        'Dec Tx Power (-2dB)',
        'Inc OBSS-PD (+4dB)',
        'Dec OBSS-PD (-4dB)',
        'Inc Channel Width',
        'Dec Channel Width',
        'Inc Channel Number',
        'Dec Channel Number',
        'No-op'
    ]
    
    def __init__(
        self,
        agents: Dict[str, Union[CQLAgent, DQNAgent, PPOAgent, RCPOAgent, IQLAgent, BCQAgent]],
        test_loader: torch.utils.data.DataLoader,
        save_dir: str = 'results'
    ):
        """
        Initialize evaluator.
        
        Args:
            agents: Dictionary of agent name to agent instance
            test_loader: Test data loader
            save_dir: Directory for saving results
        """
        self.agents = agents
        self.test_loader = test_loader
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
    
    def evaluate_all(self) -> Dict[str, Dict]:
        """
        Evaluate all agents.
        
        Returns:
            Dictionary of results for each agent
        """
        for name, agent in self.agents.items():
            print(f"\nEvaluating {name}...")
            self.results[name] = self._evaluate_agent(agent, name)
        
        return self.results
    
    def _evaluate_agent(
        self,
        agent: Union[CQLAgent, DQNAgent, PPOAgent, RCPOAgent, IQLAgent, BCQAgent],
        name: str
    ) -> Dict:
        """Evaluate a single agent."""
        results = {
            'performance': {},
            'safety': {},
            'actions': {}
        }
        
        # Get basic evaluation metrics
        eval_metrics = agent.evaluate(self.test_loader, use_safety_shield=True)
        
        # Performance metrics
        if 'mean_q' in eval_metrics:
            results['performance']['mean_q'] = eval_metrics['mean_q']
            results['performance']['max_q'] = eval_metrics['max_q']
            results['performance']['min_q'] = eval_metrics['min_q']
        
        if 'mean_entropy' in eval_metrics:
            results['performance']['entropy'] = eval_metrics['mean_entropy']
        
        # Safety metrics
        results['safety']['constraint_violation_rate'] = eval_metrics.get(
            'constraint_violation_rate', 0
        )
        
        # Action distribution
        # Get action dimension dynamically
        action_dim = 9  # Default, but will be updated from agent if available
        if hasattr(agent, 'action_dim'):
            action_dim = agent.action_dim
        results['actions']['distribution'] = eval_metrics.get(
            'action_distribution', [1.0/action_dim] * action_dim
        )
        
        # Detailed action analysis
        results['actions'] = self._analyze_actions(agent)
        
        # Compute simulated rewards
        results['performance']['simulated_reward'] = self._compute_simulated_reward(agent)
        
        return results
    
    def _analyze_actions(
        self,
        agent: Union[CQLAgent, DQNAgent, PPOAgent, RCPOAgent, IQLAgent, BCQAgent]
    ) -> Dict:
        """Analyze action selection patterns."""
        # Get action dimension from agent
        if hasattr(agent, 'action_dim'):
            action_dim = agent.action_dim
        elif hasattr(agent, 'q_network') and hasattr(agent.q_network, 'action_dim'):
            action_dim = agent.q_network.action_dim
        elif hasattr(agent, 'q_network1') and hasattr(agent.q_network1, 'action_dim'):
            action_dim = agent.q_network1.action_dim
        elif hasattr(agent, 'actor_critic') and hasattr(agent.actor_critic, 'action_dim'):
            action_dim = agent.actor_critic.action_dim
        else:
            # Default to 9 actions (current implementation)
            action_dim = 9
        
        action_counts = np.zeros(action_dim)
        safe_action_counts = np.zeros(action_dim)
        unsafe_attempts = 0
        total_samples = 0
        
        device = agent.device
        
        with torch.no_grad():
            for batch in self.test_loader:
                states = batch['state'].to(device)
                
                for i in range(states.shape[0]):
                    state = states[i].cpu().numpy()
                    
                    # Get action without safety shield
                    # Handle different agent types
                    if isinstance(agent, (IQLAgent, BCQAgent)):
                        # IQL and BCQ use dual Q-networks (min of two)
                        q1 = agent.q_network1(states[i:i+1])
                        q2 = agent.q_network2(states[i:i+1])
                        q_values = torch.min(q1, q2)
                        raw_action = q_values.argmax(dim=1).item()
                    elif isinstance(agent, (CQLAgent, DQNAgent)):
                        # CQL and DQN use single Q-network
                        q_values = agent.q_network(states[i:i+1])
                        raw_action = q_values.argmax(dim=1).item()
                    elif hasattr(agent, 'actor_critic'):  # PPO or RCPO
                        logits, _ = agent.actor_critic(states[i:i+1])
                        raw_action = logits.argmax(dim=1).item()
                    else:
                        # Fallback: use select_action method
                        raw_action = agent.select_action(state, epsilon=0.0, use_safety_shield=False)
                        if isinstance(raw_action, tuple):
                            raw_action = raw_action[0]
                    
                    action_counts[raw_action] += 1
                    
                    # Check if action is safe
                    if agent.safety.is_action_safe(state, raw_action, agent.denormalize_state):
                        safe_action_counts[raw_action] += 1
                    else:
                        unsafe_attempts += 1
                    
                    total_samples += 1
        
        return {
            'distribution': (action_counts / total_samples).tolist(),
            'safe_distribution': (safe_action_counts / total_samples).tolist(),
            'unsafe_attempt_rate': unsafe_attempts / total_samples,
            'action_names': self.ACTION_NAMES
        }
    
    def _compute_simulated_reward(
        self,
        agent: Union[CQLAgent, DQNAgent, PPOAgent, RCPOAgent]
    ) -> float:
        """Compute average reward on test set."""
        total_reward = 0
        total_samples = 0
        
        for batch in self.test_loader:
            rewards = batch['reward'].squeeze(-1)
            total_reward += rewards.sum().item()
            total_samples += len(rewards)
        
        return total_reward / total_samples
    
    def compare_agents(self) -> Dict:
        """
        Compare all agents and generate comparison metrics.
        
        Returns:
            Comparison summary
        """
        if not self.results:
            self.evaluate_all()
        
        comparison = {
            'performance_ranking': [],
            'safety_ranking': [],
            'summary': {}
        }
        
        # Rank by performance (simulated reward)
        perf_scores = {
            name: res['performance'].get('simulated_reward', 0)
            for name, res in self.results.items()
        }
        comparison['performance_ranking'] = sorted(
            perf_scores.items(), key=lambda x: x[1], reverse=True
        )
        
        # Rank by safety (lower violation rate is better)
        safety_scores = {
            name: res['safety'].get('constraint_violation_rate', 1)
            for name, res in self.results.items()
        }
        comparison['safety_ranking'] = sorted(
            safety_scores.items(), key=lambda x: x[1]
        )
        
        # Summary table
        for name, res in self.results.items():
            comparison['summary'][name] = {
                'reward': res['performance'].get('simulated_reward', 0),
                'violation_rate': res['safety'].get('constraint_violation_rate', 0),
                'mean_q': res['performance'].get('mean_q', 'N/A'),
                'unsafe_attempts': res['actions'].get('unsafe_attempt_rate', 0)
            }
        
        return comparison
    
    def plot_comparison(self, save: bool = True):
        """Generate comparison plots."""
        if not self.results:
            self.evaluate_all()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        agent_names = list(self.results.keys())
        colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green for CQL, Blue for DQN, Red for PPO
        
        # Plot 1: Action Distribution
        ax1 = axes[0, 0]
        x = np.arange(5)
        width = 0.25
        
        for i, name in enumerate(agent_names):
            dist = self.results[name]['actions']['distribution']
            ax1.bar(x + i * width, dist, width, label=name, color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('Action')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Action Distribution by Agent')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(['Inc Tx', 'Dec Tx', 'Inc OBSS', 'Dec OBSS', 'No-op'], rotation=45)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Safety Metrics
        ax2 = axes[0, 1]
        violation_rates = [
            self.results[name]['safety']['constraint_violation_rate']
            for name in agent_names
        ]
        unsafe_attempts = [
            self.results[name]['actions']['unsafe_attempt_rate']
            for name in agent_names
        ]
        
        x = np.arange(len(agent_names))
        ax2.bar(x - 0.2, violation_rates, 0.4, label='Violation Rate', color='#e74c3c', alpha=0.8)
        ax2.bar(x + 0.2, unsafe_attempts, 0.4, label='Unsafe Attempts', color='#f39c12', alpha=0.8)
        ax2.set_xlabel('Agent')
        ax2.set_ylabel('Rate')
        ax2.set_title('Safety Metrics')
        ax2.set_xticks(x)
        ax2.set_xticklabels(agent_names)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Performance Metrics (Q-values if available)
        ax3 = axes[1, 0]
        q_means = []
        q_maxs = []
        q_mins = []
        
        for name in agent_names:
            perf = self.results[name]['performance']
            q_means.append(perf.get('mean_q', 0))
            q_maxs.append(perf.get('max_q', 0))
            q_mins.append(perf.get('min_q', 0))
        
        x = np.arange(len(agent_names))
        ax3.bar(x - 0.2, q_means, 0.2, label='Mean Q', color='#3498db', alpha=0.8)
        ax3.bar(x, q_maxs, 0.2, label='Max Q', color='#2ecc71', alpha=0.8)
        ax3.bar(x + 0.2, q_mins, 0.2, label='Min Q', color='#e74c3c', alpha=0.8)
        ax3.set_xlabel('Agent')
        ax3.set_ylabel('Q-Value')
        ax3.set_title('Q-Value Statistics')
        ax3.set_xticks(x)
        ax3.set_xticklabels(agent_names)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Summary Comparison
        ax4 = axes[1, 1]
        
        # Normalize metrics for radar-like comparison
        metrics = ['Reward', 'Safety', 'Conservatism']
        
        rewards = [self.results[name]['performance'].get('simulated_reward', 0) for name in agent_names]
        safety = [1 - self.results[name]['safety']['constraint_violation_rate'] for name in agent_names]
        conservatism = [1 - self.results[name]['actions']['unsafe_attempt_rate'] for name in agent_names]
        
        # Normalize to 0-1
        def normalize(vals):
            min_v, max_v = min(vals), max(vals)
            if max_v == min_v:
                return [0.5] * len(vals)
            return [(v - min_v) / (max_v - min_v) for v in vals]
        
        rewards_norm = normalize(rewards)
        
        x = np.arange(len(metrics))
        for i, name in enumerate(agent_names):
            values = [rewards_norm[i], safety[i], conservatism[i]]
            ax4.bar(x + i * 0.25, values, 0.25, label=name, color=colors[i], alpha=0.8)
        
        ax4.set_xlabel('Metric')
        ax4.set_ylabel('Normalized Score')
        ax4.set_title('Overall Comparison (Higher is Better)')
        ax4.set_xticks(x + 0.25)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.set_ylim(0, 1.1)
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / 'comparison.png', dpi=150, bbox_inches='tight')
            print(f"Saved comparison plot to {self.save_dir / 'comparison.png'}")
        
        plt.show()
    
    def generate_report(self) -> str:
        """Generate a text report of the evaluation."""
        if not self.results:
            self.evaluate_all()
        
        comparison = self.compare_agents()
        
        report = []
        report.append("=" * 60)
        report.append("SAFE RL FOR RRM - EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Performance Ranking
        report.append("PERFORMANCE RANKING (by Simulated Reward):")
        report.append("-" * 40)
        for i, (name, score) in enumerate(comparison['performance_ranking'], 1):
            report.append(f"  {i}. {name}: {score:.4f}")
        report.append("")
        
        # Safety Ranking
        report.append("SAFETY RANKING (by Constraint Violation Rate):")
        report.append("-" * 40)
        for i, (name, rate) in enumerate(comparison['safety_ranking'], 1):
            report.append(f"  {i}. {name}: {rate:.2%}")
        report.append("")
        
        # Detailed Results
        report.append("DETAILED RESULTS:")
        report.append("-" * 40)
        
        for name, res in self.results.items():
            report.append(f"\n{name}:")
            report.append(f"  Performance:")
            for key, value in res['performance'].items():
                if isinstance(value, float):
                    report.append(f"    {key}: {value:.4f}")
            
            report.append(f"  Safety:")
            report.append(f"    Constraint Violation Rate: {res['safety']['constraint_violation_rate']:.2%}")
            report.append(f"    Unsafe Action Attempts: {res['actions']['unsafe_attempt_rate']:.2%}")
            
            report.append(f"  Action Distribution:")
            action_dist = res['actions']['distribution']
            # Handle case where distribution might have different length than ACTION_NAMES
            num_actions = min(len(self.ACTION_NAMES), len(action_dist))
            for i in range(num_actions):
                action_name = self.ACTION_NAMES[i] if i < len(self.ACTION_NAMES) else f"Action {i}"
                prob = action_dist[i] if i < len(action_dist) else 0.0
                report.append(f"    {action_name}: {prob:.2%}")
        
        report.append("")
        report.append("=" * 60)
        report.append("END OF REPORT")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        # Save report
        with open(self.save_dir / 'report.txt', 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def save_results(self):
        """Save evaluation results to JSON."""
        if not self.results:
            self.evaluate_all()
        
        # Convert numpy types
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        results = convert_types(self.results)
        
        with open(self.save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved results to {self.save_dir / 'evaluation_results.json'}")

