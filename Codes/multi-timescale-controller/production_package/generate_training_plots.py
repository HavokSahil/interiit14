#!/usr/bin/env python3
"""
Generate Training Plots from Real Pipeline Data

Generates 4 key plots:
1. Reward vs Episodes - Shows learning progress
2. Cost Violations vs Episodes - Shows safety compliance
3. Action Distribution Over Time - Shows stable policy
4. State-wise Performance Heatmap - Shows RL adapts to conditions

Usage: python generate_training_plots.py [--epochs 50] [--dataset data/rrm_dataset_expanded.h5]
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
import yaml
from collections import defaultdict
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.dataset.dataset import create_dataloaders
from src.agents.cql import CQLAgent
from src.agents.iql import IQLAgent
from src.agents.bcq import BCQAgent
from src.agents.general_ensemble import GeneralEnsembleAgent
from src.agents.safety import SafetyConfig
from src.training.trainer import Trainer


def collect_evaluation_data(agent, dataloader):
    """Collect reward, cost, and action data from evaluation."""
    all_rewards = []
    all_costs = []
    all_actions = []
    all_states = []
    violations = 0
    total_samples = 0
    
    # Handle ensemble vs individual agents
    is_ensemble = isinstance(agent, GeneralEnsembleAgent)
    
    if not is_ensemble:
        # Use agent's evaluate method to get proper eval mode handling
        eval_metrics = agent.evaluate(dataloader, use_safety_shield=True)
        
        # Set networks to eval mode properly
        if hasattr(agent, 'q_network'):
            agent.q_network.eval()
        if hasattr(agent, 'q_network1'):
            agent.q_network1.eval()
            agent.q_network2.eval()
        if hasattr(agent, 'value_network'):
            agent.value_network.eval()
        if hasattr(agent, 'policy_network'):
            agent.policy_network.eval()
        if hasattr(agent, 'actor_critic'):
            agent.actor_critic.eval()
        if hasattr(agent, 'vae'):
            agent.vae.eval()
    else:
        # For ensemble, set all sub-agents to eval mode
        for sub_agent in agent.agents:
            if hasattr(sub_agent, 'q_network'):
                sub_agent.q_network.eval()
            if hasattr(sub_agent, 'q_network1'):
                sub_agent.q_network1.eval()
                sub_agent.q_network2.eval()
            if hasattr(sub_agent, 'value_network'):
                sub_agent.value_network.eval()
            if hasattr(sub_agent, 'policy_network'):
                sub_agent.policy_network.eval()
            if hasattr(sub_agent, 'actor_critic'):
                sub_agent.actor_critic.eval()
            if hasattr(sub_agent, 'vae'):
                sub_agent.vae.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            states = batch['state']
            rewards = batch['reward'].squeeze(-1)
            costs = batch['cost'].squeeze(-1)
            
            # Get actions from agent
            batch_actions = []
            
            for i in range(states.shape[0]):
                state = states[i].numpy()
                try:
                    action = agent.select_action(state, deterministic=True, use_safety_shield=True)
                    if isinstance(action, tuple):
                        action = action[0]
                    batch_actions.append(int(action))
                except Exception as e:
                    batch_actions.append(8)  # No-op on error
            
            # Collect data
            all_rewards.extend(rewards.numpy())
            all_costs.extend(costs.numpy())
            all_actions.extend(batch_actions)
            all_states.append(states.numpy())
            
            # Count violations (cost > 0.5 indicates hard constraint violation)
            violations += (costs > 0.5).sum().item()
            total_samples += len(costs)
    
    # Set networks back to train mode
    if not is_ensemble:
        if hasattr(agent, 'q_network'):
            agent.q_network.train()
        if hasattr(agent, 'q_network1'):
            agent.q_network1.train()
            agent.q_network2.train()
        if hasattr(agent, 'value_network'):
            agent.value_network.train()
        if hasattr(agent, 'policy_network'):
            agent.policy_network.train()
        if hasattr(agent, 'actor_critic'):
            agent.actor_critic.train()
        if hasattr(agent, 'vae'):
            agent.vae.train()
    else:
        # For ensemble, set all sub-agents back to train mode
        for sub_agent in agent.agents:
            if hasattr(sub_agent, 'q_network'):
                sub_agent.q_network.train()
            if hasattr(sub_agent, 'q_network1'):
                sub_agent.q_network1.train()
                sub_agent.q_network2.train()
            if hasattr(sub_agent, 'value_network'):
                sub_agent.value_network.train()
            if hasattr(sub_agent, 'policy_network'):
                sub_agent.policy_network.train()
            if hasattr(sub_agent, 'actor_critic'):
                sub_agent.actor_critic.train()
            if hasattr(sub_agent, 'vae'):
                sub_agent.vae.train()
    
    return {
        'mean_reward': np.mean(all_rewards) if all_rewards else 0.0,
        'mean_cost': np.mean(all_costs) if all_costs else 0.0,
        'violation_rate': violations / total_samples if total_samples > 0 else 0.0,
        'action_distribution': np.bincount(all_actions, minlength=9) / len(all_actions) if len(all_actions) > 0 else np.zeros(9),
        'states_sample': np.vstack(all_states) if all_states else None,
        'all_rewards': all_rewards,
        'all_costs': all_costs,
        'all_actions': all_actions
    }


def train_with_data_collection(
    dataset_path: str,
    config_path: str,
    epochs: int = 50,
    eval_freq: int = 5
):
    """Train models and collect data for plotting."""
    print("="*60)
    print("TRAINING WITH DATA COLLECTION")
    print("="*60)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataloaders
    batch_size = config.get('training', {}).get('batch_size', 256)
    print(f"Loading dataset: {dataset_path}")
    train_loader, val_loader, test_loader = create_dataloaders(dataset_path, batch_size=batch_size)
    train_dataset = train_loader.dataset
    state_dim = train_dataset.num_features
    action_dim = train_dataset.num_actions
    
    print(f"Dataset: {len(train_dataset)} samples")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print()
    
    # Train ensemble models and collect data
    all_training_data = {}
    
    models_to_train = [
        ('CQL', CQLAgent, config.get('cql', {})),
        ('IQL', IQLAgent, config.get('iql', {})),
        ('BCQ', BCQAgent, config.get('bcq', {}))
    ]
    
    for model_name, AgentClass, model_config in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        # Create agent
        if model_name == 'CQL':
            agent = AgentClass(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=config.get('network', {}).get('hidden_layers', [256, 256, 128]),
                learning_rate=config.get('training', {}).get('learning_rate', 3e-4),
                gamma=config.get('training', {}).get('gamma', 0.99),
                alpha=model_config.get('alpha', 0.2),
                min_q_weight=model_config.get('min_q_weight', 0.1),
                target_update_freq=model_config.get('target_update_freq', 100),
                use_lagrange=model_config.get('use_lagrange', True),
                target_action_gap=model_config.get('target_action_gap', 0.5)
            )
        elif model_name == 'IQL':
            agent = AgentClass(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=config.get('network', {}).get('hidden_layers', [256, 256, 128]),
                learning_rate=model_config.get('learning_rate', 0.000138),
                gamma=config.get('training', {}).get('gamma', 0.99),
                expectile=model_config.get('expectile', 0.9),
                temperature=model_config.get('temperature', 3.56),
                tau=model_config.get('tau', 0.0043)
            )
        else:  # BCQ
            agent = AgentClass(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=config.get('network', {}).get('hidden_layers', [256, 256, 128]),
                learning_rate=model_config.get('learning_rate', 3e-4),
                gamma=config.get('training', {}).get('gamma', 0.99),
                threshold=model_config.get('threshold', 0.3)
            )
        
        # Create trainer
        trainer = Trainer(
            agent, train_loader, val_loader, test_loader,
            config=config, save_dir='checkpoints', experiment_name=model_name
        )
        
        # Collect data during training
        epoch_data = []
        
        # Modified training loop with data collection
        for epoch in range(epochs):
            # Train epoch
            train_metrics = trainer._train_epoch()
            
            # Collect data every eval_freq epochs
            if (epoch + 1) % eval_freq == 0:
                eval_data = collect_evaluation_data(agent, val_loader)
                
                epoch_info = {
                    'epoch': epoch + 1,
                    'mean_reward': eval_data['mean_reward'],
                    'mean_cost': eval_data['mean_cost'],
                    'violation_rate': eval_data['violation_rate'],
                    'action_distribution': eval_data['action_distribution'],
                    'states_sample': eval_data['states_sample']
                }
                
                epoch_data.append(epoch_info)
                print(f"Epoch {epoch + 1}: Reward={epoch_info['mean_reward']:.4f}, "
                      f"Cost={epoch_info['mean_cost']:.4f}, "
                      f"Violations={epoch_info['violation_rate']:.2%}")
        
        all_training_data[model_name] = epoch_data
        print(f"✓ {model_name} training complete!")
    
    # Create and evaluate ensemble (using final trained models)
    print(f"\n{'='*60}")
    print("Creating and Evaluating Ensemble")
    print(f"{'='*60}")
    
    # Load agents from checkpoints to create ensemble
    safety_config_dict = config.get('safety', {})
    safety_config = SafetyConfig(
        tx_power_hard_min=safety_config_dict['hard']['tx_power_min'],
        tx_power_hard_max=safety_config_dict['hard']['tx_power_max'],
        obss_pd_hard_min=safety_config_dict['hard']['obss_pd_min'],
        obss_pd_hard_max=safety_config_dict['hard']['obss_pd_max'],
        tx_power_shield_min=safety_config_dict['shield']['tx_power_min'],
        tx_power_shield_max=safety_config_dict['shield']['tx_power_max'],
        obss_pd_shield_min=safety_config_dict['shield']['obss_pd_min'],
        obss_pd_shield_max=safety_config_dict['shield']['obss_pd_max']
    )
    
    ensemble_agents = []
    ensemble_names = []
    
    # Weights from Model Config.txt (normalized)
    raw_weights = {'BCQ': 56.3, 'IQL': 63.3, 'CQL': 14.2}
    total_weight = sum(raw_weights.values())
    ensemble_weights = []
    
    print("Loading trained agents for ensemble...")
    for model_name, AgentClass, model_config in models_to_train:
        checkpoint_path = Path('checkpoints') / model_name / 'best.pt'
        if checkpoint_path.exists():
            print(f"  Loading {model_name}...")
            # Create agent
            if model_name == 'CQL':
                agent = AgentClass(
                    state_dim=state_dim, action_dim=action_dim,
                    hidden_dims=config.get('network', {}).get('hidden_layers', [256, 256, 128]),
                    learning_rate=config.get('training', {}).get('learning_rate', 3e-4),
                    gamma=config.get('training', {}).get('gamma', 0.99),
                    alpha=model_config.get('alpha', 0.2),
                    min_q_weight=model_config.get('min_q_weight', 0.1),
                    target_update_freq=model_config.get('target_update_freq', 100),
                    use_lagrange=model_config.get('use_lagrange', True),
                    target_action_gap=model_config.get('target_action_gap', 0.5),
                    safety_config=safety_config
                )
            elif model_name == 'IQL':
                agent = AgentClass(
                    state_dim=state_dim, action_dim=action_dim,
                    hidden_dims=config.get('network', {}).get('hidden_layers', [256, 256, 128]),
                    learning_rate=model_config.get('learning_rate', 0.000138),
                    gamma=config.get('training', {}).get('gamma', 0.99),
                    expectile=model_config.get('expectile', 0.9),
                    temperature=model_config.get('temperature', 3.56),
                    tau=model_config.get('tau', 0.0043),
                    safety_config=safety_config
                )
            else:  # BCQ
                agent = AgentClass(
                    state_dim=state_dim, action_dim=action_dim,
                    hidden_dims=config.get('network', {}).get('hidden_layers', [256, 256, 128]),
                    learning_rate=model_config.get('learning_rate', 3e-4),
                    gamma=config.get('training', {}).get('gamma', 0.99),
                    threshold=model_config.get('threshold', 0.3),
                    safety_config=safety_config
                )
            
            agent.load(str(checkpoint_path))
            ensemble_agents.append(agent)
            ensemble_names.append(model_name)
            ensemble_weights.append(raw_weights[model_name] / total_weight)
            print(f"    ✓ Loaded {model_name}")
    
    if len(ensemble_agents) >= 2:
        print(f"\nCreating ensemble with {len(ensemble_agents)} agents...")
        # Create ensemble
        ensemble = GeneralEnsembleAgent(
            agents=ensemble_agents,
            voting_method='weighted_voting',
            weights=ensemble_weights,
            agent_names=ensemble_names
        )
        
        # Set normalization
        if hasattr(ensemble_agents[0], 'state_mean') and ensemble_agents[0].state_mean is not None:
            ensemble.set_normalization(ensemble_agents[0].state_mean, ensemble_agents[0].state_std)
        
        # Evaluate ensemble at same epochs as individual models
        # Use the epochs from the first model's training data
        first_model = list(all_training_data.keys())[0]
        ensemble_epoch_data = []
        
        print("Evaluating ensemble at training epochs...")
        for epoch_info in all_training_data[first_model]:
            epoch = epoch_info['epoch']
            eval_data = collect_evaluation_data(ensemble, val_loader)
            ensemble_epoch_info = {
                'epoch': epoch,
                'mean_reward': eval_data['mean_reward'],
                'mean_cost': eval_data['mean_cost'],
                'violation_rate': eval_data['violation_rate'],
                'action_distribution': eval_data['action_distribution'],
                'states_sample': eval_data['states_sample']
            }
            ensemble_epoch_data.append(ensemble_epoch_info)
            print(f"  Epoch {epoch}: Reward={ensemble_epoch_info['mean_reward']:.4f}, "
                  f"Violations={ensemble_epoch_info['violation_rate']:.2%}")
        
        all_training_data['Ensemble'] = ensemble_epoch_data
        print(f"✓ Ensemble evaluation complete!")
    else:
        print("⚠ Not enough trained models for ensemble (need at least 2)")
    
    return all_training_data


def plot_reward_vs_episodes(training_data, output_dir):
    """Plot 1: Reward vs Episodes."""
    print("\nGenerating: Reward vs Episodes")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'CQL': '#2E86AB', 'IQL': '#A23B72', 'BCQ': '#F18F01', 'Ensemble': '#6A4C93'}
    
    for model_name, epoch_data in training_data.items():
        epochs = [d['epoch'] for d in epoch_data]
        rewards = [d['mean_reward'] for d in epoch_data]
        
        ax.plot(epochs, rewards, label=f'{model_name}', 
                linewidth=2.5, color=colors.get(model_name, 'gray'), marker='o', markersize=4)
    
    ax.set_xlabel('Training Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Reward', fontsize=14, fontweight='bold')
    ax.set_title('Reward vs Episodes - Learning Progress', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/reward_vs_episodes.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/reward_vs_episodes.png")


def plot_cost_violations_vs_episodes(training_data, output_dir):
    """Plot 2: Cost Violations vs Episodes."""
    print("\nGenerating: Cost Violations vs Episodes")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'CQL': '#2E86AB', 'IQL': '#A23B72', 'BCQ': '#F18F01', 'Ensemble': '#6A4C93'}
    
    for model_name, epoch_data in training_data.items():
        epochs = [d['epoch'] for d in epoch_data]
        violations = [d['violation_rate'] * 100 for d in epoch_data]  # Convert to percentage
        
        ax.plot(epochs, violations, label=f'{model_name}', 
                linewidth=2.5, color=colors.get(model_name, 'gray'), marker='s', markersize=4)
    
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Zero Violations (Target)', alpha=0.7)
    ax.set_xlabel('Training Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Constraint Violation Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Cost Violations vs Episodes - Safety Compliance', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cost_violations_vs_episodes.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/cost_violations_vs_episodes.png")


def plot_action_distribution_over_time(training_data, output_dir):
    """Plot 3: Action Distribution Over Time."""
    print("\nGenerating: Action Distribution Over Time")
    
    action_names = [
        'Inc Tx', 'Dec Tx', 'Inc OBSS', 'Dec OBSS',
        'Inc Width', 'Dec Width', 'Inc Channel', 'Dec Channel', 'No-op'
    ]
    
    # Use Ensemble if available, otherwise CQL
    if 'Ensemble' in training_data:
        model_name = 'Ensemble'
    else:
        model_name = 'CQL' if 'CQL' in training_data else list(training_data.keys())[0]
    epoch_data = training_data[model_name]
    
    # Create heatmap data
    epochs = [d['epoch'] for d in epoch_data]
    action_dist_matrix = np.array([d['action_distribution'] for d in epoch_data])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create heatmap
    im = ax.imshow(action_dist_matrix.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels([f'E{e}' for e in epochs], rotation=45, ha='right')
    ax.set_yticks(range(len(action_names)))
    ax.set_yticklabels(action_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Action Probability', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Training Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Action', fontsize=14, fontweight='bold')
    ax.set_title(f'Action Distribution Over Time - {model_name}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/action_distribution_over_time.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/action_distribution_over_time.png")
    
    # Also create stacked histogram
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Stacked bar chart
    bottom = np.zeros(len(epochs))
    colors_map = plt.cm.Set3(np.linspace(0, 1, len(action_names)))
    
    for i, action_name in enumerate(action_names):
        values = action_dist_matrix[:, i] * 100  # Convert to percentage
        ax.bar(range(len(epochs)), values, bottom=bottom, 
               label=action_name, color=colors_map[i], alpha=0.8)
        bottom += values
    
    ax.set_xlabel('Training Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Action Probability (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Action Distribution Over Time (Stacked) - {model_name}', fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels([f'E{e}' for e in epochs], rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/action_distribution_stacked.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/action_distribution_stacked.png")


def plot_state_wise_performance_heatmap(training_data, output_dir):
    """Plot 4: State-wise Performance Heatmap."""
    print("\nGenerating: State-wise Performance Heatmap")
    
    # Use Ensemble if available, otherwise CQL
    if 'Ensemble' in training_data:
        model_name = 'Ensemble'
    else:
        model_name = 'CQL' if 'CQL' in training_data else list(training_data.keys())[0]
    epoch_data = training_data[model_name]
    
    # Collect states and rewards from all epochs
    all_states = []
    all_rewards = []
    
    for epoch_info in epoch_data:
        if epoch_info['states_sample'] is not None:
            states = epoch_info['states_sample']
            # Sample subset for visualization (max 200 per epoch)
            n_sample = min(200, len(states))
            indices = np.random.choice(len(states), n_sample, replace=False)
            all_states.append(states[indices])
            # Use mean reward for this epoch
            all_rewards.extend([epoch_info['mean_reward']] * n_sample)
    
    if not all_states:
        print("  ⚠ No state data available, skipping heatmap")
        return
    
    all_states = np.vstack(all_states)
    
    # Create state bins for heatmap
    # Use key features: RSSI (idx 1), Throughput (idx 5)
    feature_indices = [1, 5]  # median_rssi, avg_throughput
    feature_names = ['RSSI (dBm)', 'Throughput (Mbps)']
    
    # Bin states
    n_bins = 8
    rssi_values = all_states[:, 1]
    tput_values = all_states[:, 5]
    
    rssi_bins = np.linspace(rssi_values.min(), rssi_values.max(), n_bins + 1)
    tput_bins = np.linspace(tput_values.min(), tput_values.max(), n_bins + 1)
    
    rssi_bin_indices = np.digitize(rssi_values, rssi_bins) - 1
    tput_bin_indices = np.digitize(tput_values, tput_bins) - 1
    rssi_bin_indices = np.clip(rssi_bin_indices, 0, n_bins - 1)
    tput_bin_indices = np.clip(tput_bin_indices, 0, n_bins - 1)
    
    # Create 2D heatmap: RSSI vs Throughput
    heatmap_data = np.zeros((n_bins, n_bins))
    counts = np.zeros((n_bins, n_bins))
    
    for rssi_bin, tput_bin, reward in zip(rssi_bin_indices, tput_bin_indices, all_rewards):
        heatmap_data[rssi_bin, tput_bin] += reward
        counts[rssi_bin, tput_bin] += 1
    
    # Average rewards
    heatmap_data = np.divide(heatmap_data, counts, out=np.zeros_like(heatmap_data), where=counts!=0)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', interpolation='nearest')
    
    # Set labels
    rssi_labels = [f'{rssi_bins[i]:.1f}' for i in range(0, len(rssi_bins)-1, max(1, (n_bins-1)//4))]
    tput_labels = [f'{tput_bins[i]:.1f}' for i in range(0, len(tput_bins)-1, max(1, (n_bins-1)//4))]
    
    ax.set_xticks(range(0, n_bins, max(1, n_bins//4)))
    ax.set_xticklabels([f'{tput_bins[i]:.1f}' for i in range(0, n_bins, max(1, n_bins//4))], rotation=45, ha='right')
    ax.set_yticks(range(0, n_bins, max(1, n_bins//4)))
    ax.set_yticklabels([f'{rssi_bins[i]:.1f}' for i in range(0, n_bins, max(1, n_bins//4))])
    
    ax.set_xlabel('Average Throughput (Mbps)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Median RSSI (dBm)', fontsize=14, fontweight='bold')
    ax.set_title(f'State-wise Performance Heatmap - {model_name}\n(Reward by Network Conditions)', 
                 fontsize=16, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Reward', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/state_wise_performance_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/state_wise_performance_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description='Generate training plots from real data')
    parser.add_argument('--dataset', default='data/rrm_dataset_expanded.h5',
                       help='Path to dataset file')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--eval-freq', type=int, default=5,
                       help='Evaluation frequency (default: 5)')
    parser.add_argument('--output-dir', default='plots',
                       help='Directory to save plots')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training, use existing data')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if dataset exists
    if not Path(args.dataset).exists():
        print(f"ERROR: Dataset not found: {args.dataset}")
        print("Please generate dataset first: python generate_dataset.py")
        sys.exit(1)
    
    # Train and collect data
    if not args.skip_training:
        print("\n" + "="*60)
        print("STEP 1: TRAINING MODELS AND COLLECTING DATA")
        print("="*60)
        training_data = train_with_data_collection(
            args.dataset, args.config, args.epochs, args.eval_freq
        )
        
        # Save training data
        output_data = {}
        for model_name, epoch_data in training_data.items():
            output_data[model_name] = []
            for d in epoch_data:
                output_data[model_name].append({
                    'epoch': d['epoch'],
                    'mean_reward': float(d['mean_reward']),
                    'mean_cost': float(d['mean_cost']),
                    'violation_rate': float(d['violation_rate']),
                    'action_distribution': d['action_distribution'].tolist()
                })
        
        with open(f'{args.output_dir}/training_data.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Training data saved to: {args.output_dir}/training_data.json")
    else:
        # Load existing data
        data_path = Path(args.output_dir) / 'training_data.json'
        if not data_path.exists():
            print(f"ERROR: No existing training data found at {data_path}")
            sys.exit(1)
        with open(data_path, 'r') as f:
            output_data = json.load(f)
        
        # Convert back to format expected by plotting functions
        training_data = {}
        for model_name, epoch_list in output_data.items():
            training_data[model_name] = []
            for d in epoch_list:
                training_data[model_name].append({
                    'epoch': d['epoch'],
                    'mean_reward': d['mean_reward'],
                    'mean_cost': d['mean_cost'],
                    'violation_rate': d['violation_rate'],
                    'action_distribution': np.array(d['action_distribution']),
                    'states_sample': None  # Not saved
                })
    
    # Generate plots
    print("\n" + "="*60)
    print("STEP 2: GENERATING PLOTS")
    print("="*60)
    
    plot_reward_vs_episodes(training_data, args.output_dir)
    plot_cost_violations_vs_episodes(training_data, args.output_dir)
    plot_action_distribution_over_time(training_data, args.output_dir)
    plot_state_wise_performance_heatmap(training_data, args.output_dir)
    
    print("\n" + "="*60)
    print("✓ ALL PLOTS GENERATED!")
    print("="*60)
    print(f"\nPlots saved to: {args.output_dir}/")
    print("  - reward_vs_episodes.png")
    print("  - cost_violations_vs_episodes.png")
    print("  - action_distribution_over_time.png")
    print("  - action_distribution_stacked.png")
    print("  - state_wise_performance_heatmap.png")


if __name__ == '__main__':
    main()
