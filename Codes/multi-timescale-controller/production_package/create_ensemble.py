#!/usr/bin/env python3
"""
Create Ensemble Model Script

Creates and saves the ensemble model from trained individual models.
Based on Model Config.txt specifications.

Usage:
    python create_ensemble.py [--checkpoint-dir checkpoints] [--model-dir model] [--config config/config.yaml]
"""

import argparse
import sys
import json
import yaml
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.general_ensemble import GeneralEnsembleAgent
from src.agents.cql import CQLAgent
from src.agents.iql import IQLAgent
from src.agents.bcq import BCQAgent
from src.agents.safety import SafetyConfig


def load_trained_agents(checkpoint_dir, config_path, state_dim=15, action_dim=9):
    """Load trained CQL, IQL, and BCQ agents."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create safety config
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
    
    agents = []
    agent_names = []
    
    # Load CQL
    print("Loading CQL model...")
    cql_config = config.get('cql', {})
    cql_agent = CQLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config.get('network', {}).get('hidden_layers', [256, 256, 128]),
        learning_rate=config.get('training', {}).get('learning_rate', 3e-4),
        gamma=config.get('training', {}).get('gamma', 0.99),
        alpha=cql_config.get('alpha', 0.2),
        min_q_weight=cql_config.get('min_q_weight', 0.1),
        target_update_freq=cql_config.get('target_update_freq', 100),
        use_lagrange=cql_config.get('use_lagrange', True),
        target_action_gap=cql_config.get('target_action_gap', 0.5),
        safety_config=safety_config
    )
    cql_path = Path(checkpoint_dir) / 'CQL' / 'best.pt'
    if cql_path.exists():
        cql_agent.load(str(cql_path))
        agents.append(cql_agent)
        agent_names.append('CQL')
        print(f"  ✓ Loaded from: {cql_path}")
    else:
        print(f"  ✗ Not found: {cql_path}")
        sys.exit(1)
    
    # Load IQL
    print("Loading IQL model...")
    iql_config = config.get('iql', {})
    iql_agent = IQLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config.get('network', {}).get('hidden_layers', [256, 256, 128]),
        learning_rate=iql_config.get('learning_rate', 0.000138),
        gamma=config.get('training', {}).get('gamma', 0.99),
        expectile=iql_config.get('expectile', 0.9),
        temperature=iql_config.get('temperature', 3.56),
        tau=iql_config.get('tau', 0.0043),
        safety_config=safety_config
    )
    iql_path = Path(checkpoint_dir) / 'IQL' / 'best.pt'
    if iql_path.exists():
        iql_agent.load(str(iql_path))
        agents.append(iql_agent)
        agent_names.append('IQL')
        print(f"  ✓ Loaded from: {iql_path}")
    else:
        print(f"  ✗ Not found: {iql_path}")
        sys.exit(1)
    
    # Load BCQ
    print("Loading BCQ model...")
    bcq_config = config.get('bcq', {})
    bcq_agent = BCQAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config.get('network', {}).get('hidden_layers', [256, 256, 128]),
        learning_rate=bcq_config.get('learning_rate', 3e-4),
        gamma=config.get('training', {}).get('gamma', 0.99),
        threshold=bcq_config.get('threshold', 0.3),
        safety_config=safety_config
    )
    bcq_path = Path(checkpoint_dir) / 'BCQ' / 'best.pt'
    if bcq_path.exists():
        bcq_agent.load(str(bcq_path))
        agents.append(bcq_agent)
        agent_names.append('BCQ')
        print(f"  ✓ Loaded from: {bcq_path}")
    else:
        print(f"  ✗ Not found: {bcq_path}")
        sys.exit(1)
    
    return agents, agent_names, safety_config


def create_ensemble(agents, agent_names, model_dir, config_path):
    """
    Create ensemble model according to Model Config.txt specifications.
    
    From Model Config.txt:
    - Composition: BCQ + IQL + CQL
    - Weights: BCQ: 56.3%, IQL: 63.3%, CQL: 14.2%
    - Voting Method: Weighted Voting
    """
    print("\n" + "="*60)
    print("CREATING ENSEMBLE MODEL")
    print("="*60)
    
    # Weights from Model Config.txt (normalized to sum to 1.0)
    # BCQ: 56.3%, IQL: 63.3%, CQL: 14.2%
    # Note: These don't sum to 100%, so we'll normalize them
    raw_weights = {
        'BCQ': 56.3,
        'IQL': 63.3,
        'CQL': 14.2
    }
    
    # Create weights list matching agent_names order
    weights = []
    total_weight = sum(raw_weights[name] for name in agent_names)
    for name in agent_names:
        weights.append(raw_weights[name] / total_weight)
    
    print(f"Agent Names: {agent_names}")
    print(f"Weights: {[f'{w*100:.1f}%' for w in weights]}")
    print(f"Voting Method: Weighted Voting")
    
    # Create ensemble
    ensemble = GeneralEnsembleAgent(
        agents=agents,
        voting_method='weighted_voting',  # From Model Config.txt
        weights=weights,
        agent_names=agent_names
    )
    
    # Set normalization from first agent
    if hasattr(agents[0], 'state_mean') and agents[0].state_mean is not None:
        ensemble.set_normalization(agents[0].state_mean, agents[0].state_std)
        print(f"\nNormalization set from {agent_names[0]}")
    
    # Save ensemble
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving ensemble to: {model_path}")
    ensemble.save(str(model_path))
    
    # Create production_metadata.json (as expected by run_inference.py)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get normalization stats if available
    norm_data = {}
    if hasattr(agents[0], 'state_mean') and agents[0].state_mean is not None:
        import torch
        mean_val = agents[0].state_mean
        std_val = agents[0].state_std
        
        # Convert to list for JSON serialization
        if isinstance(mean_val, torch.Tensor):
            mean_val = mean_val.cpu().numpy().tolist()
        elif isinstance(mean_val, np.ndarray):
            mean_val = mean_val.tolist()
        
        if isinstance(std_val, torch.Tensor):
            std_val = std_val.cpu().numpy().tolist()
        elif isinstance(std_val, np.ndarray):
            std_val = std_val.tolist()
        
        norm_data = {
            'mean': mean_val,
            'std': std_val
        }
    
    production_metadata = {
        'models': agent_names,
        'voting_method': 'weighted_voting',
        'weights': weights,
        'normalization': norm_data,
        'state_dim': 15,
        'action_dim': 9,
        'config': {
            'gamma': config.get('training', {}).get('gamma', 0.99),
            'safety': config.get('safety', {})
        }
    }
    
    metadata_path = model_path / 'production_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(production_metadata, f, indent=2)
    
    print(f"  ✓ Saved ensemble agents: {[f'agent_{i}.pt' for i in range(len(agents))]}")
    print(f"  ✓ Saved ensemble_metadata.json")
    print(f"  ✓ Saved production_metadata.json")
    
    return ensemble


def main():
    parser = argparse.ArgumentParser(description='Create Ensemble Model')
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                       help='Directory containing trained model checkpoints')
    parser.add_argument('--model-dir', default='model',
                       help='Directory to save ensemble model')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ENSEMBLE MODEL CREATION")
    print("="*60)
    print(f"Checkpoint Directory: {args.checkpoint_dir}")
    print(f"Model Directory: {args.model_dir}")
    print(f"Config: {args.config}")
    print()
    
    # Check if checkpoints exist
    checkpoint_dir = Path(args.checkpoint_dir)
    required_models = ['CQL', 'IQL', 'BCQ']
    missing = []
    for model in required_models:
        if not (checkpoint_dir / model / 'best.pt').exists():
            missing.append(model)
    
    if missing:
        print(f"ERROR: Missing trained models: {', '.join(missing)}")
        print(f"Please train models first using: python train_models.py")
        sys.exit(1)
    
    # Load agents
    agents, agent_names, safety_config = load_trained_agents(
        args.checkpoint_dir, args.config
    )
    
    # Create and save ensemble
    ensemble = create_ensemble(agents, agent_names, args.model_dir, args.config)
    
    print("\n" + "="*60)
    print("✓ ENSEMBLE MODEL CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nEnsemble saved to: {args.model_dir}/")
    print(f"  - agent_0.pt ({agent_names[0]})")
    print(f"  - agent_1.pt ({agent_names[1]})")
    print(f"  - agent_2.pt ({agent_names[2]})")
    print(f"  - ensemble_metadata.json")
    print(f"  - production_metadata.json")
    print("\nYou can now use this ensemble for inference:")
    print(f"  python run_inference.py --model-dir {args.model_dir}")


if __name__ == '__main__':
    main()

