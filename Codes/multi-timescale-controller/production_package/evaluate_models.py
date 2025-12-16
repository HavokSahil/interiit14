#!/usr/bin/env python3
"""
Evaluation Script for Production Package

Evaluates trained models and generates metrics.
Usage: python evaluate_models.py [--dataset data/rrm_dataset_expanded.h5]
"""

import argparse
import yaml
from pathlib import Path
import sys

from src.dataset.dataset import create_dataloaders
from src.agents.cql import CQLAgent
from src.agents.iql import IQLAgent
from src.agents.bcq import BCQAgent
from src.training.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--dataset', default='data/rrm_dataset_expanded.h5',
                       help='Path to dataset file')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                       help='Directory containing trained models')
    parser.add_argument('--output-dir', default='results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    print("="*60)
    print("EVALUATING MODELS")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if dataset exists
    if not Path(args.dataset).exists():
        print(f"ERROR: Dataset not found: {args.dataset}")
        sys.exit(1)
    
    # Load test set
    batch_size = config.get('training', {}).get('batch_size', 256)
    _, _, test_loader = create_dataloaders(args.dataset, batch_size=batch_size)
    test_dataset = test_loader.dataset
    state_dim = test_dataset.num_features
    action_dim = test_dataset.num_actions
    
    print(f"Test set: {len(test_dataset)} samples")
    print()
    
    # Load trained models
    agents = {}
    checkpoint_dir = Path(args.checkpoint_dir)
    
    print("Loading trained models...")
    
    # Load CQL
    cql_path = checkpoint_dir / 'CQL' / 'best.pt'
    if cql_path.exists():
        print(f"  Loading CQL from {cql_path}...")
        cql_agent = CQLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 256, 128],
            learning_rate=3e-4,
            gamma=0.99
        )
        cql_agent.load(str(cql_path))
        agents['CQL'] = cql_agent
        print("  ✓ CQL loaded")
    else:
        print(f"  ⚠ CQL checkpoint not found: {cql_path}")
    
    # Load IQL
    iql_path = checkpoint_dir / 'IQL' / 'best.pt'
    if iql_path.exists():
        print(f"  Loading IQL from {iql_path}...")
        iql_config = config.get('iql', {})
        iql_agent = IQLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 256, 128],
            learning_rate=iql_config.get('learning_rate', 0.000138),
            gamma=0.99,
            expectile=iql_config.get('expectile', 0.9),
            temperature=iql_config.get('temperature', 3.56),
            tau=iql_config.get('tau', 0.0043)
        )
        iql_agent.load(str(iql_path))
        agents['IQL'] = iql_agent
        print("  ✓ IQL loaded")
    else:
        print(f"  ⚠ IQL checkpoint not found: {iql_path}")
    
    # Load BCQ
    bcq_path = checkpoint_dir / 'BCQ' / 'best.pt'
    if bcq_path.exists():
        print(f"  Loading BCQ from {bcq_path}...")
        bcq_agent = BCQAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 256, 128],
            learning_rate=3e-4,
            gamma=0.99,
            threshold=0.3
        )
        bcq_agent.load(str(bcq_path))
        agents['BCQ'] = bcq_agent
        print("  ✓ BCQ loaded")
    else:
        print(f"  ⚠ BCQ checkpoint not found: {bcq_path}")
    
    if not agents:
        print("\nERROR: No trained models found!")
        print(f"Please train models first using: python train_models.py")
        sys.exit(1)
    
    print(f"\nEvaluating {len(agents)} model(s)...")
    print()
    
    # Evaluate
    evaluator = Evaluator(agents, test_loader, save_dir=args.output_dir)
    results = evaluator.evaluate_all()
    
    # Print results
    print("="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for name, metrics in results.items():
        print(f"\n{name}:")
        # Safety metrics
        safety_metrics = metrics.get('safety', {})
        violation_rate = safety_metrics.get('constraint_violation_rate', 0)
        safety = 1.0 - violation_rate
        print(f"  Safety: {safety:.2%}")
        print(f"  Violation Rate: {violation_rate:.2%}")
        
        # Action metrics
        action_metrics = metrics.get('actions', {})
        unsafe_attempt_rate = action_metrics.get('unsafe_attempt_rate', 0)
        action_dist = action_metrics.get('distribution', [])
        diversity = 1.0 - max(action_dist) if action_dist else 0.0  # 1 - max probability
        print(f"  Diversity: {diversity:.2%}")
        print(f"  Unsafe Attempt Rate: {unsafe_attempt_rate:.2%}")
        
        # Performance metrics
        perf_metrics = metrics.get('performance', {})
        mean_q = perf_metrics.get('mean_q', 0)
        simulated_reward = perf_metrics.get('simulated_reward', 0)
        if mean_q:
            print(f"  Mean Q-value: {mean_q:.4f}")
        if simulated_reward:
            print(f"  Simulated Reward: {simulated_reward:.4f}")
        
        if action_dist:
            print(f"  Action Distribution: {[f'{p:.2%}' for p in action_dist[:9]]}")
    
    print("\n" + "="*60)
    print("✓ EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()

