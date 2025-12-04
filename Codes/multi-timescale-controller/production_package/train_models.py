#!/usr/bin/env python3
"""
Training Script for Production Package

Trains CQL, IQL, and BCQ models on the dataset.
Usage: python train_models.py [--epochs 100] [--dataset data/rrm_dataset_expanded.h5]
"""

import argparse
import yaml
from pathlib import Path
import sys

from src.dataset.dataset import create_dataloaders
from src.agents.cql import CQLAgent
from src.agents.iql import IQLAgent
from src.agents.bcq import BCQAgent
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train all models')
    parser.add_argument('--dataset', default='data/rrm_dataset_expanded.h5',
                       help='Path to dataset file')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--save-dir', default='checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Load config
    print("="*60)
    print("TRAINING MODELS")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Config: {args.config}")
    print(f"Epochs: {args.epochs}")
    print(f"Save directory: {args.save_dir}")
    print()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if dataset exists
    if not Path(args.dataset).exists():
        print(f"ERROR: Dataset not found: {args.dataset}")
        print("Please generate dataset first using: python generate_dataset.py")
        sys.exit(1)
    
    # Create dataloaders
    batch_size = config.get('training', {}).get('batch_size', 256)
    print(f"Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(args.dataset, batch_size=batch_size)
    train_dataset = train_loader.dataset
    print(train_loader.dataset)
    print(val_loader.dataset)
    print(test_loader.dataset)
    state_dim = train_dataset.num_features
    action_dim = train_dataset.num_actions
    
    print(f"Dataset loaded: {len(train_dataset)} samples")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    print()
    
    # Train CQL
    print("="*60)
    print("[1/3] Training CQL (Conservative Q-Learning)")
    print("="*60)
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
        target_action_gap=cql_config.get('target_action_gap', 0.5)
    )
    cql_trainer = Trainer(
        cql_agent, train_loader, val_loader, test_loader,
        config=config, save_dir=args.save_dir, experiment_name='CQL'
    )
    cql_results = cql_trainer.train(epochs=args.epochs)
    print(f"✓ CQL training complete!")
    print()
    
    # Train IQL
    print("="*60)
    print("[2/3] Training IQL (Implicit Q-Learning)")
    print("="*60)
    iql_config = config.get('iql', {})
    iql_agent = IQLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config.get('network', {}).get('hidden_layers', [256, 256, 128]),
        learning_rate=iql_config.get('learning_rate', 0.000138),
        gamma=config.get('training', {}).get('gamma', 0.99),
        expectile=iql_config.get('expectile', 0.9),
        temperature=iql_config.get('temperature', 3.56),
        tau=iql_config.get('tau', 0.0043)
    )
    iql_trainer = Trainer(
        iql_agent, train_loader, val_loader, test_loader,
        config=config, save_dir=args.save_dir, experiment_name='IQL'
    )
    iql_results = iql_trainer.train(epochs=args.epochs)
    print(f"✓ IQL training complete!")
    print()
    
    # Train BCQ
    print("="*60)
    print("[3/3] Training BCQ (Batch-Constrained Q-Learning)")
    print("="*60)
    bcq_config = config.get('bcq', {})
    bcq_agent = BCQAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config.get('network', {}).get('hidden_layers', [256, 256, 128]),
        learning_rate=bcq_config.get('learning_rate', 3e-4),
        gamma=config.get('training', {}).get('gamma', 0.99),
        threshold=bcq_config.get('threshold', 0.3)
    )
    bcq_trainer = Trainer(
        bcq_agent, train_loader, val_loader, test_loader,
        config=config, save_dir=args.save_dir, experiment_name='BCQ'
    )
    bcq_results = bcq_trainer.train(epochs=args.epochs)
    print(f"✓ BCQ training complete!")
    print()
    
    print("="*60)
    print("✓ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*60)
    print(f"\nCheckpoints saved in: {args.save_dir}/")
    print("  - CQL: checkpoints/CQL/best.pt")
    print("  - IQL: checkpoints/IQL/best.pt")
    print("  - BCQ: checkpoints/BCQ/best.pt")


if __name__ == '__main__':
    main()

