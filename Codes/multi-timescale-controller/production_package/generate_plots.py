#!/usr/bin/env python3
"""
Plot Generation Script for Production Package

Generates performance plots from training and evaluation results.
Usage: python generate_plots.py [--checkpoint-dir checkpoints] [--output-dir plots]
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Generate plots')
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                       help='Directory containing training results')
    parser.add_argument('--results-dir', default='results',
                       help='Directory containing evaluation results')
    parser.add_argument('--output-dir', default='plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    print("="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load training results
    results = {}
    checkpoint_dir = Path(args.checkpoint_dir)
    
    for model in ['CQL', 'IQL', 'BCQ']:
        results_path = checkpoint_dir / model / 'results.json'
        if results_path.exists():
            with open(results_path, 'r') as f:
                results[model] = json.load(f)
    
    if not results:
        print(f"WARNING: No training results found in {args.checkpoint_dir}")
        print("Plots will be limited.")
    
    # 1. Training Loss Curve
    if results:
        print("Generating training loss plot...")
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, data in results.items():
            if 'train_loss' in data:
                epochs = range(len(data['train_loss']))
                ax.plot(epochs, data['train_loss'], label=f'{model} Train', linewidth=2)
                if 'val_loss' in data:
                    ax.plot(epochs, data['val_loss'], label=f'{model} Val', linestyle='--', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/training_loss.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {args.output_dir}/training_loss.png")
    
    # 2. Model Comparison
    if results:
        print("Generating model comparison plot...")
        fig, ax = plt.subplots(figsize=(10, 6))
        models = list(results.keys())
        safety = [results[m].get('safety', 0) for m in models]
        diversity = [results[m].get('diversity', 0) for m in models]
        x = np.arange(len(models))
        width = 0.35
        ax.bar(x - width/2, safety, width, label='Safety', alpha=0.8, color='green')
        ax.bar(x + width/2, diversity, width, label='Diversity', alpha=0.8, color='blue')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Comparison: Safety vs Diversity', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {args.output_dir}/model_comparison.png")
    
    # 3. Action Distribution
    if results:
        print("Generating action distribution plot...")
        fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
        if len(results) == 1:
            axes = [axes]
        for idx, (model, data) in enumerate(results.items()):
            if 'action_distribution' in data:
                action_dist = data['action_distribution']
                axes[idx].bar(range(len(action_dist)), action_dist, alpha=0.7)
                axes[idx].set_title(f'{model} Action Distribution', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel('Action ID', fontsize=10)
                axes[idx].set_ylabel('Probability', fontsize=10)
                axes[idx].grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/action_distribution.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {args.output_dir}/action_distribution.png")
    
    print("\n" + "="*60)
    print("✓ PLOT GENERATION COMPLETE!")
    print("="*60)
    print(f"\nPlots saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()

