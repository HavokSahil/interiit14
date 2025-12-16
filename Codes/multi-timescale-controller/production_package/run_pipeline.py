#!/usr/bin/env python3
"""
Main Pipeline Script for Production Package

Runs complete pipeline: Dataset Generation -> Training -> Evaluation -> Plotting -> Inference

Usage:
    python run_pipeline.py --mode all                    # Run everything
    python run_pipeline.py --mode dataset                # Generate dataset only
    python run_pipeline.py --mode train                   # Train models only
    python run_pipeline.py --mode evaluate               # Evaluate models only
    python run_pipeline.py --mode plots                  # Generate plots only
    python run_pipeline.py --mode inference              # Run inference only
"""

import argparse
import sys
from pathlib import Path
import subprocess


def run_dataset_generation(args):
    """Generate dataset."""
    print("\n" + "="*60)
    print("STEP 1: GENERATING DATASET")
    print("="*60)
    
    cmd = [
        sys.executable, "generate_dataset.py",
        "--samples", str(args.samples),
        "--output", args.dataset_path
    ]
    if args.augmentation:
        cmd.extend(["--augmentation", "--aug-factor", str(args.aug_factor)])
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: Dataset generation failed!")
        sys.exit(1)
    
    print("✓ Dataset generation complete!\n")


def run_training(args):
    """Train models."""
    print("\n" + "="*60)
    print("STEP 2: TRAINING MODELS")
    print("="*60)
    
    # Check if dataset exists
    if not Path(args.dataset_path).exists():
        print(f"ERROR: Dataset not found: {args.dataset_path}")
        print("Please generate dataset first using: python run_pipeline.py --mode dataset")
        sys.exit(1)
    
    cmd = [
        sys.executable, "train_models.py",
        "--dataset", args.dataset_path,
        "--epochs", str(args.epochs),
        "--save-dir", args.checkpoint_dir
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: Training failed!")
        sys.exit(1)
    
    print("✓ Training complete!\n")


def run_ensemble_creation(args):
    """Create ensemble model."""
    print("\n" + "="*60)
    print("STEP 3: CREATING ENSEMBLE MODEL")
    print("="*60)
    
    # Check if checkpoints exist
    checkpoint_dir = Path(args.checkpoint_dir)
    if not any((checkpoint_dir / model / "best.pt").exists() for model in ["CQL", "IQL", "BCQ"]):
        print(f"ERROR: No trained models found in {args.checkpoint_dir}")
        print("Please train models first using: python run_pipeline.py --mode train")
        sys.exit(1)
    
    cmd = [
        sys.executable, "create_ensemble.py",
        "--checkpoint-dir", args.checkpoint_dir,
        "--model-dir", args.model_dir,
        "--config", args.config_path
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: Ensemble creation failed!")
        sys.exit(1)
    
    print("✓ Ensemble creation complete!\n")


def run_evaluation(args):
    """Evaluate models."""
    print("\n" + "="*60)
    print("STEP 4: EVALUATING MODELS")
    print("="*60)
    
    # Check if dataset exists
    if not Path(args.dataset_path).exists():
        print(f"ERROR: Dataset not found: {args.dataset_path}")
        sys.exit(1)
    
    # Check if checkpoints exist
    checkpoint_dir = Path(args.checkpoint_dir)
    if not any((checkpoint_dir / model / "best.pt").exists() for model in ["CQL", "IQL", "BCQ"]):
        print(f"ERROR: No trained models found in {args.checkpoint_dir}")
        print("Please train models first using: python run_pipeline.py --mode train")
        sys.exit(1)
    
    cmd = [
        sys.executable, "evaluate_models.py",
        "--dataset", args.dataset_path,
        "--checkpoint-dir", args.checkpoint_dir,
        "--output-dir", args.results_dir
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: Evaluation failed!")
        sys.exit(1)
    
    print("✓ Evaluation complete!\n")


def run_plot_generation(args):
    """Generate plots."""
    print("\n" + "="*60)
    print("STEP 5: GENERATING PLOTS")
    print("="*60)
    
    cmd = [
        sys.executable, "generate_plots.py",
        "--checkpoint-dir", args.checkpoint_dir,
        "--results-dir", args.results_dir,
        "--output-dir", args.plots_dir
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: Plot generation failed!")
        sys.exit(1)
    
    print("✓ Plot generation complete!\n")


def run_inference(args):
    """Run inference."""
    print("\n" + "="*60)
    print("STEP 6: RUNNING INFERENCE")
    print("="*60)
    
    # Check if dataset exists
    if not Path(args.dataset_path).exists():
        print(f"ERROR: Dataset not found: {args.dataset_path}")
        sys.exit(1)
    
    # Check if model exists
    if not Path(args.model_dir).exists():
        print(f"ERROR: Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    cmd = [
        sys.executable, "run_inference.py",
        "--input", args.dataset_path,
        "--output", args.output_predictions,
        "--model-dir", args.model_dir,
        "--config", args.config_path
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: Inference failed!")
        sys.exit(1)
    
    print("✓ Inference complete!\n")


def main():
    parser = argparse.ArgumentParser(
        description='Complete Pipeline for Safe RL RRM System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --mode all                    # Run everything
  python run_pipeline.py --mode dataset --samples 50000 # Generate 50K dataset
  python run_pipeline.py --mode train --epochs 100      # Train models
  python run_pipeline.py --mode ensemble                # Create ensemble
  python run_pipeline.py --mode evaluate                # Evaluate models
  python run_pipeline.py --mode plots                   # Generate plots
  python run_pipeline.py --mode inference               # Run inference
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['all', 'dataset', 'train', 'ensemble', 'evaluate', 'plots', 'inference'],
        default='all',
        help='Pipeline mode (default: all)'
    )
    
    # Dataset generation options
    parser.add_argument('--samples', type=int, default=50000,
                       help='Number of samples to generate (default: 50000)')
    parser.add_argument('--augmentation', action='store_true',
                       help='Use data augmentation')
    parser.add_argument('--aug-factor', type=float, default=1.0,
                       help='Augmentation factor (default: 1.0)')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    
    # Paths
    parser.add_argument('--dataset-path', default='data/rrm_dataset_expanded.h5',
                       help='Path to dataset file')
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                       help='Directory for model checkpoints')
    parser.add_argument('--results-dir', default='results',
                       help='Directory for evaluation results')
    parser.add_argument('--plots-dir', default='plots',
                       help='Directory for plots')
    parser.add_argument('--model-dir', default='model',
                       help='Directory for ensemble model')
    parser.add_argument('--config-path', default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--output-predictions', default='data/predictions.csv',
                       help='Output path for predictions CSV')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SAFE RL RRM - PRODUCTION PIPELINE")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Samples: {args.samples}")
    print("="*60)
    
    # Run selected mode(s)
    if args.mode == 'all' or args.mode == 'dataset':
        run_dataset_generation(args)
    
    if args.mode == 'all' or args.mode == 'train':
        run_training(args)
    
    if args.mode == 'all' or args.mode == 'ensemble':
        run_ensemble_creation(args)
    
    if args.mode == 'all' or args.mode == 'evaluate':
        run_evaluation(args)
    
    if args.mode == 'all' or args.mode == 'plots':
        run_plot_generation(args)
    
    if args.mode == 'all' or args.mode == 'inference':
        run_inference(args)
    
    print("="*60)
    print("✓ PIPELINE COMPLETE!")
    print("="*60)
    print("\nResults:")
    print(f"  - Dataset: {args.dataset_path}")
    print(f"  - Models: {args.checkpoint_dir}/")
    print(f"  - Evaluation: {args.results_dir}/")
    print(f"  - Plots: {args.plots_dir}/")
    if args.mode == 'all' or args.mode == 'inference':
        print(f"  - Predictions: {args.output_predictions}")
    print()


if __name__ == '__main__':
    main()

