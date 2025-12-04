#!/usr/bin/env python3
"""
Dataset Generation Script for Production Package

Generates 50K dataset with 9 actions.
Usage: python generate_dataset.py [--samples 50000] [--output data/rrm_dataset_expanded.h5]
"""

import argparse
from pathlib import Path
from src.dataset.generator import RRMDatasetGenerator


def main():
    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('--samples', type=int, default=50000,
                       help='Number of samples to generate')
    parser.add_argument('--output', default='data/rrm_dataset_expanded.h5',
                       help='Output dataset path')
    parser.add_argument('--augmentation', action='store_true',
                       help='Use data augmentation')
    parser.add_argument('--aug-factor', type=float, default=1.0,
                       help='Augmentation factor')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("="*60)
    print("GENERATING DATASET")
    print("="*60)
    print(f"Target samples: {args.samples}")
    print(f"Augmentation: {args.augmentation}")
    print(f"Output: {args.output}")
    print()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset
    generator = RRMDatasetGenerator(seed=args.seed, use_augmentation=args.augmentation)
    dataset = generator.generate_dataset(
        num_samples=args.samples,
        use_augmentation=args.augmentation,
        augmentation_factor=args.aug_factor if args.augmentation else 0.0
    )
    
    # Save dataset
    generator.save_dataset(dataset, args.output)
    
    print("="*60)
    print("✓ DATASET GENERATION COMPLETE!")
    print("="*60)
    print(f"\nDataset saved to: {args.output}")
    print(f"Total samples: {len(dataset['states'])}")
    print(f"Features: {dataset['states'].shape[1]}")
    print(f"Actions: {dataset['actions'].max() + 1}")


if __name__ == '__main__':
    main()

