"""
Evaluation and visualization for trained GNN model (Regression Focus).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple, Dict
import os
from scipy import stats

from gnn_data import SimulationDataLoader
from gnn_model import InterferenceGNN
from torch_geometric.data import Data


def load_trained_model(model_path: str, device: str = 'cpu') -> InterferenceGNN:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = InterferenceGNN(
        in_channels=11,
        hidden_channels=32,
        num_layers=3,  # EdgeConv
        dropout=0.2
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    return model


def predict_graph(model: InterferenceGNN, data: Data, device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict interference weights for a single graph.
    
    Args:
        model: Trained GNN model
        data: PyG Data object
        device: Device to run on
        
    Returns:
        edge_index: (2, num_edges) array of edge indices
        pred_weights: (num_edges,) array of predicted weights
        true_weights: (num_edges,) array of ground truth weights
    """
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        # Get node embeddings
        node_embeddings = model(data)
        
        # We only care about edges that exist in the ground truth for evaluation
        # (Since we are evaluating regression performance on known interference links)
        if hasattr(data, 'y_edge_index'):
            edge_index = data.y_edge_index
            true_weights = data.y_edge_attr
        else:
            edge_index = data.edge_index
            true_weights = data.edge_attr
            
        # Predict weights for these edges
        pred_weights = model.predict_edges(node_embeddings, edge_index)
    
    # Convert to numpy
    edge_index_np = edge_index.cpu().numpy()
    pred_weights_np = pred_weights.cpu().numpy().flatten()
    true_weights_np = true_weights.cpu().numpy().flatten()
    
    return edge_index_np, pred_weights_np, true_weights_np


def visualize_scatter_plot(all_preds: np.ndarray, all_labels: np.ndarray, save_path: str = None):
    """
    Create a scatter plot of Predicted vs True weights.
    """
    plt.figure(figsize=(10, 8))
    
    # Calculate density for coloring
    try:
        xy = np.vstack([all_labels, all_preds])
        z = stats.gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = all_labels[idx], all_preds[idx], z[idx]
        plt.scatter(x, y, c=z, s=20, cmap='viridis', alpha=0.6)
        plt.colorbar(label='Density')
    except:
        plt.scatter(all_labels, all_preds, alpha=0.5, c='blue')
    
    # Perfect prediction line
    min_val = min(all_labels.min(), all_preds.min())
    max_val = max(all_labels.max(), all_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Ground Truth Weight', fontsize=12)
    plt.ylabel('Predicted Weight', fontsize=12)
    plt.title('Interference Weight Prediction: Ground Truth vs Predicted', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add metrics to plot
    mse = np.mean((all_preds - all_labels) ** 2)
    r2 = 1 - (np.sum((all_labels - all_preds) ** 2) / np.sum((all_labels - np.mean(all_labels)) ** 2))
    plt.text(0.05, 0.95, f'MSE: {mse:.4f}\n$R^2$: {r2:.3f}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Scatter plot saved to {save_path}")
    
    plt.close()


def compute_regression_metrics(all_preds: np.ndarray, all_labels: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.
    """
    mse = np.mean((all_preds - all_labels) ** 2)
    mae = np.mean(np.abs(all_preds - all_labels))
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((all_labels - all_preds) ** 2)
    ss_tot = np.sum((all_labels - np.mean(all_labels)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    correlation = np.corrcoef(all_preds, all_labels)[0, 1] if len(all_preds) > 1 else 0.0
    
    # Stratified Error Analysis
    low_mask = all_labels < 0.2
    mid_mask = (all_labels >= 0.2) & (all_labels < 0.5)
    high_mask = all_labels >= 0.5
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'correlation': correlation,
        'mae_low': np.mean(np.abs(all_preds[low_mask] - all_labels[low_mask])) if low_mask.any() else 0.0,
        'mae_mid': np.mean(np.abs(all_preds[mid_mask] - all_labels[mid_mask])) if mid_mask.any() else 0.0,
        'mae_high': np.mean(np.abs(all_preds[high_mask] - all_labels[high_mask])) if high_mask.any() else 0.0,
        'count_low': low_mask.sum(),
        'count_mid': mid_mask.sum(),
        'count_high': high_mask.sum()
    }


def main():
    """Main evaluation pipeline."""
    print("="*70)
    print("GNN Model Evaluation (Regression Focus)")
    print("="*70)
    
    # Configuration
    model_path = 'models/best_model.pt'
    log_dir = 'logs'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print("\n[1/4] Loading trained model...")
    model = load_trained_model(model_path, device=device)
    
    # Load data
    print("\n[2/4] Loading test data...")
    loader = SimulationDataLoader(log_dir)
    loader.load_logs()
    dataset = loader.create_dataset()
    
    # Use last 15% as test set
    test_start = int(len(dataset) * 0.85)
    test_data = dataset[test_start:]
    print(f"Test set: {len(test_data)} graphs")
    
    # Evaluate
    print(f"\n[3/4] Evaluating on test set...")
    all_preds = []
    all_labels = []
    
    for data in test_data:
        _, pred_w, true_w = predict_graph(model, data, device)
        all_preds.extend(pred_w)
        all_labels.extend(true_w)
        
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute Metrics
    print("\n[4/4] Computing metrics and visualizations...")
    metrics = compute_regression_metrics(all_preds, all_labels)
    
    print("\n" + "="*70)
    print("Test Set Results")
    print("="*70)
    print(f"MSE:         {metrics['mse']:.6f}")
    print(f"MAE:         {metrics['mae']:.6f}")
    print(f"RMSE:        {metrics['rmse']:.6f}")
    print(f"R² Score:    {metrics['r2']:.4f}")
    print(f"Correlation: {metrics['correlation']:.4f}")
    print("-" * 40)
    print("Stratified MAE:")
    print(f"  Low (<0.2):      {metrics['mae_low']:.6f} ({metrics['count_low']} edges)")
    print(f"  Mid (0.2-0.5):   {metrics['mae_mid']:.6f} ({metrics['count_mid']} edges)")
    print(f"  High (>=0.5):    {metrics['mae_high']:.6f} ({metrics['count_high']} edges)")
    print("="*70)
    
    # Visualize
    visualize_scatter_plot(all_preds, all_labels, save_path='models/prediction_scatter.png')


if __name__ == "__main__":
    main()
