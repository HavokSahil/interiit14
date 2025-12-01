"""
Evaluation and visualization for trained GNN model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple
import os

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
        in_channels=9,
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


def predict_graph(model: InterferenceGNN, data: Data, 
                 threshold: float = 0.5, device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict interference graph for a single timestep.
    
    Args:
        model: Trained GNN model
        data: PyG Data object
        threshold: Probability threshold for edge existence
        device: Device to run on
        
    Returns:
        predicted_edges: (2, num_edges) array of predicted edges
        edge_probs: (num_edges,) array of edge probabilities
    """
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        edge_index, edge_probs = model.predict_all_edges(data)
    
    # Filter by threshold
    edge_probs = edge_probs.cpu().numpy().flatten()
    edge_index = edge_index.cpu().numpy()
    
    mask = edge_probs >= threshold
    predicted_edges = edge_index[:, mask]
    filtered_probs = edge_probs[mask]
    
    return predicted_edges, filtered_probs


def visualize_graph_comparison(data: Data, predicted_edges: np.ndarray,
                               edge_probs: np.ndarray, save_path: str = None):
    """
    Visualize ground truth vs predicted interference graph.
    
    Args:
        data: PyG Data object with ground truth
        predicted_edges: Predicted edges (2, num_edges)
        edge_probs: Edge probabilities
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    num_nodes = data.num_nodes
    
    # Ground truth graph
    G_true = nx.DiGraph()
    G_true.add_nodes_from(range(num_nodes))
    
    if hasattr(data, 'y_edge_index'):
        true_edges = data.y_edge_index.cpu().numpy().T
    else:
        true_edges = data.edge_index.cpu().numpy().T
        
    for i, j in true_edges:
        G_true.add_edge(i, j)
    
    # Predicted graph
    G_pred = nx.DiGraph()
    G_pred.add_nodes_from(range(num_nodes))
    pred_edges = predicted_edges.T
    for (i, j), prob in zip(pred_edges, edge_probs):
        G_pred.add_edge(i, j, weight=prob)
    
    # Use same layout for both
    pos = nx.spring_layout(G_true, seed=42, k=2)
    
    # Plot ground truth
    ax = axes[0]
    nx.draw_networkx_nodes(G_true, pos, node_color='lightblue', 
                          node_size=800, ax=ax)
    nx.draw_networkx_edges(G_true, pos, edge_color='gray', 
                          arrows=True, arrowsize=20, ax=ax, width=2)
    nx.draw_networkx_labels(G_true, pos, font_size=12, ax=ax)
    ax.set_title(f'Ground Truth Graph\n({G_true.number_of_edges()} edges)', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Plot predictions
    ax = axes[1]
    nx.draw_networkx_nodes(G_pred, pos, node_color='lightcoral', 
                          node_size=800, ax=ax)
    
    # Color edges by probability
    edges = list(G_pred.edges())
    if len(edges) > 0:
        weights = [G_pred[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G_pred, pos, edgelist=edges, 
                              edge_color=weights, edge_cmap=plt.cm.RdYlGn,
                              arrows=True, arrowsize=20, ax=ax, width=2, 
                              edge_vmin=0.5, edge_vmax=1.0)
    
    nx.draw_networkx_labels(G_pred, pos, font_size=12, ax=ax)
    ax.set_title(f'Predicted Graph\n({G_pred.number_of_edges()} edges)', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add colorbar for edge probabilities
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                               norm=plt.Normalize(vmin=0.5, vmax=1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Edge Probability', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def compute_metrics(data: Data, predicted_edges: np.ndarray, edge_probs: np.ndarray) -> dict:
    """
    Compute evaluation metrics comparing predicted vs ground truth.
    
    Args:
        data: PyG Data with ground truth edges
        predicted_edges: Predicted edges (2, num_edges)
        edge_probs: Predicted edge weights
        
    Returns:
        Dictionary of metrics
    """
    # Get ground truth edges and weights
    if hasattr(data, 'y_edge_index'):
        true_edge_index = data.y_edge_index.cpu().numpy()
        true_weights = data.y_edge_attr.cpu().numpy().flatten() if data.y_edge_attr is not None else np.ones(true_edge_index.shape[1])
    else:
        # Fallback
        true_edge_index = data.edge_index.cpu().numpy()
        true_weights = data.edge_attr.cpu().numpy().flatten() if data.edge_attr is not None else np.ones(true_edge_index.shape[1])
    
    # Create dictionaries for fast lookup
    true_edge_dict = {}
    for i in range(true_edge_index.shape[1]):
        u, v = true_edge_index[0, i], true_edge_index[1, i]
        true_edge_dict[(u, v)] = true_weights[i]
        
    pred_edge_dict = {}
    for i in range(predicted_edges.shape[1]):
        u, v = predicted_edges[0, i], predicted_edges[1, i]
        pred_edge_dict[(u, v)] = edge_probs[i]
        
    # Compute MSE/MAE over all possible edges (or just the ones in either set)
    # Here we compute over the union of edges
    all_edges = set(true_edge_dict.keys()) | set(pred_edge_dict.keys())
    
    squared_error = 0.0
    abs_error = 0.0
    
    for u, v in all_edges:
        true_w = true_edge_dict.get((u, v), 0.0)
        pred_w = pred_edge_dict.get((u, v), 0.0)
        
        squared_error += (true_w - pred_w) ** 2
        abs_error += abs(true_w - pred_w)
        
    mse = squared_error / len(all_edges) if all_edges else 0.0
    mae = abs_error / len(all_edges) if all_edges else 0.0
    
    # Detection metrics (binary classification at threshold 0.1)
    tp = 0
    fp = 0
    fn = 0
    
    for u, v in all_edges:
        true_exists = true_edge_dict.get((u, v), 0.0) > 0
        pred_exists = pred_edge_dict.get((u, v), 0.0) > 0.1
        
        if true_exists and pred_exists:
            tp += 1
        elif not true_exists and pred_exists:
            fp += 1
        elif true_exists and not pred_exists:
            fn += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'num_edges': len(all_edges),
        'mse': mse,
        'mae': mae,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_true_edges': len(true_edge_dict),
        'num_pred_edges': len(pred_edge_dict)
    }


def main():
    """Main evaluation pipeline."""
    print("="*70)
    print("GNN Model Evaluation")
    print("="*70)
    
    # Configuration
    model_path = 'models/best_model.pt'
    log_dir = 'logs'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    threshold = 0.2
    
    # Load model
    print("\n[1/4] Loading trained model...")
    model = load_trained_model(model_path, device=device)
    
    # Load data
    print("\n[2/4] Loading test data...")
    loader = SimulationDataLoader(log_dir)
    loader.load_logs()
    dataset = loader.create_dataset()
    
    # Use last 15% as test set (matching train script)
    test_start = int(len(dataset) * 0.85)
    test_data = dataset[test_start:]
    print(f"Test set: {len(test_data)} graphs")
    
    # Evaluate on test set
    print(f"\n[3/4] Evaluating on test set (threshold={threshold})...")
    all_metrics = []
    
    for i, data in enumerate(test_data):
        pred_edges, edge_probs = predict_graph(model, data, threshold, device)
        metrics = compute_metrics(data, pred_edges, edge_probs)
        all_metrics.append(metrics)
        
        if i < 3:  # Visualize first 3 samples
            print(f"\n  Sample {i+1}:")
            print(f"    True edges: {metrics['num_true_edges']}, Predicted: {metrics['num_pred_edges']}")
            print(f"    Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")
            
            visualize_graph_comparison(
                data, pred_edges, edge_probs,
                save_path=f'models/prediction_sample_{i+1}.png'
            )
    
    # Aggregate metrics
    print("\n[4/4] Computing aggregate metrics...")
    avg_metrics = {
        'mse': np.mean([m['mse'] for m in all_metrics]),
        'mae': np.mean([m['mae'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1': np.mean([m['f1'] for m in all_metrics])
    }
    
    print("\n" + "="*70)
    print("Test Set Results (Averaged)")
    print("="*70)
    print(f"MSE:       {avg_metrics['mse']:.4f}")
    print(f"MAE:       {avg_metrics['mae']:.4f}")
    print(f"Precision: {avg_metrics['precision']:.3f} (threshold 0.1)")
    print(f"Recall:    {avg_metrics['recall']:.3f} (threshold 0.1)")
    print(f"F1 Score:  {avg_metrics['f1']:.3f} (threshold 0.1)")
    print("="*70)


if __name__ == "__main__":
    main()
