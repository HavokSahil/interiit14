"""
Training script for GNN-based interference graph prediction.
"""

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os
from datetime import datetime

from gnn_data import SimulationDataLoader
from gnn_model import InterferenceGNN, compute_edge_loss, get_positive_edges, get_negative_edges


def evaluate_model(model: InterferenceGNN, dataset: List, device: str, 
                   num_neg_per_pos: int = 2) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Note: For fully connected graphs, focuses on edge weight prediction accuracy.
    num_neg_per_pos is ignored since there are no negative edges.
    
    Args:
        model: Trained GNN model
        dataset: List of Data objects
        device: Device to run on
        num_neg_per_pos: DEPRECATED - ignored for fully connected graphs
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            
            # Compute loss
            loss = compute_edge_loss(model, data, num_neg_per_pos=num_neg_per_pos, pos_weight=1.0)
            total_loss += loss.item()
            
            # Get predictions for all edges
            node_embeddings = model(data)
            edge_index = get_positive_edges(data)
            edge_preds = model.predict_edges(node_embeddings, edge_index)
            
            # Get ground truth weights
            if hasattr(data, 'y_edge_attr') and data.y_edge_attr is not None and len(data.y_edge_attr) > 0:
                edge_labels = data.y_edge_attr
            else:
                # Skip this data point if no edge attributes
                continue
            
            # Collect predictions and labels
            all_preds.extend(edge_preds.cpu().numpy().flatten())
            all_labels.extend(edge_labels.cpu().numpy().flatten())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Edge weight prediction metrics (primary metrics for regression)
    mse = np.mean((all_preds - all_labels) ** 2)
    mae = np.mean(np.abs(all_preds - all_labels))
    rmse = np.sqrt(mse)
    
    # Correlation
    if len(all_preds) > 1 and np.std(all_preds) > 0 and np.std(all_labels) > 0:
        correlation = np.corrcoef(all_preds, all_labels)[0, 1]
    else:
        correlation = 0.0
    
    # R-squared
    ss_res = np.sum((all_labels - all_preds) ** 2)
    ss_tot = np.sum((all_labels - np.mean(all_labels)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Weight distribution analysis
    # Check how well we predict weights in different ranges
    low_weight_mask = all_labels < 0.2
    mid_weight_mask = (all_labels >= 0.2) & (all_labels < 0.5)
    high_weight_mask = all_labels >= 0.5
    
    low_mae = np.mean(np.abs(all_preds[low_weight_mask] - all_labels[low_weight_mask])) if low_weight_mask.sum() > 0 else 0.0
    mid_mae = np.mean(np.abs(all_preds[mid_weight_mask] - all_labels[mid_weight_mask])) if mid_weight_mask.sum() > 0 else 0.0
    high_mae = np.mean(np.abs(all_preds[high_weight_mask] - all_labels[high_weight_mask])) if high_weight_mask.sum() > 0 else 0.0
    
    avg_loss = total_loss / len(dataset)
    
    return {
        'loss': avg_loss,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation,
        'r2': r2,
        'pred_mean': np.mean(all_preds),
        'pred_std': np.std(all_preds),
        'label_mean': np.mean(all_labels),
        'label_std': np.std(all_labels),
        'low_weight_mae': low_mae,
        'mid_weight_mae': mid_mae,
        'high_weight_mae': high_mae,
        'low_weight_count': low_weight_mask.sum(),
        'mid_weight_count': mid_weight_mask.sum(),
        'high_weight_count': high_weight_mask.sum()
    }


def train_model(model: InterferenceGNN,
                train_data: List,
                val_data: List,
                num_epochs: int = 100,
                lr: float = 0.001,
                device: str = 'cpu',
                save_dir: str = 'models',
                num_neg_per_pos: int = 2,
                pos_weight: float = 3.0) -> Dict[str, List[float]]:
    """
    Train the GNN model.
    
    Args:
        model: GNN model to train
        train_data: Training dataset
        val_data: Validation dataset
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cpu' or 'cuda')
        save_dir: Directory to save model checkpoints
        num_neg_per_pos: Number of negative samples per positive edge
        pos_weight: Weight for positive edge loss
        
    Returns:
        Dictionary of training history
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse': [],
        'val_mae': [],
        'val_rmse': [],
        'val_correlation': [],
        'val_r2': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 30
    patience_counter = 0
    
    print(f"\nTraining Configuration:")
    print(f"  Device: {device}")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Learning rate: {lr}")
    print(f"  Max epochs: {num_epochs}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Note: Pure regression (no negative sampling for fully connected graphs)")
    print("\n" + "="*100)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for data in train_data:
            data = data.to(device)
            
            optimizer.zero_grad()
            loss = compute_edge_loss(model, data, num_neg_per_pos=num_neg_per_pos, pos_weight=pos_weight)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_data)
        
        # Validation
        val_metrics = evaluate_model(model, val_data, device, num_neg_per_pos=num_neg_per_pos)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_mse'].append(val_metrics['mse'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_correlation'].append(val_metrics['correlation'])
        history['val_r2'].append(val_metrics['r2'])
        history['learning_rate'].append(current_lr)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {avg_train_loss:.4f}/{val_metrics['loss']:.4f} | "
                  f"MSE: {val_metrics['mse']:.4f} | "
                  f"MAE: {val_metrics['mae']:.4f} | "
                  f"RMSE: {val_metrics['rmse']:.4f} | "
                  f"R²: {val_metrics['r2']:.3f} | "
                  f"Corr: {val_metrics['correlation']:.3f}")
        
        # Detailed logging every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"  → Detailed metrics:")
            print(f"     Weight MAE (low/mid/high): {val_metrics['low_weight_mae']:.4f}/{val_metrics['mid_weight_mae']:.4f}/{val_metrics['high_weight_mae']:.4f}")
            print(f"     Weight counts (low/mid/high): {val_metrics['low_weight_count']}/{val_metrics['mid_weight_count']}/{val_metrics['high_weight_count']}")
            print(f"     Pred stats: mean={val_metrics['pred_mean']:.4f}, std={val_metrics['pred_std']:.4f}")
            print(f"     Label stats: mean={val_metrics['label_mean']:.4f}, std={val_metrics['label_std']:.4f}")
            print(f"     Learning rate: {current_lr:.6f}")
        
        # Early stopping and checkpointing
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'config': {
                    'hidden_channels': model.hidden_channels,
                    'num_layers': model.num_layers,
                    'dropout': model.dropout,
                    'in_channels': model.in_channels
                }
            }, os.path.join(save_dir, 'best_model.pt'))
            
            print(f"  → New best model saved! (Val loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n{'='*100}")
            print(f"Early stopping at epoch {epoch+1}")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
            print(f"{'='*100}")
            break
    
    print("="*100)
    print(f"Training completed!")
    print(f"Best model saved to {save_dir}/best_model.pt")
    print(f"Best epoch: {best_epoch+1}, Best val loss: {best_val_loss:.4f}")
    
    return history


def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MSE and MAE
    axes[0, 1].plot(epochs, history['val_mse'], label='MSE', linewidth=2, color='green')
    axes[0, 1].plot(epochs, history['val_mae'], label='MAE', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Error', fontsize=11)
    axes[0, 1].set_title('Validation MSE and MAE', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSE
    axes[0, 2].plot(epochs, history['val_rmse'], label='RMSE', linewidth=2, color='purple')
    axes[0, 2].set_xlabel('Epoch', fontsize=11)
    axes[0, 2].set_ylabel('RMSE', fontsize=11)
    axes[0, 2].set_title('Validation RMSE', fontsize=12, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # R²
    axes[1, 0].plot(epochs, history['val_r2'], label='R²', linewidth=2, color='brown')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('R²', fontsize=11)
    axes[1, 0].set_title('Validation R² Score', fontsize=12, fontweight='bold')
    axes[1, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation
    axes[1, 1].plot(epochs, history['val_correlation'], label='Correlation', linewidth=2, color='red')
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Correlation', fontsize=11)
    axes[1, 1].set_title('Prediction-Label Correlation', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylim([-1.05, 1.05])
    axes[1, 1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 2].plot(epochs, history['learning_rate'], linewidth=2, color='teal')
    axes[1, 2].set_xlabel('Epoch', fontsize=11)
    axes[1, 2].set_ylabel('Learning Rate', fontsize=11)
    axes[1, 2].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def main():
    """Main training pipeline."""
    # Configuration
    config = {
        'log_dir': 'logs',
        'max_steps': None,  # Use all timesteps
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'num_epochs': 150,
        'lr': 0.001,
        'hidden_channels': 32,
        'num_layers': 3,  # EdgeConv uses more layers than GAT
        'dropout': 0.2,
        'num_neg_per_pos': 2,  # DEPRECATED but kept for compatibility
        'pos_weight': 3.0,      # DEPRECATED but kept for compatibility
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'models'
    }
    
    print("="*100)
    print("GNN Training for Interference Graph Edge Weight Prediction")
    print("="*100)
    
    # Load data
    print("\n[1/4] Loading simulation data...")
    loader = SimulationDataLoader(config['log_dir'])
    loader.load_logs()
    
    # Create dataset
    print("\n[2/4] Creating graph dataset...")
    dataset = loader.create_dataset(max_steps=config['max_steps'])
    train_data, val_data, test_data = loader.train_val_test_split(
        dataset, 
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio']
    )
    
    # Save normalization stats
    loader.save_normalization_stats(os.path.join(config['save_dir'], 'norm_stats.pt'))
    
    # Initialize model
    print("\n[3/4] Initializing model...")
    model = InterferenceGNN(
        in_channels=11,  # 11 node features (3 channel energies: ch1, ch6, ch11 + 8 other features)
        hidden_channels=config['hidden_channels'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    print(f"Model architecture:")
    print(f"  Type: EdgeConv (Edge-Aware Message Passing)")
    print(f"  Input features: 11 (3 channel energies + throughput, clients, duty, roam_in, roam_out, channel, bw, tx_power)")
    print(f"  Hidden channels: {config['hidden_channels']}")
    print(f"  EdgeConv layers: {config['num_layers']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n[4/4] Training model...")
    history = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        num_epochs=config['num_epochs'],
        lr=config['lr'],
        device=config['device'],
        save_dir=config['save_dir'],
        num_neg_per_pos=config['num_neg_per_pos'],
        pos_weight=config['pos_weight']
    )
    
    # Plot results
    print("\nGenerating training plots...")
    plot_training_history(history, save_path=os.path.join(config['save_dir'], 'training_curves.png'))
    
    # Final test evaluation
    print("\n" + "="*100)
    print("Evaluating on test set...")
    print("="*100)
    checkpoint = torch.load(os.path.join(config['save_dir'], 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate_model(model, test_data, config['device'], num_neg_per_pos=config['num_neg_per_pos'])
    
    print(f"\nTest Set Performance:")
    print(f"  Overall Metrics:")
    print(f"    Loss: {test_metrics['loss']:.4f}")
    print(f"    MSE:  {test_metrics['mse']:.4f}")
    print(f"    MAE:  {test_metrics['mae']:.4f}")
    print(f"    RMSE: {test_metrics['rmse']:.4f}")
    print(f"\n  Regression Quality:")
    print(f"    R² Score: {test_metrics['r2']:.3f}")
    print(f"    Correlation: {test_metrics['correlation']:.3f}")
    print(f"\n  Prediction Statistics:")
    print(f"    Pred Mean: {test_metrics['pred_mean']:.4f}")
    print(f"    Pred Std:  {test_metrics['pred_std']:.4f}")
    print(f"    Label Mean: {test_metrics['label_mean']:.4f}")
    print(f"    Label Std:  {test_metrics['label_std']:.4f}")
    print(f"\n  Weight Range Analysis:")
    print(f"    Low weight MAE (<0.2):   {test_metrics['low_weight_mae']:.4f} ({test_metrics['low_weight_count']} edges)")
    print(f"    Mid weight MAE (0.2-0.5): {test_metrics['mid_weight_mae']:.4f} ({test_metrics['mid_weight_count']} edges)")
    print(f"    High weight MAE (>0.5):  {test_metrics['high_weight_mae']:.4f} ({test_metrics['high_weight_count']} edges)")
    
    print("\n" + "="*100)
    print("Training complete!")
    print(f"Model saved to {config['save_dir']}/best_model.pt")
    print(f"Best epoch: {checkpoint['epoch'] + 1}")
    print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
    print("="*100)


if __name__ == "__main__":
    main()