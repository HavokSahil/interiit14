"""
GNN model for interference graph edge weight prediction.
Uses Edge-Aware Message Passing (EdgeConv) to learn representations
that are naturally suited for predicting edge weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_mean_pool
from torch_geometric.data import Data
from typing import Tuple


class InterferenceGNN(nn.Module):
    """
    Graph Neural Network for predicting interference edge weights between APs.
    
    Architecture:
        - Input: Node features (6D: energy, throughput, clients, duty_cycle, roam_in, roam_out)
        - EdgeConv layers for edge-aware message passing
        - Edge predictor for weight regression
        
    EdgeConv is ideal for this task because it:
    - Learns from pairs of nodes (edge-centric)
    - Naturally captures relationships for edge prediction
    - Works well on fully connected graphs
    """
    
    def __init__(self, 
                 in_channels: int = 6,
                 hidden_channels: int = 32,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 aggr: str = 'max'):
        """
        Initialize edge-aware GNN model.
        
        Args:
            in_channels: Number of input node features
            hidden_channels: Hidden dimension size
            num_layers: Number of EdgeConv layers
            dropout: Dropout rate
            aggr: Aggregation method ('max', 'mean', 'add')
        """
        super(InterferenceGNN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        # EdgeConv layers
        # EdgeConv applies MLP to [x_i, x_j - x_i] for each edge (i,j)
        self.convs = nn.ModuleList()
        
        # First layer
        edge_nn = nn.Sequential(
            nn.Linear(2 * in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(EdgeConv(edge_nn, aggr=aggr))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            edge_nn = nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(EdgeConv(edge_nn, aggr=aggr))
        
        # Final embedding dimension
        self.embedding_dim = hidden_channels
        
        # Edge predictor MLP
        self.edge_predictor = EdgePredictor(
            in_channels=self.embedding_dim * 2,  # Concatenate src and dst embeddings
            hidden_channels=64,
            dropout=dropout
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass for node embedding using edge-aware message passing.
        
        Args:
            data: PyG Data object with node features and edge_index
            
        Returns:
            Node embeddings (num_nodes, embedding_dim)
        """
        x, edge_index = data.x, data.edge_index
        
        # Apply EdgeConv layers with activation and dropout
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.elu(x) 
            if i < len(self.convs) - 1:  # Dropout for all but last layer
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def predict_edges(self, node_embeddings: torch.Tensor, 
                     edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict edge probabilities given node embeddings.
        
        Args:
            node_embeddings: (num_nodes, embedding_dim)
            edge_index: (2, num_edges) candidate edges to score
            
        Returns:
            Edge probabilities (num_edges, 1)
        """
        # Get source and destination embeddings
        src_embeddings = node_embeddings[edge_index[0]]
        dst_embeddings = node_embeddings[edge_index[1]]
        
        # Concatenate
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
        
        # Predict
        edge_probs = self.edge_predictor(edge_embeddings)
        
        return edge_probs
    
    def predict_all_edges(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict edges for all possible AP pairs.
        
        Args:
            data: PyG Data object
            
        Returns:
            edge_index: (2, num_possible_edges) all possible edges
            edge_probs: (num_possible_edges, 1) predicted probabilities
        """
        # Get node embeddings
        node_embeddings = self.forward(data)
        
        # Generate all possible edges (excluding self-loops)
        num_nodes = data.num_nodes
        all_edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    all_edges.append([i, j])
        
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().to(data.x.device)
        
        # Predict
        edge_probs = self.predict_edges(node_embeddings, edge_index)
        
        return edge_index, edge_probs


class EdgePredictor(nn.Module):
    """MLP for edge weight regression (0-1 range)."""
    
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.2):
        super(EdgePredictor, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Edge embeddings (num_edges, in_channels)
            
        Returns:
            Edge weights (num_edges, 1) in range [0, 1]
        """
        weights = self.mlp(x)
        # Use sigmoid for smooth gradient flow instead of clamping
        return torch.sigmoid(weights)


def get_positive_edges(data: Data) -> torch.Tensor:
    """
    Get positive edges (ground truth) from data.
    
    Args:
        data: PyG Data object
        
    Returns:
        Positive edge index (2, num_pos_edges)
    """
    if not hasattr(data, 'y_edge_index') or data.y_edge_index is None:
        raise ValueError("Data object must have 'y_edge_index' attribute with ground truth edges")
    return data.y_edge_index


def get_negative_edges(data: Data, num_neg_per_pos: int = 2) -> torch.Tensor:
    """
    Sample negative edges (non-existent edges).
    
    Args:
        data: PyG Data object
        num_neg_per_pos: Number of negative samples per positive edge
        
    Returns:
        Negative edge index (2, num_neg_edges)
    """
    num_nodes = data.num_nodes
    num_pos = data.y_edge_index.shape[1]
    num_neg = num_pos * num_neg_per_pos
    
    # Convert positive edges to set for fast lookup
    pos_edges = set(map(tuple, data.y_edge_index.t().tolist()))
    
    # Sample negative edges
    neg_edges = []
    attempts = 0
    max_attempts = num_neg * 10
    
    while len(neg_edges) < num_neg and attempts < max_attempts:
        i = torch.randint(0, num_nodes, (1,)).item()
        j = torch.randint(0, num_nodes, (1,)).item()
        
        if i != j and (i, j) not in pos_edges:
            neg_edges.append([i, j])
        
        attempts += 1
    
    if len(neg_edges) == 0:
        # Fallback: return empty tensor
        print(f"Warning: Failed to sample any negative edges (tried {attempts} times)")
        return torch.tensor([[], []], dtype=torch.long)
    
    if len(neg_edges) < num_neg:
        print(f"Warning: Only sampled {len(neg_edges)}/{num_neg} negative edges")
    
    return torch.tensor(neg_edges, dtype=torch.long).t()


def compute_edge_loss(model: InterferenceGNN, data: Data, 
                     num_neg_per_pos: int = 2,
                     pos_weight: float = 3.0) -> torch.Tensor:
    """
    Compute MSE loss for edge weight prediction.
    
    Note: For fully connected interference graphs (all AP pairs have edges),
    this is pure regression without negative sampling.
    
    Args:
        model: GNN model
        data: PyG Data object with y_edge_attr containing ground truth weights
        num_neg_per_pos: DEPRECATED - kept for backward compatibility, ignored
        pos_weight: DEPRECATED - kept for backward compatibility, ignored
        
    Returns:
        Loss value (MSE between predicted and ground truth edge weights)
    """
    # Get node embeddings
    node_embeddings = model.forward(data)
    
    # Get all edges with ground truth weights
    edge_index = get_positive_edges(data)
    edge_preds = model.predict_edges(node_embeddings, edge_index)
    
    # Get ground truth weights - standardize on y_edge_attr
    if not hasattr(data, 'y_edge_attr') or data.y_edge_attr is None or len(data.y_edge_attr) == 0:
        raise ValueError("Data object must have 'y_edge_attr' attribute with ground truth edge weights")
    
    edge_labels = data.y_edge_attr.view(-1, 1)
    
    # Pure MSE loss on edge weights (no negative sampling for fully connected graphs)
    loss = F.mse_loss(edge_preds, edge_labels)
    
    return loss


def train_model(model, train_data, val_data, epochs=100, lr=0.001, device='cpu'):
    """
    Train the GNN model.
    
    Args:
        model: InterferenceGNN model
        train_data: List of training Data objects
        val_data: List of validation Data objects
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cpu' or 'cuda')
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for data in train_data:
            data = data.to(device)
            optimizer.zero_grad()
            loss = compute_edge_loss(model, data, num_neg_per_pos=2, pos_weight=3.0)
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_data)
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                all_pos_preds = []
                all_pos_labels = []
                
                for data in val_data:
                    data = data.to(device)
                    val_loss += compute_edge_loss(model, data, num_neg_per_pos=2, pos_weight=3.0).item()
                    
                    # Get predictions for positive edges
                    embeddings = model(data)
                    pos_preds = model.predict_edges(embeddings, data.y_edge_index)
                    all_pos_preds.append(pos_preds)
                    
                    if hasattr(data, 'y_edge_attr') and data.y_edge_attr is not None:
                        all_pos_labels.append(data.y_edge_attr)
                
                val_loss /= len(val_data)
                
                # Aggregate predictions
                if all_pos_preds:
                    all_pos_preds = torch.cat(all_pos_preds, dim=0)
                    print(f"Epoch {epoch}: Train Loss={avg_loss:.4f}, Val Loss={val_loss:.4f}")
                    print(f"  Pred stats - min: {all_pos_preds.min():.4f}, max: {all_pos_preds.max():.4f}, mean: {all_pos_preds.mean():.4f}")
                    
                    if all_pos_labels:
                        all_pos_labels = torch.cat(all_pos_labels, dim=0)
                        print(f"  True stats - min: {all_pos_labels.min():.4f}, max: {all_pos_labels.max():.4f}, mean: {all_pos_labels.mean():.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), 'best_gnn_model.pt')
                    print(f"  → New best model saved!")
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    return model


if __name__ == "__main__":
    # Test model
    from gnn_data import SimulationDataLoader
    
    print("Loading data...")
    loader = SimulationDataLoader("logs")
    loader.load_logs()
    dataset = loader.create_dataset(max_steps=100)
    
    # Split dataset
    train_data, val_data, test_data = loader.train_val_test_split(dataset)
    
    print("\nInitializing model...")
    model = InterferenceGNN(
        in_channels=9,  # 9 node features (added channel, bandwidth, tx_power)
        hidden_channels=32,
        num_layers=3,  # EdgeConv typically uses more layers
        dropout=0.2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    data = dataset[0]
    embeddings = model(data)
    print(f"Node embeddings shape: {embeddings.shape}")
    
    # Test edge prediction
    edge_index, edge_probs = model.predict_all_edges(data)
    print(f"Predicted {edge_probs.shape[0]} edges")
    print(f"Mean probability: {edge_probs.mean():.4f}")
    
    # Test loss computation
    loss = compute_edge_loss(model, data)
    print(f"Initial loss: {loss.item():.4f}")
    
    # Train model
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_model(model, train_data, val_data, epochs=100, lr=0.001, device=device)