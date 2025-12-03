"""
Data loading and preprocessing for GNN-based interference graph prediction.
Reads simulation CSV logs and creates PyTorch Geometric graph snapshots.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import torch
from torch_geometric.data import Data
from collections import defaultdict


class SimulationDataLoader:
    """Load and preprocess simulation data from CSV logs."""
    
    def __init__(self, log_dir: str):
        """
        Initialize data loader.
        
        Args:
            log_dir: Directory containing CSV log files
        """
        self.log_dir = log_dir
        self.ap_df = None
        self.client_df = None
        self.roam_df = None
        self.graph_df = None
        
        # Feature normalization statistics (computed during dataset creation)
        self.feature_mean = None
        self.feature_std = None
        
    def load_logs(self, prefix: str = "sim") -> None:
        """
        Load all CSV log files.
        
        Args:
            prefix: Filename prefix for logs (default: "sim")
        """
        # Find latest log files with the given prefix
        files = os.listdir(self.log_dir)
        
        # Get most recent logs
        ap_files = sorted([f for f in files if f.startswith(f"{prefix}_ap_")])
        client_files = sorted([f for f in files if f.startswith(f"{prefix}_client_")])
        roam_files = sorted([f for f in files if f.startswith(f"{prefix}_roam_")])
        graph_files = sorted([f for f in files if f.startswith(f"{prefix}_graph_")])
        
        if not (ap_files and client_files and roam_files and graph_files):
            raise FileNotFoundError(f"Could not find all required log files in {self.log_dir}")
        
        # Load most recent logs
        self.ap_df = pd.read_csv(os.path.join(self.log_dir, ap_files[-1]))
        self.client_df = pd.read_csv(os.path.join(self.log_dir, client_files[-1]))
        self.roam_df = pd.read_csv(os.path.join(self.log_dir, roam_files[-1]))
        self.graph_df = pd.read_csv(os.path.join(self.log_dir, graph_files[-1]))
        
        print(f"Loaded logs from {self.log_dir}:")
        print(f"  AP states: {len(self.ap_df)} records")
        print(f"  Client states: {len(self.client_df)} records")
        print(f"  Roaming events: {len(self.roam_df)} records")
        print(f"  Graph edges: {len(self.graph_df)} records")
        
    def get_timesteps(self) -> List[int]:
        """Get list of all timesteps in the data."""
        return sorted(self.ap_df['step'].unique().tolist())
    
    def get_ap_ids(self) -> List[int]:
        """Get list of all AP IDs."""
        return sorted(self.ap_df['ap_id'].unique().tolist())
    
    def extract_ap_features(self, ap: dict) -> List[float]:
        """
        Extract features for a single AP.
        
        Features (11D):
            1. inc_energy_ch1_dbm: Incoming energy on channel 1 (raw values)
            2. inc_energy_ch6_dbm: Incoming energy on channel 6 (raw values)
            3. inc_energy_ch11_dbm: Incoming energy on channel 11 (raw values)
            4. total_allocated_throughput: Sum of allocated throughput to connected clients
            5. num_connected_clients: Number of connected clients
            6. duty_cycle: AP airtime utilization
            7. roam_in: Number of clients roamed in
            8. roam_out: Number of clients roamed out
            9. channel: Operating channel number
            10. bandwidth: Channel bandwidth (MHz)
            11. tx_power: Transmission power (dBm)
        
        Args:
            ap: AP state dictionary
            
        Returns:
            List of 11 feature values
        """
        # Raw inc_energy for each channel (will be z-score normalized later)
        inc_energy_ch1 = float(ap['inc_energy_ch1_dbm']) if ap['inc_energy_ch1_dbm'] != 'N/A' else -100.0
        inc_energy_ch6 = float(ap['inc_energy_ch6_dbm']) if ap['inc_energy_ch6_dbm'] != 'N/A' else -100.0
        inc_energy_ch11 = float(ap['inc_energy_ch11_dbm']) if ap['inc_energy_ch11_dbm'] != 'N/A' else -100.0
        
        # Client and traffic features
        total_throughput = float(ap['allocated_throughput'])
        num_clients = int(ap['num_clients'])
        
        # Airtime duty cycle
        duty_cycle = float(ap['duty_cycle']) if 'duty_cycle' in ap else 0.0
        
        # Roaming events
        roam_in = int(ap.get('roam_in', 0))
        roam_out = int(ap.get('roam_out', 0))
        
        # RF parameters
        channel = int(ap.get('channel', 0))
        bandwidth = int(ap.get('bandwidth', 20))  # Default 20 MHz
        tx_power = float(ap.get('tx_power', 20.0))
        
        return [
            inc_energy_ch1,
            inc_energy_ch6,
            inc_energy_ch11,
            total_throughput,
            num_clients,
            duty_cycle,
            roam_in,
            roam_out,
            channel,
            bandwidth,
            tx_power
        ]
    
    def compute_roaming_features(self, step: int, ap_ids: List[int]) -> np.ndarray:
        """
        Compute roaming-based features for each AP at a given timestep.
        
        Args:
            step: Timestep to compute features for
            ap_ids: List of AP IDs
            
        Returns:
            roaming_features: (num_aps, 2) array [roam_in_count, roam_out_count]
        """
        step_roams = self.roam_df[self.roam_df['step'] == step]
        
        roam_in = defaultdict(int)  # Clients roaming TO this AP
        roam_out = defaultdict(int)  # Clients roaming FROM this AP
        
        for _, roam in step_roams.iterrows():
            from_ap = roam['from_ap']
            to_ap = roam['to_ap']
            
            # Skip initial associations (from_ap is NaN)
            if pd.notna(from_ap):
                roam_out[int(from_ap)] += 1
                roam_in[int(to_ap)] += 1
        
        # Build feature array
        features = np.zeros((len(ap_ids), 2), dtype=np.float32)
        for i, ap_id in enumerate(ap_ids):
            features[i, 0] = roam_in[ap_id]
            features[i, 1] = roam_out[ap_id]
        
        return features
    
    def normalize_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalize features using z-score normalization.
        
        Args:
            features: (num_samples, num_features) array
            fit: If True, compute and store normalization statistics
            
        Returns:
            Normalized features
        """
        if fit:
            # Compute statistics on the full feature set
            self.feature_mean = features.mean(axis=0)
            self.feature_std = features.std(axis=0) + 1e-8  # Add small epsilon to avoid division by zero
            print(f"\nFeature normalization statistics:")
            print(f"  Mean: {self.feature_mean}")
            print(f"  Std:  {self.feature_std}")
        
        if self.feature_mean is None or self.feature_std is None:
            raise ValueError("Must call normalize_features with fit=True first")
        
        # Apply z-score normalization
        normalized = (features - self.feature_mean) / self.feature_std
        
        # Validate normalized features
        if np.isnan(normalized).any():
            raise ValueError("Normalized features contain NaN values")
        if np.isinf(normalized).any():
            raise ValueError("Normalized features contain infinite values")
        
        return normalized
    
    def save_normalization_stats(self, path: str) -> None:
        """Save normalization statistics to file."""
        if self.feature_mean is None or self.feature_std is None:
            raise ValueError("No normalization stats to save")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'mean': self.feature_mean,
            'std': self.feature_std
        }, path)
        print(f"Saved normalization stats to {path}")
        
    def load_normalization_stats(self, path: str) -> None:
        """Load normalization statistics from file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Normalization stats not found at {path}")
            
        stats = torch.load(path, weights_only=False)
        self.feature_mean = stats['mean']
        self.feature_std = stats['std']
        print(f"Loaded normalization stats from {path}")
    
    def extract_edges(self, step: int, ap_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract interference graph edges for a given timestep.
        
        Args:
            step: Timestep to extract edges for
            ap_ids: List of AP IDs (for indexing)
            
        Returns:
            edge_index: (2, num_edges) array of edge indices
            edge_weights: (num_edges,) array of edge weights
        """
        step_edges = self.graph_df[self.graph_df['step'] == step]
        
        # Create AP ID to index mapping
        ap_to_idx = {ap_id: idx for idx, ap_id in enumerate(ap_ids)}
        
        edge_list = []
        weights = []
        
        for _, edge in step_edges.iterrows():
            src = int(edge['source_ap'])
            dst = int(edge['dest_ap'])
            weight = float(edge['weight'])
            
            if src in ap_to_idx and dst in ap_to_idx:
                edge_list.append([ap_to_idx[src], ap_to_idx[dst]])
                weights.append(weight)
        
        if len(edge_list) == 0:
            # No edges - return empty arrays
            return np.array([[], []], dtype=np.int64), np.array([], dtype=np.float32)
        
        edge_index = np.array(edge_list, dtype=np.int64).T
        edge_weights = np.array(weights, dtype=np.float32)
        
        return edge_index, edge_weights
    
    def build_graph_snapshot(self, step: int, normalize: bool = True) -> Data:
        """
        Build a PyTorch Geometric Data object for a single timestep.
        
        Args:
            step: Timestep to build graph for
            normalize: Whether to normalize features
            
        Returns:
            PyG Data object with node features, edge_index, and edge weights
        """
        # Get AP states for this step
        step_aps = self.ap_df[self.ap_df['step'] == step].sort_values('ap_id')
        ap_ids = step_aps['ap_id'].tolist()
       
        # Extract features for each AP using the updated method
        ap_features = []
        for _, ap_row in step_aps.iterrows():
            # Convert row to dict for extract_ap_features
            ap_dict = ap_row.to_dict()
            features = self.extract_ap_features(ap_dict)
            ap_features.append(features)
        
        ap_features = np.array(ap_features, dtype=np.float32)
        
        # Get roaming features
        # roaming_features = self.compute_roaming_features(step, ap_ids) # Removed as roam_in/out are now in extract_ap_features
        
        # Note: roaming features (roam_in, roam_out) are now included in extract_ap_features
        # So we don't need to append them separately anymore
        # The extract_ap_features already returns 9 features including roam_in and roam_out
        
        # Get edges and weights
        edge_index, edge_weights = self.extract_edges(step, ap_ids)
        
        # Normalize features if requested
        if normalize and self.feature_mean is not None: # Corrected from self.norm_mean
            ap_features = (ap_features - self.feature_mean) / self.feature_std # Corrected from self.norm_mean/std
        
        # Convert to tensors
        x = torch.tensor(ap_features, dtype=torch.float)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        
        num_nodes = len(ap_ids)
        src_list, dst_list = [], []

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    src_list.append(i)
                    dst_list.append(j)

        edge_attr = torch.tensor(edge_weights, dtype=torch.float)

        if len(src_list) > 0:
            fc_edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            fc_edge_index = torch.tensor([[], []], dtype=torch.long)

        # Create Data object
        data = Data(
            x=x,
            edge_index=fc_edge_index,       # Fully connected for GAT message passing
            y_edge_index=edge_index_tensor, # Ground truth edges for loss calculation
            y_edge_attr=edge_attr,          # Ground truth weights
            num_nodes=num_nodes
        )
        
        return data
    
    def create_dataset(self, max_steps: int = None) -> List[Data]:
        """
        Create dataset of graph snapshots from loaded logs.
        
        Args:
            max_steps: Maximum number of timesteps to include (None = all)
            
        Returns:
            List of PyG Data objects
        """
        steps = sorted(self.ap_df['step'].unique())
        if max_steps is not None:
            steps = steps[:max_steps]
        
        print(f"Creating dataset with {len(steps)} graph snapshots")
        
        # First pass: collect all features to compute normalization statistics
        all_features = []
        for step in steps:
            step_aps = self.ap_df[self.ap_df['step'] == step].sort_values('ap_id')
            for _, ap_row in step_aps.iterrows():
                ap_dict = ap_row.to_dict()
                features = self.extract_ap_features(ap_dict)
                all_features.append(features)
        
        all_features = np.array(all_features, dtype=np.float32)
        
        # Compute normalization statistics
        self.feature_mean = np.mean(all_features, axis=0)
        self.feature_std = np.std(all_features, axis=0)
        
        # Avoid division by zero
        self.feature_std = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)
        
        print(f"\nFeature normalization statistics:")
        print(f"  Mean: {self.feature_mean}")
        print(f"  Std:  {self.feature_std}")
        
        # Second pass: build normalized graphs
        dataset = []
        for step in steps:
            data = self.build_graph_snapshot(step, normalize=True)
            dataset.append(data)
        
        # Print dataset statistics
        num_nodes_list = [data.num_nodes for data in dataset]
        num_edges_list = [data.y_edge_index.shape[1] for data in dataset]
        
        # Edge weight statistics
        all_weights = []
        for data in dataset:
            if hasattr(data, 'y_edge_attr') and data.y_edge_attr is not None:
                all_weights.extend(data.y_edge_attr.numpy().flatten())
        
        print(f"\nCreated dataset with {len(dataset)} graph snapshots")
        print(f"  Node features: {dataset[0].x.shape[1]}")
        print(f"  Avg nodes per graph: {np.mean([d.num_nodes for d in dataset]):.1f}")
        print(f"  Avg edges per graph: {np.mean(num_edges_list):.1f}")
        
        if all_weights:
            all_weights = np.array(all_weights)
            print(f"  Edge weight stats - min: {all_weights.min():.4f}, max: {all_weights.max():.4f}, mean: {all_weights.mean():.4f}")
        
        return dataset
    
    def train_val_test_split(self, dataset: List[Data], 
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15) -> Tuple[List[Data], List[Data], List[Data]]:
        """
        Split dataset into train/val/test sets temporally.
        
        Args:
            dataset: List of Data objects
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            
        Returns:
            train_data, val_data, test_data
        """
        n = len(dataset)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = dataset[:train_end]
        val_data = dataset[train_end:val_end]
        test_data = dataset[val_end:]
        
        print(f"\nDataset split:")
        print(f"  Train: {len(train_data)} graphs")
        print(f"  Val: {len(val_data)} graphs")
        print(f"  Test: {len(test_data)} graphs")
        
        return train_data, val_data, test_data


if __name__ == "__main__":
    # Example usage
    loader = SimulationDataLoader("logs")
    loader.load_logs()
    
    dataset = loader.create_dataset(max_steps=100)
    train, val, test = loader.train_val_test_split(dataset)
    
    print(f"\nSample graph (after normalization):")
    print(f"  Nodes: {train[0].num_nodes}")
    print(f"  Edges: {train[0].y_edge_index.shape[1]}")
    print(f"  Node features shape: {train[0].x.shape}")
    print(f"  Feature stats - min: {train[0].x.min():.4f}, max: {train[0].x.max():.4f}, mean: {train[0].x.mean():.4f}")