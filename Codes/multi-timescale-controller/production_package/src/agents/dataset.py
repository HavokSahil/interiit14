"""
RRM Dataset class for loading and batching data for training.
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional
from pathlib import Path


class RRMDataset(Dataset):
    """
    PyTorch Dataset for RRM offline RL training.
    
    Loads data from HDF5 file and provides batched access.
    """
    
    def __init__(
        self,
        filepath: str,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        normalize: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize the dataset.
        
        Args:
            filepath: Path to HDF5 dataset file
            split: One of 'train', 'val', 'test'
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            normalize: Whether to normalize states
            device: Device to load tensors to
        """
        self.filepath = Path(filepath)
        self.split = split
        self.normalize = normalize
        self.device = device
        
        # Load data from file
        with h5py.File(self.filepath, 'r') as f:
            states = f['states'][:]
            actions = f['actions'][:]
            rewards = f['rewards'][:]
            next_states = f['next_states'][:]
            dones = f['dones'][:]
            costs = f['costs'][:]
            
            self.num_features = f.attrs['num_features']
            self.num_actions = f.attrs['num_actions']
        
        # Split data
        n_samples = len(states)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        if split == 'train':
            idx_start, idx_end = 0, n_train
        elif split == 'val':
            idx_start, idx_end = n_train, n_train + n_val
        else:  # test
            idx_start, idx_end = n_train + n_val, n_samples
        
        self.states = states[idx_start:idx_end]
        self.actions = actions[idx_start:idx_end]
        self.rewards = rewards[idx_start:idx_end]
        self.next_states = next_states[idx_start:idx_end]
        self.dones = dones[idx_start:idx_end]
        self.costs = costs[idx_start:idx_end]
        
        # Compute normalization statistics from training data
        if normalize:
            if split == 'train':
                self.state_mean = self.states.mean(axis=0)
                self.state_std = self.states.std(axis=0) + 1e-8
            else:
                # Load stats from training set
                train_dataset = RRMDataset(
                    filepath, split='train', train_ratio=train_ratio,
                    val_ratio=val_ratio, normalize=False
                )
                self.state_mean = train_dataset.states.mean(axis=0)
                self.state_std = train_dataset.states.std(axis=0) + 1e-8
            
            # Normalize states
            self.states = (self.states - self.state_mean) / self.state_std
            self.next_states = (self.next_states - self.state_mean) / self.state_std
        else:
            self.state_mean = np.zeros(self.num_features)
            self.state_std = np.ones(self.num_features)
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'state': torch.FloatTensor(self.states[idx]),
            'action': torch.LongTensor([self.actions[idx]]),
            'reward': torch.FloatTensor([self.rewards[idx]]),
            'next_state': torch.FloatTensor(self.next_states[idx]),
            'done': torch.FloatTensor([float(self.dones[idx])]),
            'cost': torch.FloatTensor([self.costs[idx]])
        }
    
    def get_all_data(self) -> Dict[str, torch.Tensor]:
        """Return all data as tensors."""
        return {
            'states': torch.FloatTensor(self.states).to(self.device),
            'actions': torch.LongTensor(self.actions).to(self.device),
            'rewards': torch.FloatTensor(self.rewards).to(self.device),
            'next_states': torch.FloatTensor(self.next_states).to(self.device),
            'dones': torch.FloatTensor(self.dones.astype(np.float32)).to(self.device),
            'costs': torch.FloatTensor(self.costs).to(self.device)
        }
    
    def get_normalization_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return normalization parameters."""
        return self.state_mean, self.state_std


class ReplayBuffer:
    """
    Replay buffer for storing and sampling transitions.
    Used for DQN and can be populated from dataset.
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        device: str = 'cpu'
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.device = device
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.costs = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        cost: float = 0.0
    ):
        """Add a transition to the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.costs[self.ptr] = cost
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def populate_from_dataset(self, dataset: RRMDataset):
        """Populate buffer from a dataset."""
        data = dataset.get_all_data()
        n = min(len(dataset), self.capacity)
        
        self.states[:n] = data['states'][:n].cpu().numpy()
        self.actions[:n] = data['actions'][:n].cpu().numpy()
        self.rewards[:n] = data['rewards'][:n].cpu().numpy()
        self.next_states[:n] = data['next_states'][:n].cpu().numpy()
        self.dones[:n] = data['dones'][:n].cpu().numpy()
        self.costs[:n] = data['costs'][:n].cpu().numpy()
        
        self.size = n
        self.ptr = n % self.capacity
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.LongTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).to(self.device),
            'costs': torch.FloatTensor(self.costs[indices]).to(self.device)
        }
    
    def __len__(self) -> int:
        return self.size


def create_dataloaders(
    filepath: str,
    batch_size: int = 256,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 0,
    normalize: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = RRMDataset(
        filepath, split='train', train_ratio=train_ratio,
        val_ratio=val_ratio, normalize=normalize
    )
    val_dataset = RRMDataset(
        filepath, split='val', train_ratio=train_ratio,
        val_ratio=val_ratio, normalize=normalize
    )
    test_dataset = RRMDataset(
        filepath, split='test', train_ratio=train_ratio,
        val_ratio=val_ratio, normalize=normalize
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

