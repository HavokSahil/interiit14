"""
Data Augmentation Utilities for RRM Dataset

Provides domain randomization and trajectory augmentation techniques
to improve dataset quality and model generalization.
"""

import numpy as np
from typing import Dict, List, Tuple
from copy import deepcopy


class DomainRandomizer:
    """
    Domain randomization for Wi-Fi network simulation.
    
    Adds variations to make the simulation more robust:
    - Propagation model variations (indoor/outdoor)
    - Interference pattern variations
    - Measurement noise
    - Client behavior variations
    """
    
    def __init__(self, seed: int = 42):
        """Initialize domain randomizer."""
        self.rng = np.random.default_rng(seed)
    
    def add_measurement_noise(
        self,
        state: np.ndarray,
        noise_levels: Dict[str, float] = None
    ) -> np.ndarray:
        """
        Add realistic measurement noise to state features.
        
        Args:
            state: State vector [15 features]
            noise_levels: Dictionary mapping feature indices to noise std
            
        Returns:
            State with added noise
        """
        if noise_levels is None:
            # Default noise levels (in feature units)
            noise_levels = {
                0: 0.5,   # Client count: ±0.5 clients
                1: 1.0,   # RSSI: ±1 dBm
                2: 0.01,  # Retry rate: ±1%
                3: 0.01,  # PER: ±1%
                4: 0.02,  # Channel utilization: ±2%
                5: 2.0,   # Throughput: ±2 Mbps
                6: 1.0,   # Edge throughput: ±1 Mbps
                7: 1.5,   # Neighbor RSSI: ±1.5 dBm
                10: 0.5,  # Noise floor: ±0.5 dBm
            }
        
        noisy_state = state.copy()
        for idx, noise_std in noise_levels.items():
            if idx < len(state):
                noise = self.rng.normal(0, noise_std)
                noisy_state[idx] += noise
        
        return noisy_state
    
    def vary_propagation_model(
        self,
        rssi: float,
        tx_power: float,
        model_type: str = 'random'
    ) -> float:
        """
        Vary propagation model (indoor vs outdoor).
        
        Args:
            rssi: Current RSSI
            tx_power: Current Tx power
            model_type: 'indoor', 'outdoor', or 'random'
            
        Returns:
            Adjusted RSSI
        """
        if model_type == 'random':
            model_type = self.rng.choice(['indoor', 'outdoor'])
        
        # Path loss exponent varies: indoor (3-4) vs outdoor (2-2.5)
        if model_type == 'indoor':
            # More attenuation (higher path loss)
            path_loss_factor = self.rng.uniform(0.4, 0.6)  # 0.4-0.6 dB per 1 dB Tx
        else:  # outdoor
            # Less attenuation (lower path loss)
            path_loss_factor = self.rng.uniform(0.6, 0.9)  # 0.6-0.9 dB per 1 dB Tx
        
        # Adjust RSSI based on model
        rssi_adjustment = (path_loss_factor - 0.5) * 2  # Center around 0.5
        return rssi + rssi_adjustment
    
    def vary_interference_pattern(
        self,
        neighbor_rssi: float,
        channel_util: float
    ) -> Tuple[float, float]:
        """
        Vary interference patterns.
        
        Args:
            neighbor_rssi: Current neighbor AP RSSI
            channel_util: Current channel utilization
            
        Returns:
            (adjusted_neighbor_rssi, adjusted_channel_util)
        """
        # Vary neighbor interference
        neighbor_noise = self.rng.normal(0, 2.0)  # ±2 dBm
        neighbor_rssi_new = np.clip(neighbor_rssi + neighbor_noise, -90, -50)
        
        # Vary channel utilization (correlated with interference)
        util_noise = self.rng.normal(0, 0.05)  # ±5%
        channel_util_new = np.clip(channel_util + util_noise, 0, 1)
        
        return neighbor_rssi_new, channel_util_new
    
    def augment_state(self, state: np.ndarray) -> np.ndarray:
        """
        Apply all augmentation techniques to a state.
        
        Args:
            state: Original state vector
            
        Returns:
            Augmented state vector
        """
        augmented = state.copy()
        
        # Add measurement noise
        augmented = self.add_measurement_noise(augmented)
        
        # Vary propagation model (affects RSSI)
        if len(augmented) > 1:
            augmented[1] = self.vary_propagation_model(
                augmented[1],
                augmented[9] if len(augmented) > 9 else 15.0
            )
        
        # Vary interference pattern
        if len(augmented) > 7 and len(augmented) > 4:
            neighbor_rssi, channel_util = self.vary_interference_pattern(
                augmented[7],
                augmented[4]
            )
            augmented[7] = neighbor_rssi
            augmented[4] = channel_util
        
        return augmented


class TrajectoryAugmenter:
    """
    Augment trajectories by stitching and mixing.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize trajectory augmenter."""
        self.rng = np.random.default_rng(seed)
    
    def stitch_trajectories(
        self,
        traj1: List[Dict],
        traj2: List[Dict],
        stitch_point: int = None
    ) -> List[Dict]:
        """
        Stitch two partial trajectories together.
        
        Args:
            traj1: First trajectory
            traj2: Second trajectory
            stitch_point: Point to stitch (None = random)
            
        Returns:
            Stitched trajectory
        """
        if stitch_point is None:
            stitch_point = self.rng.integers(1, min(len(traj1), len(traj2)))
        
        stitched = traj1[:stitch_point] + traj2[stitch_point:]
        return stitched
    
    def mix_trajectories(
        self,
        trajectories: List[List[Dict]],
        mix_ratio: float = 0.5
    ) -> List[Dict]:
        """
        Mix multiple trajectories with given ratio.
        
        Args:
            trajectories: List of trajectories
            mix_ratio: Ratio to mix (0.5 = equal mix)
            
        Returns:
            Mixed trajectory
        """
        if not trajectories:
            return []
        
        # Sample from each trajectory based on mix_ratio
        min_len = min(len(t) for t in trajectories)
        mixed = []
        
        for i in range(min_len):
            # Choose which trajectory to sample from
            traj_idx = self.rng.choice(len(trajectories))
            mixed.append(deepcopy(trajectories[traj_idx][i]))
        
        return mixed


class BalancedSampler:
    """
    Balanced sampling for rare states and actions.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize balanced sampler."""
        self.rng = np.random.default_rng(seed)
    
    def identify_edge_cases(
        self,
        states: np.ndarray,
        thresholds: Dict[str, Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Identify edge case states.
        
        Args:
            states: Array of states [N, 15]
            thresholds: Dictionary mapping feature indices to (low, high) thresholds
            
        Returns:
            Boolean array indicating edge cases
        """
        if thresholds is None:
            thresholds = {
                1: (-85, -55),  # Low/high RSSI
                2: (0.15, 0.30),  # High retry rate
                4: (0.7, 1.0),  # High channel utilization
                5: (1, 20),  # Low throughput
            }
        
        edge_cases = np.zeros(len(states), dtype=bool)
        
        for idx, (low, high) in thresholds.items():
            if idx < states.shape[1]:
                # Low edge cases
                low_edge = states[:, idx] < low
                # High edge cases
                high_edge = states[:, idx] > high
                edge_cases |= (low_edge | high_edge)
        
        return edge_cases
    
    def create_balanced_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        batch_size: int,
        edge_case_ratio: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create balanced batch with oversampled edge cases.
        
        Args:
            states: All states
            actions: All actions
            edge_case_ratio: Ratio of edge cases in batch
            
        Returns:
            (batch_states, batch_actions)
        """
        edge_cases = self.identify_edge_cases(states)
        normal_cases = ~edge_cases
        
        n_edge = int(batch_size * edge_case_ratio)
        n_normal = batch_size - n_edge
        
        # Sample edge cases
        edge_indices = np.where(edge_cases)[0]
        if len(edge_indices) > 0:
            edge_selected = self.rng.choice(edge_indices, size=min(n_edge, len(edge_indices)), replace=True)
        else:
            edge_selected = np.array([], dtype=int)
        
        # Sample normal cases
        normal_indices = np.where(normal_cases)[0]
        if len(normal_indices) > 0:
            normal_selected = self.rng.choice(normal_indices, size=n_normal, replace=True)
        else:
            normal_selected = np.array([], dtype=int)
        
        # Combine
        all_selected = np.concatenate([edge_selected, normal_selected])
        self.rng.shuffle(all_selected)
        
        return states[all_selected], actions[all_selected]
    
    def balance_action_distribution(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        target_dist: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance action distribution in dataset.
        
        Args:
            states: All states
            actions: All actions
            target_dist: Target action distribution (None = uniform)
            
        Returns:
            (balanced_states, balanced_actions)
        """
        if target_dist is None:
            target_dist = np.ones(5) / 5  # Uniform
        
        # Count current distribution
        unique, counts = np.unique(actions, return_counts=True)
        current_dist = np.zeros(5)
        current_dist[unique] = counts
        current_dist = current_dist / current_dist.sum()
        
        # Calculate sampling weights
        weights = target_dist / (current_dist + 1e-10)
        weights = weights / weights.sum()
        
        # Sample with weights
        action_weights = weights[actions.astype(int)]
        sample_indices = self.rng.choice(
            len(states),
            size=len(states),
            replace=True,
            p=action_weights / action_weights.sum()
        )
        
        return states[sample_indices], actions[sample_indices]

