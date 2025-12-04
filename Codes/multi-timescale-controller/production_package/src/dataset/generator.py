"""
Synthetic Dataset Generator for RRM Safe RL

Generates 10k realistic Wi-Fi network state transitions with:
- 15 observation features
- 5 discrete actions
- Rewards based on performance metrics
- Safety costs for constraint violations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import h5py
from pathlib import Path

from .physics_models import (
    calculate_sinr, select_mcs, calculate_per, calculate_throughput,
    calculate_edge_throughput, calculate_retry_rate_from_per
)
from .augmentation import DomainRandomizer, TrajectoryAugmenter, BalancedSampler


@dataclass
class FeatureConfig:
    """Configuration for a single observation feature."""
    name: str
    min_val: float
    max_val: float
    distribution: str
    params: Dict


class RRMDatasetGenerator:
    """
    Generates synthetic RRM dataset with realistic Wi-Fi network parameters.
    
    Features (15):
        0: client_count - Number of clients connected to AP
        1: median_rssi - Median RSSI of connected clients (dBm)
        2: p95_retry_rate - 95th percentile retry rate
        3: p95_per - 95th percentile packet error rate
        4: channel_utilization - Channel busy percentage
        5: avg_throughput - Average throughput (Mbps)
        6: edge_p10_throughput - 10th percentile edge client throughput
        7: neighbor_ap_rssi - Interference from neighbor APs (dBm)
        8: obss_pd_threshold - OBSS/PD threshold (dBm)
        9: tx_power - Current transmit power (dBm)
        10: noise_floor - Noise floor (dBm)
        11: channel_width - Channel width (20/40/80 MHz)
        12: airtime_usage - Airtime usage percentage
        13: cca_busy - CCA busy percentage
        14: roaming_rate - Client roaming rate
    
    Actions (9):
        0: Increase Tx Power by +2 dBm
        1: Decrease Tx Power by -2 dBm
        2: Increase OBSS/PD threshold by +4 dBm
        3: Decrease OBSS/PD threshold by -4 dBm
        4: Increase Channel Width (20→40 or 40→80 MHz)
        5: Decrease Channel Width (80→40 or 40→20 MHz)
        6: Increase Channel Number (next DFS-compliant channel)
        7: Decrease Channel Number (previous DFS-compliant channel)
        8: No-op
    """
    
    # Feature indices
    IDX_CLIENT_COUNT = 0
    IDX_MEDIAN_RSSI = 1
    IDX_P95_RETRY = 2
    IDX_P95_PER = 3
    IDX_CHANNEL_UTIL = 4
    IDX_AVG_THROUGHPUT = 5
    IDX_EDGE_THROUGHPUT = 6
    IDX_NEIGHBOR_RSSI = 7
    IDX_OBSS_PD = 8
    IDX_TX_POWER = 9
    IDX_NOISE_FLOOR = 10
    IDX_CHANNEL_WIDTH = 11
    IDX_AIRTIME = 12
    IDX_CCA_BUSY = 13
    IDX_ROAMING_RATE = 14
    
    NUM_FEATURES = 15
    NUM_ACTIONS = 9
    
    # Action definitions
    ACTION_INCREASE_TX = 0
    ACTION_DECREASE_TX = 1
    ACTION_INCREASE_OBSS = 2
    ACTION_DECREASE_OBSS = 3
    ACTION_INCREASE_CHANNEL_WIDTH = 4
    ACTION_DECREASE_CHANNEL_WIDTH = 5
    ACTION_INCREASE_CHANNEL_NUMBER = 6
    ACTION_DECREASE_CHANNEL_NUMBER = 7
    ACTION_NOOP = 8
    
    # Safety Shield limits
    TX_POWER_MIN = 10.0
    TX_POWER_MAX = 20.0
    OBSS_PD_MIN = -82.0
    OBSS_PD_MAX = -68.0
    
    # Hard constraint limits
    TX_POWER_HARD_MIN = 0.0
    TX_POWER_HARD_MAX = 30.0
    OBSS_PD_HARD_MIN = -82.0
    OBSS_PD_HARD_MAX = -62.0
    
    def __init__(self, seed: int = 42, use_augmentation: bool = False):
        """
        Initialize the generator with a random seed.
        
        Args:
            seed: Random seed
            use_augmentation: Whether to use data augmentation
        """
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.use_augmentation = use_augmentation
        
        # Initialize augmentation modules if enabled
        if use_augmentation:
            self.domain_randomizer = DomainRandomizer(seed=seed)
            self.trajectory_augmenter = TrajectoryAugmenter(seed=seed)
            self.balanced_sampler = BalancedSampler(seed=seed)
        
        # Reward weights (Option 1: Enhanced Reward)
        self.reward_weights = {
            'edge_throughput': 0.5,      # QoE component
            'median_throughput': 0.2,    # QoE component
            'rssi_improvement': 0.1,      # NEW: RSSI reward for edge clients
            'stability_bonus': 0.1,       # NEW: Stability reward
            'p95_retry': 0.6,            # Reliability penalty (increased)
            'per': 0.4,                  # Reliability penalty
            'config_churn': 0.1          # NEW: Config churn penalty
        }
        
        # Track config changes for churn calculation (per episode)
        self.episode_actions = []  # Track actions in current episode
        
        # Safety thresholds
        self.safety_thresholds = {
            'p95_retry': 0.08,
            'min_rssi': -70.0,
            'tx_step_max': 6.0,
            'config_churn': 0.2
        }
    
    def _generate_initial_state(self) -> np.ndarray:
        """Generate a single initial state with realistic correlations."""
        state = np.zeros(self.NUM_FEATURES)
        
        # Client count (Poisson, mean 15)
        state[self.IDX_CLIENT_COUNT] = max(1, self.rng.poisson(15))
        
        # Median RSSI (Normal, influenced by client count)
        base_rssi = -65 + self.rng.normal(0, 8)
        # More clients often means some are farther away
        rssi_penalty = min(0, -0.3 * (state[self.IDX_CLIENT_COUNT] - 15))
        state[self.IDX_MEDIAN_RSSI] = np.clip(base_rssi + rssi_penalty, -90, -30)
        
        # Tx Power (Uniform in safety shield range)
        state[self.IDX_TX_POWER] = self.rng.uniform(self.TX_POWER_MIN, self.TX_POWER_MAX)
        
        # OBSS/PD Threshold (Uniform in safety shield range)
        state[self.IDX_OBSS_PD] = self.rng.uniform(self.OBSS_PD_MIN, self.OBSS_PD_MAX)
        
        # Noise floor (Normal around -92 dBm)
        state[self.IDX_NOISE_FLOOR] = np.clip(self.rng.normal(-92, 3), -100, -80)
        
        # Neighbor AP RSSI (interference)
        state[self.IDX_NEIGHBOR_RSSI] = np.clip(self.rng.normal(-70, 10), -90, -50)
        
        # Channel width (categorical: 20, 40, 80 MHz)
        state[self.IDX_CHANNEL_WIDTH] = self.rng.choice([20, 40, 80], p=[0.3, 0.4, 0.3])
        
        # Calculate derived metrics with correlations
        state = self._calculate_correlated_metrics(state)
        
        return state
    
    def _calculate_correlated_metrics(self, state: np.ndarray) -> np.ndarray:
        """Calculate metrics with realistic correlations using improved physics models."""
        
        # Calculate SINR (not just SNR) - accounts for interference
        sinr = calculate_sinr(
            rssi=state[self.IDX_MEDIAN_RSSI],
            noise_floor=state[self.IDX_NOISE_FLOOR],
            neighbor_rssi=state[self.IDX_NEIGHBOR_RSSI],
            interference_factor=0.5
        )
        
        # Interference factor (higher neighbor RSSI = more interference)
        interference = np.clip((state[self.IDX_NEIGHBOR_RSSI] + 90) / 40, 0, 1)
        
        # Load factor based on client count
        load_factor = state[self.IDX_CLIENT_COUNT] / 50.0
        
        # Select MCS based on SINR
        mcs = select_mcs(sinr, state[self.IDX_CHANNEL_WIDTH])
        
        # Calculate PER using PER curves
        per = calculate_per(sinr, mcs)
        
        # Add some variation to PER based on load and interference
        per_variation = self.rng.normal(0, 0.02)
        per = np.clip(per + per_variation + load_factor * 0.01 + interference * 0.01, 0.001, 0.5)
        state[self.IDX_P95_PER] = per
        
        # Calculate retry rate from PER (more realistic)
        base_retry = calculate_retry_rate_from_per(per, max_retries=7)
        # Add variation for load and interference
        retry_variation = self.rng.normal(0, 0.01)
        retry_load_factor = load_factor * 0.02
        retry_interference_factor = interference * 0.02
        state[self.IDX_P95_RETRY] = np.clip(
            base_retry + retry_variation + retry_load_factor + retry_interference_factor,
            0.01, 0.30
        )
        
        # Channel Utilization: based on load and interference
        base_util = 0.3
        state[self.IDX_CHANNEL_UTIL] = np.clip(
            base_util + load_factor * 0.4 + interference * 0.2 + self.rng.beta(2, 5) * 0.1,
            0, 1.0
        )
        
        # Average Throughput: using improved physics model with MCS
        avg_throughput = calculate_throughput(
            sinr=sinr,
            channel_width=state[self.IDX_CHANNEL_WIDTH],
            mcs=mcs,
            per=per,
            channel_util=state[self.IDX_CHANNEL_UTIL],
            frame_aggregation=True
        )
        # Add some natural variation
        state[self.IDX_AVG_THROUGHPUT] = np.clip(
            avg_throughput * np.exp(self.rng.normal(0, 0.15)),  # Reduced variation
            1, 500
        )
        
        # Edge P10 Throughput: using improved model
        edge_throughput = calculate_edge_throughput(
            avg_throughput=state[self.IDX_AVG_THROUGHPUT],
            sinr=sinr,
            edge_sinr_penalty=self.rng.uniform(5.0, 10.0)  # Variable penalty
        )
        state[self.IDX_EDGE_THROUGHPUT] = np.clip(
            edge_throughput * np.exp(self.rng.normal(0, 0.2)),
            0.5, 200
        )
        
        # Airtime Usage: correlated with channel utilization and load
        state[self.IDX_AIRTIME] = np.clip(
            state[self.IDX_CHANNEL_UTIL] * 0.8 + load_factor * 0.1 + self.rng.beta(2, 4) * 0.1,
            0, 1.0
        )
        
        # CCA Busy: correlated with channel utilization and interference
        state[self.IDX_CCA_BUSY] = np.clip(
            state[self.IDX_CHANNEL_UTIL] * 0.6 + interference * 0.3 + self.rng.beta(2, 6) * 0.1,
            0, 1.0
        )
        
        # Roaming Rate: higher with poor RSSI and high load
        rssi_roaming = max(0, (-70 - state[self.IDX_MEDIAN_RSSI]) / 20) * 0.05
        state[self.IDX_ROAMING_RATE] = np.clip(
            rssi_roaming + load_factor * 0.02 + self.rng.beta(1, 15) * 0.03,
            0, 0.20
        )
        
        return state
    
    def _apply_action(self, state: np.ndarray, action: int) -> np.ndarray:
        """Apply action to state and return next state."""
        next_state = state.copy()
        
        if action == self.ACTION_INCREASE_TX:
            # Increase Tx Power by 2 dBm
            new_power = state[self.IDX_TX_POWER] + 2.0
            next_state[self.IDX_TX_POWER] = np.clip(new_power, self.TX_POWER_MIN, self.TX_POWER_MAX)
            
        elif action == self.ACTION_DECREASE_TX:
            # Decrease Tx Power by 2 dBm
            new_power = state[self.IDX_TX_POWER] - 2.0
            next_state[self.IDX_TX_POWER] = np.clip(new_power, self.TX_POWER_MIN, self.TX_POWER_MAX)
            
        elif action == self.ACTION_INCREASE_OBSS:
            # Increase OBSS/PD threshold by 4 dBm (more spatial reuse)
            new_obss = state[self.IDX_OBSS_PD] + 4.0
            next_state[self.IDX_OBSS_PD] = np.clip(new_obss, self.OBSS_PD_MIN, self.OBSS_PD_MAX)
            
        elif action == self.ACTION_DECREASE_OBSS:
            # Decrease OBSS/PD threshold by 4 dBm (more isolation)
            new_obss = state[self.IDX_OBSS_PD] - 4.0
            next_state[self.IDX_OBSS_PD] = np.clip(new_obss, self.OBSS_PD_MIN, self.OBSS_PD_MAX)
            
        elif action == self.ACTION_INCREASE_CHANNEL_WIDTH:
            # Increase Channel Width: 20→40 or 40→80 MHz
            current_width = state[self.IDX_CHANNEL_WIDTH]
            if current_width == 20:
                next_state[self.IDX_CHANNEL_WIDTH] = 40
            elif current_width == 40:
                next_state[self.IDX_CHANNEL_WIDTH] = 80
            # If already 80, no change
            
        elif action == self.ACTION_DECREASE_CHANNEL_WIDTH:
            # Decrease Channel Width: 80→40 or 40→20 MHz
            current_width = state[self.IDX_CHANNEL_WIDTH]
            if current_width == 80:
                next_state[self.IDX_CHANNEL_WIDTH] = 40
            elif current_width == 40:
                next_state[self.IDX_CHANNEL_WIDTH] = 20
            # If already 20, no change
            
        elif action == self.ACTION_INCREASE_CHANNEL_NUMBER:
            # Increase Channel Number: Move to next DFS-compliant channel
            # Effect: Changes interference pattern (affects neighbor_ap_rssi)
            interference_change = self.rng.uniform(-3, 3)
            next_state[self.IDX_NEIGHBOR_RSSI] = np.clip(
                state[self.IDX_NEIGHBOR_RSSI] + interference_change,
                -90, -50
            )
            
        elif action == self.ACTION_DECREASE_CHANNEL_NUMBER:
            # Decrease Channel Number: Move to previous DFS-compliant channel
            # Effect: Changes interference pattern (affects neighbor_ap_rssi)
            interference_change = self.rng.uniform(-3, 3)
            next_state[self.IDX_NEIGHBOR_RSSI] = np.clip(
                state[self.IDX_NEIGHBOR_RSSI] + interference_change,
                -90, -50
            )
            
        # No-op: state unchanged (action == 8)
        
        # Add some natural variation to other metrics
        next_state = self._add_state_dynamics(state, next_state, action)
        
        return next_state
    
    def _add_state_dynamics(self, state: np.ndarray, next_state: np.ndarray, action: int) -> np.ndarray:
        """Add realistic dynamics based on action effects."""
        
        # Client count can change slightly
        client_change = self.rng.choice([-2, -1, 0, 1, 2], p=[0.1, 0.2, 0.4, 0.2, 0.1])
        next_state[self.IDX_CLIENT_COUNT] = max(1, min(50, state[self.IDX_CLIENT_COUNT] + client_change))
        
        # Tx Power affects RSSI (higher power = better RSSI for clients)
        tx_delta = next_state[self.IDX_TX_POWER] - state[self.IDX_TX_POWER]
        if tx_delta != 0:
            # RSSI improves with higher Tx power
            rssi_improvement = tx_delta * 0.5  # 0.5 dB RSSI per 1 dB Tx
            # Add small noise, but ensure physics is preserved
            # Noise should not override the physics (limit noise based on tx_delta direction)
            noise = self.rng.normal(0, 0.5)  # Reduced noise std from 1.0 to 0.5
            # Ensure noise doesn't reverse the physics direction
            if tx_delta > 0:
                # Increasing Tx: noise shouldn't make RSSI worse
                noise = max(noise, -0.5 * abs(tx_delta))
            else:
                # Decreasing Tx: noise shouldn't make RSSI better
                noise = min(noise, 0.5 * abs(tx_delta))
            
            next_state[self.IDX_MEDIAN_RSSI] = np.clip(
                state[self.IDX_MEDIAN_RSSI] + rssi_improvement + noise,
                -90, -30
            )
            
            # Recalculate correlated metrics with improved physics after RSSI change
            next_state = self._calculate_correlated_metrics(next_state)
        
        # OBSS/PD affects interference perception and throughput
        obss_delta = next_state[self.IDX_OBSS_PD] - state[self.IDX_OBSS_PD]
        if obss_delta != 0:
            # Higher OBSS-PD = more aggressive spatial reuse = potentially more interference
            if obss_delta > 0:
                # Increasing threshold: more reuse, potentially more interference
                interference_change = self.rng.uniform(0, 2)
                next_state[self.IDX_NEIGHBOR_RSSI] = np.clip(
                    state[self.IDX_NEIGHBOR_RSSI] + interference_change,
                    -90, -50
                )
            else:
                # Decreasing threshold: more conservative, less interference
                interference_change = self.rng.uniform(-2, 0)
                next_state[self.IDX_NEIGHBOR_RSSI] = np.clip(
                    state[self.IDX_NEIGHBOR_RSSI] + interference_change,
                    -90, -50
                )
            # Recalculate metrics after interference change
            next_state = self._calculate_correlated_metrics(next_state)
        
        # Noise floor has small random walk
        next_state[self.IDX_NOISE_FLOOR] = np.clip(
            state[self.IDX_NOISE_FLOOR] + self.rng.normal(0, 0.5),
            -100, -80
        )
        
        # Channel width changes handled in _apply_action
        # If channel width changed, recalculate throughput
        channel_width_delta = next_state[self.IDX_CHANNEL_WIDTH] - state[self.IDX_CHANNEL_WIDTH]
        if channel_width_delta != 0:
            # Channel width change affects throughput
            # Wider channel = more capacity, but may have different interference
            next_state = self._calculate_correlated_metrics(next_state)
        
        # Channel number changes affect interference (handled in _apply_action)
        # Recalculate correlated metrics if not already done
        if tx_delta == 0 and obss_delta == 0 and channel_width_delta == 0:
            next_state = self._calculate_correlated_metrics(next_state)
        
        return next_state
    
    def _calculate_reward(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        """
        Calculate ENHANCED reward based on performance changes.
        
        VERSION 2.0 - Major improvements:
        - Rewards scaled to [-1.5, +1.5] range (was ~0.01)
        - Percentage-based normalization for clear signal
        - State-quality based rewards (not just deltas)
        - Smart No-op bonus when network is healthy
        
        Target: Strong learning signal for all agents.
        """
        # Track config churn (per episode, using sliding window)
        self.episode_actions.append(action)
        if len(self.episode_actions) > 100:
            self.episode_actions = self.episode_actions[-100:]
        
        # ===== 1. THROUGHPUT IMPROVEMENT (percentage-based) =====
        # Edge throughput: most important for QoE
        edge_old = max(state[self.IDX_EDGE_THROUGHPUT], 1.0)
        edge_new = max(next_state[self.IDX_EDGE_THROUGHPUT], 1.0)
        edge_pct_change = (edge_new - edge_old) / edge_old
        edge_reward = np.clip(edge_pct_change * 5.0, -1.0, 1.0)  # 20% change = 1.0 reward
        
        # Average throughput
        avg_old = max(state[self.IDX_AVG_THROUGHPUT], 1.0)
        avg_new = max(next_state[self.IDX_AVG_THROUGHPUT], 1.0)
        avg_pct_change = (avg_new - avg_old) / avg_old
        avg_reward = np.clip(avg_pct_change * 5.0, -1.0, 1.0)
        
        # ===== 2. RSSI QUALITY (absolute quality, not just delta) =====
        # Target: -70 to -50 dBm is good
        rssi = next_state[self.IDX_MEDIAN_RSSI]
        if rssi >= -60:
            rssi_quality = 1.0  # Excellent
        elif rssi >= -70:
            rssi_quality = 0.5 + (rssi + 70) / 20  # -70 to -60 → 0.5 to 1.0
        elif rssi >= -80:
            rssi_quality = (rssi + 80) / 20  # -80 to -70 → 0 to 0.5
        else:
            rssi_quality = -0.5  # Poor signal penalty
        
        # RSSI improvement bonus
        rssi_delta = next_state[self.IDX_MEDIAN_RSSI] - state[self.IDX_MEDIAN_RSSI]
        rssi_improvement = np.clip(rssi_delta / 5.0, -0.5, 0.5)  # 5 dB = 0.5
        
        # ===== 3. RELIABILITY PENALTIES (relative to thresholds) =====
        # P95 Retry Rate: threshold 8%
        retry_rate = next_state[self.IDX_P95_RETRY]
        if retry_rate <= 0.05:
            retry_penalty = 0.0  # Good
        elif retry_rate <= 0.08:
            retry_penalty = (retry_rate - 0.05) / 0.03 * 0.3  # Warning zone
        else:
            retry_penalty = 0.3 + (retry_rate - 0.08) / 0.08 * 0.7  # Bad, scales up
        retry_penalty = np.clip(retry_penalty, 0, 1.0)
        
        # Retry rate INCREASE penalty (making things worse)
        retry_delta = next_state[self.IDX_P95_RETRY] - state[self.IDX_P95_RETRY]
        if retry_delta > 0.01:  # Increased by >1%
            retry_penalty += np.clip(retry_delta * 10, 0, 0.5)  # Extra penalty
        
        # PER: threshold 5%
        per = next_state[self.IDX_P95_PER]
        if per <= 0.03:
            per_penalty = 0.0
        elif per <= 0.05:
            per_penalty = (per - 0.03) / 0.02 * 0.3
        else:
            per_penalty = 0.3 + (per - 0.05) / 0.05 * 0.5
        per_penalty = np.clip(per_penalty, 0, 0.8)
        
        # ===== 4. STABILITY BONUS (smart No-op) =====
        # Reward No-op when network is already healthy
        stability_bonus = 0.0
        is_healthy = (retry_rate < 0.05 and per < 0.03 and rssi > -65)
        
        if action == self.ACTION_NOOP:
            if is_healthy:
                stability_bonus = 0.3  # Good: don't fix what ain't broken
            else:
                stability_bonus = -0.1  # Bad: should be taking action
        else:
            if is_healthy:
                stability_bonus = -0.1  # Unnecessary change
            # else: no bonus/penalty for action when needed
        
        # ===== 5. CONFIG CHURN PENALTY =====
        recent = self.episode_actions[-20:] if len(self.episode_actions) >= 20 else self.episode_actions
        changes = sum(1 for a in recent if a != self.ACTION_NOOP)
        churn_rate = changes / max(len(recent), 1)
        
        if churn_rate > 0.3:  # More than 30% actions are changes
            churn_penalty = (churn_rate - 0.3) * 0.5
        else:
            churn_penalty = 0.0
        
        # ===== FINAL REWARD CALCULATION =====
        reward = (
            # Positive: performance improvements
            0.35 * edge_reward +           # 35%: Edge throughput (QoE)
            0.15 * avg_reward +            # 15%: Average throughput
            0.10 * rssi_quality +          # 10%: RSSI quality
            0.05 * rssi_improvement +      # 5%: RSSI improvement
            0.15 * stability_bonus +       # 15%: Smart stability
            
            # Negative: reliability penalties
            -0.25 * retry_penalty +        # 25%: Retry rate
            -0.15 * per_penalty +          # 15%: Packet error rate
            -0.10 * churn_penalty          # 10%: Config churn
        )
        
        # Clip to target range
        reward = np.clip(reward, -1.5, 1.5)
        
        return float(reward)
    
    def _calculate_cost(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        """
        Calculate safety cost.
        
        Cost = 1.0 if hard constraint violated, else sum of soft penalties.
        """
        cost = 0.0
        
        # Hard constraints (any violation = cost 1.0)
        # Tx Power out of hard bounds
        if next_state[self.IDX_TX_POWER] < self.TX_POWER_HARD_MIN or \
           next_state[self.IDX_TX_POWER] > self.TX_POWER_HARD_MAX:
            return 1.0
        
        # OBSS-PD out of hard bounds
        if next_state[self.IDX_OBSS_PD] < self.OBSS_PD_HARD_MIN or \
           next_state[self.IDX_OBSS_PD] > self.OBSS_PD_HARD_MAX:
            return 1.0
        
        # Channel Width out of hard bounds [20, 80] MHz
        if next_state[self.IDX_CHANNEL_WIDTH] < 20 or \
           next_state[self.IDX_CHANNEL_WIDTH] > 80:
            return 1.0
        
        # Soft constraints (accumulate penalties)
        # P95 Retry Rate > 8%
        if next_state[self.IDX_P95_RETRY] > self.safety_thresholds['p95_retry']:
            cost += (next_state[self.IDX_P95_RETRY] - self.safety_thresholds['p95_retry'])
        
        # Tx Power step > 6 dB
        tx_step = abs(next_state[self.IDX_TX_POWER] - state[self.IDX_TX_POWER])
        if tx_step > self.safety_thresholds['tx_step_max']:
            cost += (tx_step - self.safety_thresholds['tx_step_max']) / 6.0
        
        # Client RSSI below -70 dBm (penalty proportional to shortfall)
        if next_state[self.IDX_MEDIAN_RSSI] < self.safety_thresholds['min_rssi']:
            cost += (self.safety_thresholds['min_rssi'] - next_state[self.IDX_MEDIAN_RSSI]) * 0.01
        
        # Increased retry rate (safety threshold violation)
        delta_retry = next_state[self.IDX_P95_RETRY] - state[self.IDX_P95_RETRY]
        if delta_retry > 0.02:  # Significant increase
            cost += delta_retry
        
        return float(min(cost, 1.0))  # Cap at 1.0
    
    def _select_behavior_action(self, state: np.ndarray) -> int:
        """
        Select action using a behavior policy for dataset generation.
        
        Uses a mixture of:
        - Random exploration (30%)
        - Heuristic policy (50%)
        - No-op (20%)
        """
        r = self.rng.random()
        
        if r < 0.2:
            # No-op
            return self.ACTION_NOOP
        elif r < 0.5:
            # Random action
            return self.rng.integers(0, self.NUM_ACTIONS)
        else:
            # Heuristic policy based on state
            return self._heuristic_action(state)
    
    def _heuristic_action(self, state: np.ndarray) -> int:
        """Simple heuristic policy for generating reasonable behavior data."""
        
        # If RSSI is low, try increasing Tx power
        if state[self.IDX_MEDIAN_RSSI] < -75:
            if state[self.IDX_TX_POWER] < self.TX_POWER_MAX - 2:
                return self.ACTION_INCREASE_TX
        
        # If RSSI is very high and retry rate is low, can decrease power
        if state[self.IDX_MEDIAN_RSSI] > -55 and state[self.IDX_P95_RETRY] < 0.03:
            if state[self.IDX_TX_POWER] > self.TX_POWER_MIN + 2:
                return self.ACTION_DECREASE_TX
        
        # If high interference, try decreasing OBSS-PD for more isolation
        if state[self.IDX_NEIGHBOR_RSSI] > -60:
            if state[self.IDX_OBSS_PD] > self.OBSS_PD_MIN + 4:
                return self.ACTION_DECREASE_OBSS
        
        # If low interference and high utilization, try more spatial reuse
        if state[self.IDX_NEIGHBOR_RSSI] < -75 and state[self.IDX_CHANNEL_UTIL] > 0.7:
            if state[self.IDX_OBSS_PD] < self.OBSS_PD_MAX - 4:
                return self.ACTION_INCREASE_OBSS
        
        # Default: no-op
        return self.ACTION_NOOP
    
    def generate_episode(self, max_steps: int = 100) -> List[Tuple]:
        """Generate a single episode of transitions."""
        # Reset episode tracking for churn calculation
        self.episode_actions = []
        
        transitions = []
        state = self._generate_initial_state()
        
        for step in range(max_steps):
            action = self._select_behavior_action(state)
            next_state = self._apply_action(state, action)
            reward = self._calculate_reward(state, action, next_state)
            cost = self._calculate_cost(state, action, next_state)
            
            # Episode terminates with small probability or if severe constraint violation
            done = self.rng.random() < 0.01 or cost >= 1.0
            
            transitions.append((
                state.copy(),
                action,
                reward,
                next_state.copy(),
                done,
                cost
            ))
            
            if done:
                break
            
            state = next_state
        
        return transitions
    
    def generate_dataset(
        self,
        num_samples: int = 10000,
        use_augmentation: bool = None,
        augmentation_factor: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Generate complete dataset with specified number of samples.
        
        Args:
            num_samples: Target number of samples
            use_augmentation: Whether to use data augmentation (None = use self.use_augmentation)
            augmentation_factor: Factor to multiply base samples (1.0 = no augmentation, 2.0 = double)
        
        Returns:
            Dictionary with keys: states, actions, rewards, next_states, dones, costs
        """
        if use_augmentation is None:
            use_augmentation = self.use_augmentation
        
        # Generate base samples
        base_samples = int(num_samples / (1 + augmentation_factor))
        all_transitions = []
        
        print(f"Generating {base_samples} base samples...")
        while len(all_transitions) < base_samples:
            episode = self.generate_episode()
            all_transitions.extend(episode)
        
        # Apply augmentation if enabled
        if use_augmentation and augmentation_factor > 0:
            print(f"Applying data augmentation (factor: {augmentation_factor})...")
            augmented_transitions = []
            
            # Augment states with domain randomization
            for state, action, reward, next_state, done, cost in all_transitions:
                # Augment state and next_state
                aug_state = self.domain_randomizer.augment_state(state)
                aug_next_state = self.domain_randomizer.augment_state(next_state)
                
                # Recalculate reward and cost for augmented states
                aug_reward = self._calculate_reward(aug_state, action, aug_next_state)
                aug_cost = self._calculate_cost(aug_state, action, aug_next_state)
                
                augmented_transitions.append((
                    aug_state, action, aug_reward, aug_next_state, done, aug_cost
                ))
            
            # Combine original and augmented
            all_transitions = all_transitions + augmented_transitions[:int(len(all_transitions) * augmentation_factor)]
        
        # Trim to exact size
        all_transitions = all_transitions[:num_samples]
        
        # Convert to arrays
        states = np.array([t[0] for t in all_transitions], dtype=np.float32)
        actions = np.array([t[1] for t in all_transitions], dtype=np.int64)
        rewards = np.array([t[2] for t in all_transitions], dtype=np.float32)
        next_states = np.array([t[3] for t in all_transitions], dtype=np.float32)
        dones = np.array([t[4] for t in all_transitions], dtype=np.bool_)
        costs = np.array([t[5] for t in all_transitions], dtype=np.float32)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'costs': costs
        }
    
    def save_dataset(self, dataset: Dict[str, np.ndarray], filepath: str, save_csv: bool = True):
        """
        Save dataset to HDF5 file and optionally to CSV.
        
        Args:
            dataset: Dictionary with states, actions, rewards, next_states, dones, costs
            filepath: Path to save HDF5 file
            save_csv: If True, also save as CSV
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save HDF5
        with h5py.File(filepath, 'w') as f:
            print(list(dataset.keys()))
            print(dataset["states"][0])
            for key, value in dataset.items():
                f.create_dataset(key, data=value, compression='gzip')
            
            # Store metadata
            f.attrs['num_samples'] = len(dataset['states'])
            f.attrs['num_features'] = self.NUM_FEATURES
            f.attrs['num_actions'] = self.NUM_ACTIONS
            f.attrs['seed'] = self.seed
        
        print(f"Dataset saved to {filepath}")
        print(f"  Samples: {len(dataset['states'])}")
        print(f"  Features: {self.NUM_FEATURES}")
        print(f"  Actions: {self.NUM_ACTIONS}")
        
        # Save CSV if requested
        if save_csv:
            csv_path = filepath.with_suffix('.csv')
            self.save_dataset_csv(dataset, csv_path)
    
    def save_dataset_csv(self, dataset: Dict[str, np.ndarray], csv_path: str):
        """
        Save dataset to CSV format.
        
        Args:
            dataset: Dictionary with states, actions, rewards, next_states, dones, costs
            csv_path: Path to save CSV file
        """
        import pandas as pd
        
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Feature names
        feature_names = [
            'client_count', 'median_rssi', 'p95_retry_rate', 'p95_per',
            'channel_utilization', 'avg_throughput', 'edge_p10_throughput',
            'neighbor_ap_rssi', 'obss_pd_threshold', 'tx_power',
            'noise_floor', 'channel_width', 'airtime_usage', 'cca_busy', 'roaming_rate'
        ]
        
        # Action names
        action_names = {
            0: 'Increase_Tx_Power_+2dBm',
            1: 'Decrease_Tx_Power_-2dBm',
            2: 'Increase_OBSS_PD_+4dBm',
            3: 'Decrease_OBSS_PD_-4dBm',
            4: 'Increase_Channel_Width',
            5: 'Decrease_Channel_Width',
            6: 'Increase_Channel_Number',
            7: 'Decrease_Channel_Number',
            8: 'No_op'
        }
        
        # Create DataFrame
        n_samples = len(dataset['states'])
        
        # State columns
        state_cols = {f'state_{name}': dataset['states'][:, i] for i, name in enumerate(feature_names)}
        
        # Next state columns
        next_state_cols = {f'next_state_{name}': dataset['next_states'][:, i] for i, name in enumerate(feature_names)}
        
        # Action
        actions = dataset['actions']
        action_cols = {
            'action_id': actions,
            'action_name': [action_names[int(a)] for a in actions]
        }
        
        # Other columns
        other_cols = {
            'reward': dataset['rewards'],
            'cost': dataset['costs'],
            'done': dataset['dones'].astype(int)
        }
        
        # Combine all columns
        all_data = {**state_cols, **action_cols, **other_cols, **next_state_cols}
        df = pd.DataFrame(all_data)
        
        # Reorder columns: states, action, reward, cost, done, next_states
        state_cols_order = [f'state_{name}' for name in feature_names]
        next_state_cols_order = [f'next_state_{name}' for name in feature_names]
        column_order = state_cols_order + ['action_id', 'action_name', 'reward', 'cost', 'done'] + next_state_cols_order
        
        df = df[column_order]
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        
        print(f"Dataset also saved to CSV: {csv_path}")
        print(f"  CSV rows: {len(df)}")
        print(f"  CSV columns: {len(df.columns)}")
    
    def get_dataset_statistics(self, dataset: Dict[str, np.ndarray]) -> Dict:
        """Compute and return dataset statistics."""
        stats = {
            'num_samples': len(dataset['states']),
            'feature_means': dataset['states'].mean(axis=0),
            'feature_stds': dataset['states'].std(axis=0),
            'action_distribution': np.bincount(dataset['actions'], minlength=self.NUM_ACTIONS) / len(dataset['actions']),
            'reward_mean': dataset['rewards'].mean(),
            'reward_std': dataset['rewards'].std(),
            'cost_mean': dataset['costs'].mean(),
            'cost_nonzero_rate': (dataset['costs'] > 0).mean(),
            'done_rate': dataset['dones'].mean()
        }
        return stats


def main():
    """Generate and save the RRM dataset."""
    print("=" * 60)
    print("RRM Safe RL Dataset Generator")
    print("=" * 60)
    
    generator = RRMDatasetGenerator(seed=42)
    
    print("\nGenerating 10,000 samples...")
    dataset = generator.generate_dataset(num_samples=10000)
    
    print("\nDataset Statistics:")
    stats = generator.get_dataset_statistics(dataset)
    print(f"  Total samples: {stats['num_samples']}")
    print(f"  Reward mean: {stats['reward_mean']:.4f} (std: {stats['reward_std']:.4f})")
    print(f"  Cost mean: {stats['cost_mean']:.4f}")
    print(f"  Non-zero cost rate: {stats['cost_nonzero_rate']:.2%}")
    print(f"  Episode termination rate: {stats['done_rate']:.2%}")
    print(f"\n  Action distribution:")
    action_names = ['Inc Tx', 'Dec Tx', 'Inc OBSS', 'Dec OBSS', 'Inc Ch Width', 'Dec Ch Width', 'Inc Ch Num', 'Dec Ch Num', 'No-op']
    for i, (name, prob) in enumerate(zip(action_names, stats['action_distribution'])):
        print(f"    {i}: {name}: {prob:.2%}")
    
    # Save dataset
    generator.save_dataset(dataset, "data/rrm_dataset.h5")
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

