"""
Enhanced Reward Functions for Safe RL.

This module provides multiple reward function implementations:
1. BasicReward - Original simple reward
2. EnhancedReward - Current implementation with stability
3. AdvancedReward - NEW: Full-featured reward with all improvements

New components added:
- Latency penalty (critical for voice/video QoE)
- Fairness metric (Jain's fairness index)
- Interference reduction reward
- Energy efficiency bonus
- Long-term stability tracking
- Peak hour awareness
- Roaming smoothness

Reference: Problem Statement reward requirements + industry best practices.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class RewardType(Enum):
    """Available reward function types."""
    BASIC = "basic"
    ENHANCED = "enhanced"
    ADVANCED = "advanced"


@dataclass
class RewardConfig:
    """Configuration for reward function weights."""
    # QoE Components
    edge_throughput_weight: float = 0.25
    avg_throughput_weight: float = 0.10
    latency_weight: float = 0.15  # NEW
    
    # RSSI Components
    rssi_quality_weight: float = 0.08
    rssi_improvement_weight: float = 0.02
    
    # Reliability Penalties
    retry_penalty_weight: float = 0.15
    per_penalty_weight: float = 0.10
    
    # Stability Components
    stability_bonus_weight: float = 0.10
    churn_penalty_weight: float = 0.05
    
    # NEW Components
    fairness_weight: float = 0.05
    interference_weight: float = 0.08
    energy_efficiency_weight: float = 0.02
    roaming_smoothness_weight: float = 0.03
    peak_hour_multiplier: float = 1.5


class AdvancedRewardCalculator:
    """
    Advanced reward calculator with all improvements.
    
    Features:
    - Latency-aware rewards (critical for real-time apps)
    - Fairness index (prevent edge client starvation)
    - Interference reduction incentives
    - Energy efficiency bonuses
    - Long-term stability tracking
    - Peak hour awareness
    - Roaming smoothness
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
    
    # Action indices
    ACTION_NOOP = 4
    
    def __init__(self, config: Optional[RewardConfig] = None):
        """Initialize with configuration."""
        self.config = config or RewardConfig()
        
        # Tracking for long-term metrics
        self.episode_actions: List[int] = []
        self.throughput_history: List[float] = []
        self.rssi_history: List[float] = []
        self.stability_window = 20  # Steps to track for stability
    
    def reset_episode(self):
        """Reset episode-level tracking."""
        self.episode_actions = []
        self.throughput_history = []
        self.rssi_history = []
    
    def calculate_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        is_peak_hour: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate comprehensive reward with all components.
        
        Args:
            state: Current state (15 features)
            action: Action taken (0-4)
            next_state: Resulting state
            is_peak_hour: Whether it's peak usage time
            
        Returns:
            reward: Total reward value
            components: Dictionary of individual reward components
        """
        components = {}
        
        # Track action for churn calculation
        self.episode_actions.append(action)
        if len(self.episode_actions) > 100:
            self.episode_actions = self.episode_actions[-100:]
        
        # Track throughput for stability
        self.throughput_history.append(next_state[self.IDX_EDGE_THROUGHPUT])
        if len(self.throughput_history) > self.stability_window:
            self.throughput_history = self.throughput_history[-self.stability_window:]
        
        # ========== 1. THROUGHPUT REWARDS ==========
        edge_reward = self._calculate_throughput_reward(
            state[self.IDX_EDGE_THROUGHPUT],
            next_state[self.IDX_EDGE_THROUGHPUT]
        )
        components['edge_throughput'] = edge_reward * self.config.edge_throughput_weight
        
        avg_reward = self._calculate_throughput_reward(
            state[self.IDX_AVG_THROUGHPUT],
            next_state[self.IDX_AVG_THROUGHPUT]
        )
        components['avg_throughput'] = avg_reward * self.config.avg_throughput_weight
        
        # ========== 2. LATENCY REWARD (NEW) ==========
        # Approximate latency from retry rate and channel utilization
        latency_reward = self._calculate_latency_reward(next_state)
        components['latency'] = latency_reward * self.config.latency_weight
        
        # ========== 3. RSSI QUALITY ==========
        rssi_quality = self._calculate_rssi_quality(next_state[self.IDX_MEDIAN_RSSI])
        components['rssi_quality'] = rssi_quality * self.config.rssi_quality_weight
        
        rssi_improvement = self._calculate_rssi_improvement(
            state[self.IDX_MEDIAN_RSSI],
            next_state[self.IDX_MEDIAN_RSSI]
        )
        components['rssi_improvement'] = rssi_improvement * self.config.rssi_improvement_weight
        
        # ========== 4. RELIABILITY PENALTIES ==========
        retry_penalty = self._calculate_retry_penalty(
            state[self.IDX_P95_RETRY],
            next_state[self.IDX_P95_RETRY]
        )
        components['retry_penalty'] = -retry_penalty * self.config.retry_penalty_weight
        
        per_penalty = self._calculate_per_penalty(
            state[self.IDX_P95_PER],
            next_state[self.IDX_P95_PER]
        )
        components['per_penalty'] = -per_penalty * self.config.per_penalty_weight
        
        # ========== 5. STABILITY BONUS ==========
        stability_bonus = self._calculate_stability_bonus(state, action, next_state)
        components['stability'] = stability_bonus * self.config.stability_bonus_weight
        
        # ========== 6. CONFIG CHURN PENALTY ==========
        churn_penalty = self._calculate_churn_penalty()
        components['churn_penalty'] = -churn_penalty * self.config.churn_penalty_weight
        
        # ========== 7. FAIRNESS REWARD (NEW) ==========
        fairness_reward = self._calculate_fairness_reward(next_state)
        components['fairness'] = fairness_reward * self.config.fairness_weight
        
        # ========== 8. INTERFERENCE REDUCTION (NEW) ==========
        interference_reward = self._calculate_interference_reward(
            state[self.IDX_CCA_BUSY],
            next_state[self.IDX_CCA_BUSY],
            state[self.IDX_NEIGHBOR_RSSI],
            next_state[self.IDX_NEIGHBOR_RSSI]
        )
        components['interference'] = interference_reward * self.config.interference_weight
        
        # ========== 9. ENERGY EFFICIENCY (NEW) ==========
        energy_reward = self._calculate_energy_reward(
            state[self.IDX_TX_POWER],
            next_state[self.IDX_TX_POWER],
            next_state[self.IDX_MEDIAN_RSSI]
        )
        components['energy_efficiency'] = energy_reward * self.config.energy_efficiency_weight
        
        # ========== 10. ROAMING SMOOTHNESS (NEW) ==========
        roaming_reward = self._calculate_roaming_reward(
            state[self.IDX_ROAMING_RATE],
            next_state[self.IDX_ROAMING_RATE]
        )
        components['roaming'] = roaming_reward * self.config.roaming_smoothness_weight
        
        # ========== TOTAL REWARD ==========
        total_reward = sum(components.values())
        
        # Apply peak hour multiplier
        if is_peak_hour:
            total_reward *= self.config.peak_hour_multiplier
            components['peak_hour_bonus'] = total_reward * (self.config.peak_hour_multiplier - 1)
        
        # Clip to reasonable range
        total_reward = np.clip(total_reward, -2.0, 2.0)
        
        return float(total_reward), components
    
    def _calculate_throughput_reward(self, old: float, new: float) -> float:
        """Calculate percentage-based throughput improvement reward."""
        old = max(old, 1.0)
        pct_change = (new - old) / old
        return np.clip(pct_change * 5.0, -1.0, 1.0)
    
    def _calculate_latency_reward(self, state: np.ndarray) -> float:
        """
        Calculate latency reward based on proxy metrics.
        
        Latency is approximated from:
        - Retry rate (higher retries = higher latency)
        - Channel utilization (congestion = higher latency)
        - Airtime usage (contention = higher latency)
        
        Target latency thresholds:
        - Voice: <30ms (excellent), <50ms (good), <100ms (acceptable)
        - Video: <50ms (excellent), <100ms (good), <200ms (acceptable)
        """
        retry_rate = state[self.IDX_P95_RETRY]
        util = state[self.IDX_CHANNEL_UTIL]
        airtime = state[self.IDX_AIRTIME]
        
        # Approximate latency score (inverse of latency - higher is better)
        # This is a heuristic based on network conditions
        latency_score = 1.0 - (retry_rate * 3 + util * 0.3 + airtime * 0.2)
        
        # Normalize to reward range
        if latency_score >= 0.8:
            return 1.0  # Excellent latency
        elif latency_score >= 0.6:
            return 0.5  # Good latency
        elif latency_score >= 0.4:
            return 0.0  # Acceptable
        else:
            return -0.5  # Poor latency
    
    def _calculate_rssi_quality(self, rssi: float) -> float:
        """Calculate RSSI quality score."""
        if rssi >= -60:
            return 1.0  # Excellent
        elif rssi >= -70:
            return 0.5 + (rssi + 70) / 20  # Good to Excellent
        elif rssi >= -80:
            return (rssi + 80) / 20  # Poor to Good
        else:
            return -0.5  # Very poor
    
    def _calculate_rssi_improvement(self, old: float, new: float) -> float:
        """Calculate RSSI improvement reward."""
        delta = new - old
        return np.clip(delta / 5.0, -0.5, 0.5)
    
    def _calculate_retry_penalty(self, old: float, new: float) -> float:
        """Calculate retry rate penalty."""
        # Absolute penalty based on current rate
        if new <= 0.05:
            penalty = 0.0
        elif new <= 0.08:
            penalty = (new - 0.05) / 0.03 * 0.3
        else:
            penalty = 0.3 + (new - 0.08) / 0.08 * 0.7
        
        # Additional penalty for making it worse
        delta = new - old
        if delta > 0.01:
            penalty += np.clip(delta * 10, 0, 0.5)
        
        return np.clip(penalty, 0, 1.0)
    
    def _calculate_per_penalty(self, old: float, new: float) -> float:
        """Calculate packet error rate penalty."""
        if new <= 0.03:
            penalty = 0.0
        elif new <= 0.05:
            penalty = (new - 0.03) / 0.02 * 0.3
        else:
            penalty = 0.3 + (new - 0.05) / 0.05 * 0.5
        return np.clip(penalty, 0, 0.8)
    
    def _calculate_stability_bonus(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray
    ) -> float:
        """Calculate stability bonus for smart No-op decisions."""
        retry = next_state[self.IDX_P95_RETRY]
        per = next_state[self.IDX_P95_PER]
        rssi = next_state[self.IDX_MEDIAN_RSSI]
        
        is_healthy = (retry < 0.05 and per < 0.03 and rssi > -65)
        
        if action == self.ACTION_NOOP:
            if is_healthy:
                return 0.3  # Good: maintain working config
            else:
                return -0.1  # Bad: should be taking action
        else:
            if is_healthy:
                return -0.1  # Unnecessary change
            return 0.0  # Action when needed
    
    def _calculate_churn_penalty(self) -> float:
        """Calculate config churn penalty."""
        recent = self.episode_actions[-20:] if len(self.episode_actions) >= 20 else self.episode_actions
        if not recent:
            return 0.0
        
        changes = sum(1 for a in recent if a != self.ACTION_NOOP)
        churn_rate = changes / len(recent)
        
        if churn_rate > 0.3:
            return (churn_rate - 0.3) * 0.5
        return 0.0
    
    def _calculate_fairness_reward(self, state: np.ndarray) -> float:
        """
        Calculate fairness reward using ratio of edge to average throughput.
        
        Jain's fairness index approximation:
        - Edge throughput should be reasonable compared to average
        - Penalize when edge clients are starving
        """
        edge_tput = state[self.IDX_EDGE_THROUGHPUT]
        avg_tput = state[self.IDX_AVG_THROUGHPUT]
        
        if avg_tput < 1.0:
            return 0.0
        
        # Fairness ratio (edge P10 / average)
        # Ideal: edge is at least 30-50% of average
        fairness_ratio = edge_tput / avg_tput
        
        if fairness_ratio >= 0.5:
            return 1.0  # Excellent fairness
        elif fairness_ratio >= 0.3:
            return 0.5 + (fairness_ratio - 0.3) / 0.2 * 0.5
        elif fairness_ratio >= 0.1:
            return (fairness_ratio - 0.1) / 0.2 * 0.5
        else:
            return -0.5  # Edge clients starving
    
    def _calculate_interference_reward(
        self,
        old_cca: float,
        new_cca: float,
        old_neighbor: float,
        new_neighbor: float
    ) -> float:
        """
        Calculate interference reduction reward.
        
        Lower CCA busy and lower neighbor RSSI = less interference = better.
        """
        # CCA reduction reward
        cca_improvement = old_cca - new_cca
        cca_reward = np.clip(cca_improvement * 5, -0.5, 0.5)
        
        # Current CCA quality
        if new_cca <= 0.3:
            cca_quality = 0.5
        elif new_cca <= 0.5:
            cca_quality = 0.25
        elif new_cca <= 0.7:
            cca_quality = 0.0
        else:
            cca_quality = -0.25
        
        return cca_reward + cca_quality
    
    def _calculate_energy_reward(
        self,
        old_power: float,
        new_power: float,
        rssi: float
    ) -> float:
        """
        Calculate energy efficiency reward.
        
        Reward lower power when RSSI is still good.
        """
        # Power reduction with maintained quality
        power_reduction = old_power - new_power
        
        if rssi >= -65:
            # RSSI is good, reward power reduction
            return np.clip(power_reduction / 5, -0.5, 0.5)
        elif rssi >= -70:
            # RSSI is acceptable, small reward for efficiency
            return np.clip(power_reduction / 10, -0.25, 0.25)
        else:
            # RSSI is poor, don't reward power reduction
            return 0.0
    
    def _calculate_roaming_reward(self, old_rate: float, new_rate: float) -> float:
        """
        Calculate roaming smoothness reward.
        
        Lower roaming rate = more stable clients = better.
        """
        # Roaming reduction reward
        improvement = old_rate - new_rate
        improvement_reward = np.clip(improvement * 10, -0.5, 0.5)
        
        # Absolute roaming quality
        if new_rate <= 0.02:
            quality = 0.5  # Excellent
        elif new_rate <= 0.05:
            quality = 0.25
        elif new_rate <= 0.10:
            quality = 0.0
        else:
            quality = -0.25  # Too much roaming
        
        return improvement_reward + quality
    
    def get_reward_breakdown_str(self, components: Dict[str, float]) -> str:
        """Get formatted string of reward breakdown."""
        lines = ["Reward Breakdown:"]
        lines.append("-" * 40)
        
        total = 0
        for name, value in sorted(components.items(), key=lambda x: -abs(x[1])):
            sign = "+" if value >= 0 else ""
            lines.append(f"  {name:20s}: {sign}{value:.4f}")
            total += value
        
        lines.append("-" * 40)
        lines.append(f"  {'TOTAL':20s}: {'+' if total >= 0 else ''}{total:.4f}")
        
        return "\n".join(lines)


def create_reward_calculator(
    reward_type: str = "advanced",
    config: Optional[RewardConfig] = None
) -> AdvancedRewardCalculator:
    """Factory function to create reward calculator."""
    return AdvancedRewardCalculator(config)


def demo():
    """Demo the advanced reward calculator."""
    print("Advanced Reward Calculator Demo")
    print("=" * 60)
    
    calc = AdvancedRewardCalculator()
    
    # Sample state transition
    state = np.array([
        15,      # client_count
        -72,     # median_rssi (below threshold)
        0.09,    # p95_retry (above 8%)
        0.03,    # p95_per
        0.65,    # channel_util
        80,      # avg_throughput
        18,      # edge_throughput (low)
        -65,     # neighbor_rssi
        -75,     # obss_pd
        15,      # tx_power
        -92,     # noise_floor
        40,      # channel_width
        0.5,     # airtime
        0.4,     # cca_busy
        0.02,    # roaming_rate
    ])
    
    # After increasing Tx power
    next_state = state.copy()
    next_state[1] = -68   # RSSI improved
    next_state[2] = 0.06  # Retry rate decreased
    next_state[5] = 95    # Avg throughput improved
    next_state[6] = 28    # Edge throughput improved
    next_state[9] = 17    # Tx power increased
    
    reward, components = calc.calculate_reward(state, action=0, next_state=next_state)
    
    print(f"\nAction: Increase Tx Power (+2 dBm)")
    print(f"Total Reward: {reward:.4f}")
    print()
    print(calc.get_reward_breakdown_str(components))


if __name__ == "__main__":
    demo()

