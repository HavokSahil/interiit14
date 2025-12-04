"""
Explainability Module for Safe RL Agents.

Provides human-readable explanations for why the RL agent selected a particular action.
This is critical for:
1. Debugging agent behavior
2. Building trust with network operators
3. Compliance and audit requirements
4. Understanding when the safety shield intervenes

Reference: Problem Statement requirement for explainability and reason codes.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ReasonCode(Enum):
    """
    Reason codes for action selection.
    
    These provide human-readable explanations for why the agent took a specific action.
    """
    # Throughput-related reasons
    LOW_EDGE_THROUGHPUT = "LOW_EDGE_THROUGHPUT"
    LOW_AVG_THROUGHPUT = "LOW_AVG_THROUGHPUT"
    HIGH_THROUGHPUT_STABLE = "HIGH_THROUGHPUT_STABLE"
    
    # RSSI-related reasons
    LOW_RSSI_CLIENTS = "LOW_RSSI_CLIENTS"
    RSSI_IMPROVING = "RSSI_IMPROVING"
    RSSI_OPTIMAL = "RSSI_OPTIMAL"
    
    # Retry/Error-related reasons
    HIGH_RETRY_RATE = "HIGH_RETRY_RATE"
    HIGH_PER = "HIGH_PER"
    LOW_ERROR_RATES = "LOW_ERROR_RATES"
    
    # Interference-related reasons
    HIGH_INTERFERENCE = "HIGH_INTERFERENCE"
    LOW_INTERFERENCE = "LOW_INTERFERENCE"
    HIGH_CHANNEL_UTILIZATION = "HIGH_CHANNEL_UTILIZATION"
    
    # Safety-related reasons
    SAFETY_SHIELD_BLOCKED = "SAFETY_SHIELD_BLOCKED"
    TX_POWER_AT_LIMIT = "TX_POWER_AT_LIMIT"
    OBSS_PD_AT_LIMIT = "OBSS_PD_AT_LIMIT"
    CONFIG_CHURN_LIMIT = "CONFIG_CHURN_LIMIT"
    
    # Policy-related reasons
    EXPLORATION_BONUS = "EXPLORATION_BONUS"
    CONSERVATIVE_POLICY = "CONSERVATIVE_POLICY"
    HIGH_Q_VALUE = "HIGH_Q_VALUE"
    NETWORK_STABLE = "NETWORK_STABLE"
    
    # Action-specific reasons
    INCREASE_COVERAGE = "INCREASE_COVERAGE"
    REDUCE_INTERFERENCE = "REDUCE_INTERFERENCE"
    ENABLE_SPATIAL_REUSE = "ENABLE_SPATIAL_REUSE"
    INCREASE_ISOLATION = "INCREASE_ISOLATION"
    MAINTAIN_STABILITY = "MAINTAIN_STABILITY"


@dataclass
class ActionExplanation:
    """Complete explanation for an action selection."""
    action: int
    action_name: str
    primary_reason: ReasonCode
    secondary_reasons: List[ReasonCode]
    confidence: float
    q_values: Optional[Dict[str, float]]
    state_analysis: Dict[str, str]
    safety_info: Dict[str, bool]
    recommendation: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'action': self.action,
            'action_name': self.action_name,
            'primary_reason': self.primary_reason.value,
            'secondary_reasons': [r.value for r in self.secondary_reasons],
            'confidence': self.confidence,
            'q_values': self.q_values,
            'state_analysis': self.state_analysis,
            'safety_info': self.safety_info,
            'recommendation': self.recommendation
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"═══════════════════════════════════════════════════════════",
            f"ACTION EXPLANATION",
            f"═══════════════════════════════════════════════════════════",
            f"",
            f"Selected Action: {self.action_name} (Action {self.action})",
            f"Confidence: {self.confidence:.1%}",
            f"",
            f"PRIMARY REASON: {self.primary_reason.value}",
        ]
        
        if self.secondary_reasons:
            lines.append(f"Secondary Reasons: {', '.join(r.value for r in self.secondary_reasons)}")
        
        lines.extend([
            f"",
            f"STATE ANALYSIS:",
            f"───────────────────────────────────────────────────────────",
        ])
        
        for key, value in self.state_analysis.items():
            lines.append(f"  {key}: {value}")
        
        lines.extend([
            f"",
            f"SAFETY STATUS:",
            f"───────────────────────────────────────────────────────────",
        ])
        
        for key, value in self.safety_info.items():
            status = "✓" if value else "✗"
            lines.append(f"  [{status}] {key}")
        
        if self.q_values:
            lines.extend([
                f"",
                f"Q-VALUES (Agent's Action Preferences):",
                f"───────────────────────────────────────────────────────────",
            ])
            for action_name, q_val in self.q_values.items():
                lines.append(f"  {action_name}: {q_val:+.4f}")
        
        lines.extend([
            f"",
            f"RECOMMENDATION:",
            f"───────────────────────────────────────────────────────────",
            f"  {self.recommendation}",
            f"",
            f"═══════════════════════════════════════════════════════════",
        ])
        
        return "\n".join(lines)


class ActionExplainer:
    """
    Explainability engine for Safe RL agents.
    
    Provides detailed explanations for why the agent selected a particular action,
    including:
    - State analysis (what the agent "sees")
    - Q-value breakdown (what the agent "thinks")
    - Safety analysis (what constraints apply)
    - Human-readable reason codes
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
    
    # Action names
    ACTION_NAMES = {
        0: "Increase Tx Power (+2 dBm)",
        1: "Decrease Tx Power (-2 dBm)",
        2: "Increase OBSS-PD (+4 dBm)",
        3: "Decrease OBSS-PD (-4 dBm)",
        4: "Increase Channel Width",
        5: "Decrease Channel Width",
        6: "Increase Channel Number",
        7: "Decrease Channel Number",
        8: "No-op (Maintain)"
    }
    
    # Thresholds for analysis
    THRESHOLDS = {
        'low_rssi': -70.0,
        'good_rssi': -60.0,
        'high_retry': 0.08,
        'high_per': 0.05,
        'high_util': 0.7,
        'low_throughput': 20.0,
        'high_interference': -60.0,
        'tx_power_min': 10.0,
        'tx_power_max': 20.0,
        'obss_pd_min': -82.0,
        'obss_pd_max': -68.0,
        'channel_width_min': 20.0,
        'channel_width_max': 80.0,
    }
    
    def __init__(self, safety_module=None):
        """Initialize explainer with optional safety module."""
        self.safety = safety_module
    
    def explain_action(
        self,
        state: np.ndarray,
        action: int,
        q_values: Optional[np.ndarray] = None,
        agent_type: str = "DQN",
        denormalize_fn: Optional[callable] = None
    ) -> ActionExplanation:
        """
        Generate a complete explanation for why this action was selected.
        
        Args:
            state: Current state (may be normalized)
            action: Selected action index
            q_values: Q-values for all actions (if available)
            agent_type: Type of agent (CQL, DQN, PPO, RCPO)
            denormalize_fn: Function to denormalize state
            
        Returns:
            ActionExplanation with full details
        """
        # Denormalize state if needed
        if denormalize_fn is not None:
            state = denormalize_fn(state)
        
        # Analyze state
        state_analysis = self._analyze_state(state)
        
        # Analyze safety
        safety_info = self._analyze_safety(state, action)
        
        # Determine reasons
        primary_reason, secondary_reasons = self._determine_reasons(
            state, action, q_values, agent_type, safety_info
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(q_values, action)
        
        # Format Q-values
        q_dict = None
        if q_values is not None:
            q_dict = {
                self.ACTION_NAMES[i]: float(q_values[i])
                for i in range(len(q_values))
            }
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            state, action, primary_reason, state_analysis
        )
        
        return ActionExplanation(
            action=action,
            action_name=self.ACTION_NAMES[action],
            primary_reason=primary_reason,
            secondary_reasons=secondary_reasons,
            confidence=confidence,
            q_values=q_dict,
            state_analysis=state_analysis,
            safety_info=safety_info,
            recommendation=recommendation
        )
    
    def _analyze_state(self, state: np.ndarray) -> Dict[str, str]:
        """Analyze state and return human-readable descriptions."""
        analysis = {}
        
        # Client count
        clients = int(state[self.IDX_CLIENT_COUNT])
        if clients < 5:
            analysis['Client Load'] = f"Low ({clients} clients)"
        elif clients < 20:
            analysis['Client Load'] = f"Medium ({clients} clients)"
        else:
            analysis['Client Load'] = f"High ({clients} clients)"
        
        # RSSI
        rssi = state[self.IDX_MEDIAN_RSSI]
        if rssi < self.THRESHOLDS['low_rssi']:
            analysis['Signal Strength'] = f"Poor ({rssi:.1f} dBm) - Below -70 dBm threshold"
        elif rssi < self.THRESHOLDS['good_rssi']:
            analysis['Signal Strength'] = f"Acceptable ({rssi:.1f} dBm)"
        else:
            analysis['Signal Strength'] = f"Good ({rssi:.1f} dBm)"
        
        # Retry rate
        retry = state[self.IDX_P95_RETRY] * 100
        if retry > self.THRESHOLDS['high_retry'] * 100:
            analysis['Retry Rate'] = f"HIGH ({retry:.1f}%) - Above 8% threshold ⚠"
        else:
            analysis['Retry Rate'] = f"Normal ({retry:.1f}%)"
        
        # PER
        per = state[self.IDX_P95_PER] * 100
        if per > self.THRESHOLDS['high_per'] * 100:
            analysis['Packet Error Rate'] = f"HIGH ({per:.1f}%) ⚠"
        else:
            analysis['Packet Error Rate'] = f"Normal ({per:.1f}%)"
        
        # Channel utilization
        util = state[self.IDX_CHANNEL_UTIL] * 100
        if util > self.THRESHOLDS['high_util'] * 100:
            analysis['Channel Utilization'] = f"HIGH ({util:.1f}%) - Consider spatial reuse"
        else:
            analysis['Channel Utilization'] = f"Normal ({util:.1f}%)"
        
        # Throughput
        edge_tput = state[self.IDX_EDGE_THROUGHPUT]
        avg_tput = state[self.IDX_AVG_THROUGHPUT]
        if edge_tput < self.THRESHOLDS['low_throughput']:
            analysis['Edge Throughput'] = f"LOW ({edge_tput:.1f} Mbps) - QoE at risk ⚠"
        else:
            analysis['Edge Throughput'] = f"Acceptable ({edge_tput:.1f} Mbps)"
        analysis['Avg Throughput'] = f"{avg_tput:.1f} Mbps"
        
        # Interference
        neighbor = state[self.IDX_NEIGHBOR_RSSI]
        if neighbor > self.THRESHOLDS['high_interference']:
            analysis['Interference'] = f"HIGH (Neighbor AP: {neighbor:.1f} dBm) ⚠"
        else:
            analysis['Interference'] = f"Low (Neighbor AP: {neighbor:.1f} dBm)"
        
        # Current configuration
        tx_power = state[self.IDX_TX_POWER]
        obss_pd = state[self.IDX_OBSS_PD]
        channel_width = state[self.IDX_CHANNEL_WIDTH]
        analysis['Current Tx Power'] = f"{tx_power:.1f} dBm (Range: 10-20 dBm)"
        analysis['Current OBSS-PD'] = f"{obss_pd:.1f} dBm (Range: -82 to -68 dBm)"
        analysis['Current Channel Width'] = f"{channel_width:.0f} MHz (20/40/80 MHz)"
        
        return analysis
    
    def _analyze_safety(self, state: np.ndarray, action: int) -> Dict[str, bool]:
        """Analyze safety constraints."""
        tx_power = state[self.IDX_TX_POWER]
        obss_pd = state[self.IDX_OBSS_PD]
        channel_width = state[self.IDX_CHANNEL_WIDTH]
        retry = state[self.IDX_P95_RETRY]
        
        safety_info = {
            'Tx Power in safe range': self.THRESHOLDS['tx_power_min'] <= tx_power <= self.THRESHOLDS['tx_power_max'],
            'OBSS-PD in safe range': self.THRESHOLDS['obss_pd_min'] <= obss_pd <= self.THRESHOLDS['obss_pd_max'],
            'Channel Width in safe range': self.THRESHOLDS['channel_width_min'] <= channel_width <= self.THRESHOLDS['channel_width_max'],
            'Retry rate below threshold': retry <= self.THRESHOLDS['high_retry'],
            'Action is safe': True,  # Will be updated below
        }
        
        # Check if action would violate safety
        if action == 0:  # Increase Tx
            safety_info['Action is safe'] = tx_power + 2 <= self.THRESHOLDS['tx_power_max']
        elif action == 1:  # Decrease Tx
            safety_info['Action is safe'] = tx_power - 2 >= self.THRESHOLDS['tx_power_min']
        elif action == 2:  # Increase OBSS-PD
            safety_info['Action is safe'] = obss_pd + 4 <= self.THRESHOLDS['obss_pd_max']
        elif action == 3:  # Decrease OBSS-PD
            safety_info['Action is safe'] = obss_pd - 4 >= self.THRESHOLDS['obss_pd_min']
        elif action == 4:  # Increase Channel Width
            # 20→40 or 40→80, already at 80 = no change (safe)
            safety_info['Action is safe'] = channel_width < 80
        elif action == 5:  # Decrease Channel Width
            # 80→40 or 40→20, already at 20 = no change (safe)
            safety_info['Action is safe'] = channel_width > 20
        elif action == 6 or action == 7:  # Channel Number changes
            # Channel number changes are generally safe (DFS-compliant)
            safety_info['Action is safe'] = True
        # Action 8 (No-op) is always safe
        
        return safety_info
    
    def _determine_reasons(
        self,
        state: np.ndarray,
        action: int,
        q_values: Optional[np.ndarray],
        agent_type: str,
        safety_info: Dict[str, bool]
    ) -> Tuple[ReasonCode, List[ReasonCode]]:
        """Determine primary and secondary reasons for action selection."""
        secondary_reasons = []
        
        # Extract state values
        rssi = state[self.IDX_MEDIAN_RSSI]
        retry = state[self.IDX_P95_RETRY]
        per = state[self.IDX_P95_PER]
        util = state[self.IDX_CHANNEL_UTIL]
        edge_tput = state[self.IDX_EDGE_THROUGHPUT]
        avg_tput = state[self.IDX_AVG_THROUGHPUT]
        neighbor = state[self.IDX_NEIGHBOR_RSSI]
        tx_power = state[self.IDX_TX_POWER]
        obss_pd = state[self.IDX_OBSS_PD]
        channel_width = state[self.IDX_CHANNEL_WIDTH]
        
        # Check for safety-related reasons first
        if not safety_info['Action is safe']:
            return ReasonCode.SAFETY_SHIELD_BLOCKED, [ReasonCode.CONSERVATIVE_POLICY]
        
        # Determine based on action and state
        if action == 0:  # Increase Tx Power
            if rssi < self.THRESHOLDS['low_rssi']:
                primary = ReasonCode.LOW_RSSI_CLIENTS
                secondary_reasons.append(ReasonCode.INCREASE_COVERAGE)
            elif edge_tput < self.THRESHOLDS['low_throughput']:
                primary = ReasonCode.LOW_EDGE_THROUGHPUT
                secondary_reasons.append(ReasonCode.INCREASE_COVERAGE)
            else:
                primary = ReasonCode.HIGH_Q_VALUE
            
            if tx_power >= self.THRESHOLDS['tx_power_max'] - 2:
                secondary_reasons.append(ReasonCode.TX_POWER_AT_LIMIT)
                
        elif action == 1:  # Decrease Tx Power
            if neighbor > self.THRESHOLDS['high_interference']:
                primary = ReasonCode.HIGH_INTERFERENCE
                secondary_reasons.append(ReasonCode.REDUCE_INTERFERENCE)
            elif rssi > self.THRESHOLDS['good_rssi']:
                primary = ReasonCode.RSSI_OPTIMAL
                secondary_reasons.append(ReasonCode.REDUCE_INTERFERENCE)
            else:
                primary = ReasonCode.HIGH_Q_VALUE
                
        elif action == 2:  # Increase OBSS-PD (more spatial reuse)
            if util > self.THRESHOLDS['high_util']:
                primary = ReasonCode.HIGH_CHANNEL_UTILIZATION
                secondary_reasons.append(ReasonCode.ENABLE_SPATIAL_REUSE)
            elif neighbor < -70:
                primary = ReasonCode.LOW_INTERFERENCE
                secondary_reasons.append(ReasonCode.ENABLE_SPATIAL_REUSE)
            else:
                primary = ReasonCode.HIGH_Q_VALUE
                
        elif action == 3:  # Decrease OBSS-PD (more isolation)
            if neighbor > self.THRESHOLDS['high_interference']:
                primary = ReasonCode.HIGH_INTERFERENCE
                secondary_reasons.append(ReasonCode.INCREASE_ISOLATION)
            elif retry > self.THRESHOLDS['high_retry']:
                primary = ReasonCode.HIGH_RETRY_RATE
                secondary_reasons.append(ReasonCode.INCREASE_ISOLATION)
            else:
                primary = ReasonCode.HIGH_Q_VALUE
                
        elif action == 4:  # Increase Channel Width
            if avg_tput < 30 and channel_width < 80:
                primary = ReasonCode.LOW_AVG_THROUGHPUT
                secondary_reasons.append(ReasonCode.ENABLE_SPATIAL_REUSE)
            elif edge_tput < self.THRESHOLDS['low_throughput']:
                primary = ReasonCode.LOW_EDGE_THROUGHPUT
                secondary_reasons.append(ReasonCode.ENABLE_SPATIAL_REUSE)
            else:
                primary = ReasonCode.HIGH_Q_VALUE
                
        elif action == 5:  # Decrease Channel Width
            if neighbor > self.THRESHOLDS['high_interference']:
                primary = ReasonCode.HIGH_INTERFERENCE
                secondary_reasons.append(ReasonCode.INCREASE_ISOLATION)
            elif retry > self.THRESHOLDS['high_retry']:
                primary = ReasonCode.HIGH_RETRY_RATE
                secondary_reasons.append(ReasonCode.INCREASE_ISOLATION)
            else:
                primary = ReasonCode.HIGH_Q_VALUE
                
        elif action == 6 or action == 7:  # Change Channel Number
            if neighbor > self.THRESHOLDS['high_interference']:
                primary = ReasonCode.HIGH_INTERFERENCE
                secondary_reasons.append(ReasonCode.REDUCE_INTERFERENCE)
            elif retry > self.THRESHOLDS['high_retry']:
                primary = ReasonCode.HIGH_RETRY_RATE
                secondary_reasons.append(ReasonCode.REDUCE_INTERFERENCE)
            else:
                primary = ReasonCode.HIGH_Q_VALUE
                
        else:  # No-op (action == 8)
            if retry < 0.05 and per < 0.03 and rssi > -65:
                primary = ReasonCode.NETWORK_STABLE
                secondary_reasons.append(ReasonCode.MAINTAIN_STABILITY)
            elif agent_type == "CQL":
                primary = ReasonCode.CONSERVATIVE_POLICY
                secondary_reasons.append(ReasonCode.MAINTAIN_STABILITY)
            else:
                primary = ReasonCode.HIGH_Q_VALUE
                secondary_reasons.append(ReasonCode.MAINTAIN_STABILITY)
        
        # Add agent-specific reasons
        if agent_type == "CQL" and action == 8:
            if ReasonCode.CONSERVATIVE_POLICY not in secondary_reasons:
                secondary_reasons.append(ReasonCode.CONSERVATIVE_POLICY)
        
        return primary, secondary_reasons
    
    def _calculate_confidence(
        self,
        q_values: Optional[np.ndarray],
        action: int
    ) -> float:
        """Calculate confidence in action selection."""
        if q_values is None:
            return 0.5  # Unknown confidence
        
        # Softmax to get probabilities
        q_max = np.max(q_values)
        exp_q = np.exp(q_values - q_max)  # Numerical stability
        probs = exp_q / np.sum(exp_q)
        
        return float(probs[action])
    
    def _generate_recommendation(
        self,
        state: np.ndarray,
        action: int,
        primary_reason: ReasonCode,
        state_analysis: Dict[str, str]
    ) -> str:
        """Generate a human-readable recommendation."""
        action_name = self.ACTION_NAMES[action]
        
        recommendations = {
            ReasonCode.LOW_RSSI_CLIENTS: 
                f"Increasing Tx power to improve signal strength for edge clients. "
                f"Current RSSI is below -70 dBm threshold.",
            
            ReasonCode.LOW_EDGE_THROUGHPUT:
                f"Taking action to improve edge client throughput, which is currently low. "
                f"This should improve QoE for worst-performing clients.",
            
            ReasonCode.HIGH_RETRY_RATE:
                f"High retry rate detected (>{8}%). Taking action to reduce retransmissions "
                f"and improve network efficiency.",
            
            ReasonCode.HIGH_INTERFERENCE:
                f"High interference from neighboring APs detected. Adjusting configuration "
                f"to reduce co-channel interference.",
            
            ReasonCode.HIGH_CHANNEL_UTILIZATION:
                f"Channel utilization is high (>70%). Enabling more spatial reuse to "
                f"improve overall network capacity.",
            
            ReasonCode.NETWORK_STABLE:
                f"Network is performing well (low errors, good signal). Maintaining current "
                f"configuration to preserve stability.",
            
            ReasonCode.CONSERVATIVE_POLICY:
                f"CQL agent is being conservative with limited training data. No-op selected "
                f"to avoid potential negative outcomes from untested actions.",
            
            ReasonCode.SAFETY_SHIELD_BLOCKED:
                f"The proposed action was blocked by the safety shield. Selecting a safe "
                f"alternative to maintain system stability.",
            
            ReasonCode.HIGH_Q_VALUE:
                f"Agent selected '{action_name}' based on learned Q-values indicating "
                f"this action has the highest expected return in this state.",
            
            ReasonCode.RSSI_OPTIMAL:
                f"RSSI is already good. Reducing Tx power to minimize interference "
                f"with neighboring cells while maintaining acceptable coverage.",
        }
        
        # Add recommendations for channel width/number actions
        if action == 4:  # Increase Channel Width
            channel_width = state[self.IDX_CHANNEL_WIDTH]
            if channel_width == 20:
                return f"Increasing channel width from 20 to 40 MHz to double data rate capacity. " \
                       f"This should improve throughput for all clients."
            elif channel_width == 40:
                return f"Increasing channel width from 40 to 80 MHz to quadruple data rate capacity. " \
                       f"This maximizes throughput but may increase interference risk."
        elif action == 5:  # Decrease Channel Width
            channel_width = state[self.IDX_CHANNEL_WIDTH]
            if channel_width == 80:
                return f"Decreasing channel width from 80 to 40 MHz to reduce interference and improve " \
                       f"stability. Trade-off: lower maximum throughput."
            elif channel_width == 40:
                return f"Decreasing channel width from 40 to 20 MHz to minimize interference and improve " \
                       f"coexistence. Trade-off: lower maximum throughput."
        elif action == 6 or action == 7:  # Change Channel Number
            return f"Changing channel number to reduce co-channel interference from neighboring APs. " \
                   f"This should improve SINR and reduce retry rates."
        
        return recommendations.get(
            primary_reason,
            f"Selected '{action_name}' based on current network conditions."
        )
    
    def explain_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        q_values_batch: Optional[np.ndarray] = None,
        agent_type: str = "DQN",
        denormalize_fn: Optional[callable] = None
    ) -> List[ActionExplanation]:
        """Generate explanations for a batch of actions."""
        explanations = []
        
        for i in range(len(states)):
            q_vals = q_values_batch[i] if q_values_batch is not None else None
            exp = self.explain_action(
                states[i], actions[i], q_vals, agent_type, denormalize_fn
            )
            explanations.append(exp)
        
        return explanations
    
    def generate_summary_report(
        self,
        explanations: List[ActionExplanation]
    ) -> str:
        """Generate a summary report from multiple explanations."""
        if not explanations:
            return "No explanations to summarize."
        
        # Count reasons
        reason_counts = {}
        action_counts = {i: 0 for i in range(9)}
        total_confidence = 0
        
        for exp in explanations:
            reason = exp.primary_reason.value
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            action_counts[exp.action] += 1
            total_confidence += exp.confidence
        
        # Sort by count
        sorted_reasons = sorted(reason_counts.items(), key=lambda x: -x[1])
        
        lines = [
            "═══════════════════════════════════════════════════════════",
            "EXPLAINABILITY SUMMARY REPORT",
            "═══════════════════════════════════════════════════════════",
            f"",
            f"Total Actions Analyzed: {len(explanations)}",
            f"Average Confidence: {total_confidence / len(explanations):.1%}",
            f"",
            f"ACTION DISTRIBUTION:",
            f"───────────────────────────────────────────────────────────",
        ]
        
        for action, count in action_counts.items():
            pct = count / len(explanations) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            lines.append(f"  {self.ACTION_NAMES[action]}: [{bar}] {pct:.1f}%")
        
        lines.extend([
            f"",
            f"TOP REASONS FOR ACTIONS:",
            f"───────────────────────────────────────────────────────────",
        ])
        
        for reason, count in sorted_reasons[:5]:
            pct = count / len(explanations) * 100
            lines.append(f"  {reason}: {count} ({pct:.1f}%)")
        
        lines.extend([
            f"",
            f"═══════════════════════════════════════════════════════════",
        ])
        
        return "\n".join(lines)


def demo():
    """Demo the explainability module."""
    print("Explainability Module Demo")
    print("=" * 60)
    
    # Create sample state
    state = np.array([
        15,      # client_count
        -72,     # median_rssi (below threshold)
        0.09,    # p95_retry (above 8%)
        0.03,    # p95_per
        0.65,    # channel_util
        80,      # avg_throughput
        18,      # edge_throughput (low)
        -65,     # neighbor_rssi (high interference)
        -75,     # obss_pd
        15,      # tx_power
        -92,     # noise_floor
        40,      # channel_width
        0.5,     # airtime
        0.4,     # cca_busy
        0.02,    # roaming_rate
    ])
    
    # Sample Q-values (simulating DQN output)
    q_values = np.array([0.5, -0.2, 0.3, 0.1, 0.15])
    
    explainer = ActionExplainer()
    
    # Explain action 0 (Increase Tx Power)
    explanation = explainer.explain_action(
        state, action=0, q_values=q_values, agent_type="DQN"
    )
    
    print(explanation)


if __name__ == "__main__":
    demo()

