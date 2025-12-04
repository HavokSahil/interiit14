"""
Slow Loop Controller for RRMEngine.

Long-term optimization running periodically:
- Channel assignment optimization
- Transmit power optimization
- Uses SensingAPI for interference detection
- Uses ClientViewAPI for QoE monitoring
"""

from typing import Dict, List, Optional, Tuple
from datatype import AccessPoint, Client
from policy_engine import PolicyEngine
from config_engine import ConfigEngine, APConfig, NetworkConfig
from sensing import SensingAPI
from clientview import ClientViewAPI
from utils import compute_distance
import random
import production_package.run_inference as run_inference
model_dir = "production_package/model"
config_path = "production_package/config/config.yaml"


class SlowLoopController:
    """
    Slow Loop Controller for long-term network optimization.
    
    Executes periodically (e.g., every 100 steps) to optimize:
    - Channel assignments
    - Transmit power levels
    
    Uses sensing data and QoE metrics to make informed decisions.
    """
    
    def __init__(self,
                 policy_engine: PolicyEngine,
                 config_engine: ConfigEngine,
                 sensing_api: Optional[SensingAPI],
                 client_view_api: ClientViewAPI,
                 period: int = 100):
        """
        Initialize Slow Loop Controller.
        
        Args:
            policy_engine: PolicyEngine instance
            config_engine: ConfigEngine instance
            sensing_api: SensingAPI instance (optional)
            client_view_api: ClientViewAPI instance
            period: Steps between executions
        """
        self.policy_engine = policy_engine
        self.config_engine = config_engine
        self.sensing_api = sensing_api
        self.client_view_api = client_view_api
        self.period = period
        self.stats = {}
        self.ensemble, self.model = run_inference.load_ensemble(model_dir, config_path)
        self.explainer = run_inference.ActionExplainer(safety_module=self.ensemble.safety if hasattr(self.ensemble, 'safety') else None)
        def denormalize_fn(state):
            mean = 0
            if hasattr(self.ensemble, 'state_mean') and self.ensemble.state_mean is not None:
                mean = self.ensemble.state_mean
            std = self.ensemble.state_std
            return state * std + mean
        self.denormalize_fn = denormalize_fn
        
        # Optimization settings
        self.allowed_channels = [1, 2, 3, 6, 7, 10, 11, 36, 40, 44, 48, 52, 149, 153, 157, 161]
        self.power_levels = [10.0, 15.0, 20.0, 25.0, 30.0]  # dBm
        
        # Tracking
        self.last_execution = -period  # Execute on first call
        self.optimization_mode = "channel"  # "channel", "power", or "both"
    
    def should_execute(self, step: int) -> bool:
        """
        Determine if slow loop should execute at this step.
        
        Args:
            step: Current simulation step
            
        Returns:
            True if should execute
        """
        return (step - self.last_execution) >= self.period
    
    def execute(self, step: int, safe_rl_data) -> Optional[NetworkConfig]:
        """
        Main execution method.
        
        Args:
            step: Current simulation step
            
        Returns:
            NetworkConfig with new configuration, or None if no changes
        """
        if not self.should_execute(step):
            return None
        
        self.last_execution = step
        config = self.config_engine.get_current_config()
        self.bandwidths = [20, 40, 80]
        before = config
        for i, j in safe_rl_data.items():
            inference = run_inference.run_inference_on_state(j, self.ensemble, self.explainer, self.denormalize_fn)
            ap_config = config.ap_configs[i]
            print(f'[Slow Loop] AP {i} Action: {inference["ACTION"]}, Reason: {inference["REASON"]}, Confidence: {inference["CONFIDENCE"]}, Status: {inference["STATUS"]}, Current QoE: {inference["Current_QoE"]}')
            if inference["ACTION"] in self.stats:
                self.stats[inference["ACTION"]] += 1
            else:
                self.stats[inference["ACTION"]] = 1
            if inference["ACTION"] == "+2 dBm Tx Power":
                if ap_config.tx_power >= max(self.power_levels):
                    continue
                new_power = -1.0
                for power_level in self.power_levels:
                    if power_level > ap_config.tx_power:
                        new_power = power_level
                        break
                ap_config.tx_power = new_power
            elif inference["ACTION"] == "-2 dBm Tx Power":
                if ap_config.tx_power <= min(self.power_levels):
                    continue
                new_power = 30.0
                for power_level in self.power_levels[::-1]:
                    if power_level < ap_config.tx_power:
                        new_power = power_level
                        break
                ap_config.tx_power = new_power
            elif inference["ACTION"] == "+4 dB OBSS-PD":
                ap_config.obss_pd_threshold += 4
            elif inference["ACTION"] == "-4 dB OBSS-PD":
                ap_config.obss_pd_threshold -= 4
            elif inference["ACTION"] == "Increase Channel Width":
                if ap_config.bandwidth >= max(self.bandwidths):
                    continue
                new_bandwidth = 20
                for bandwidth in self.bandwidths:
                    if bandwidth > ap_config.bandwidth:
                        new_bandwidth = bandwidth
                        break
                ap_config.bandwidth = new_bandwidth
            elif inference["ACTION"] == "Decrease Channel Width":
                if ap_config.bandwidth <= min(self.bandwidths):
                    continue
                new_bandwidth = 80
                for bandwidth in self.bandwidths[::-1]:
                    if bandwidth < ap_config.bandwidth:
                        new_bandwidth = bandwidth
                        break
                ap_config.bandwidth = new_bandwidth
            elif inference["ACTION"] == "Increase Channel Number":
                if ap_config.channel >= max(self.allowed_channels):
                    continue
                new_channel = 1
                for channel in self.allowed_channels:
                    if channel > ap_config.channel:
                        new_channel = channel
                        break
                ap_config.channel = new_channel
            elif inference["ACTION"] == "Decrease Channel Number":
                if ap_config.channel >= min(self.allowed_channels):
                    continue
                new_channel = 11
                for channel in self.allowed_channels[::-1]:
                    if channel < ap_config.channel:
                        new_channel = channel
                        break
                ap_config.channel = new_channel
            config.ap_configs[i] = ap_config
#        print(config == before)
        return config
                
        
        # Choose optimization based on mode
#        if self.optimization_mode == "channel":
#            return self.optimize_channels()
#        elif self.optimization_mode == "power":
#            return self.optimize_power()
#        elif self.optimization_mode == "both":
#            # First optimize channels, then power
#            channel_config = self.optimize_channels()
#            if channel_config:
#                self.config_engine.apply_config(channel_config)
#            return self.optimize_power()
        
        return None
    
    def optimize_channels(self) -> Optional[NetworkConfig]:
        """
        Optimize channel assignments for all APs.
        
        Strategy:
        1. Calculate interference score for each AP on each channel
        2. Use greedy algorithm to assign channels
        3. Prioritize APs with poorest QoE or highest interference
        
        Returns:
            NetworkConfig with new channel assignments, or None if no changes
        """
        import time
        
        aps = list(self.config_engine.aps.values())
        
        if len(aps) == 0:
            return None
        
        # Calculate scores for each AP on each channel
        ap_channel_scores = {}
        for ap in aps:
            scores = {}
            for channel in self.allowed_channels:
                scores[channel] = self._calculate_channel_score(ap, channel)
            ap_channel_scores[ap.id] = scores
        
        # Get current QoE to prioritize APs
        qoe_views = self.client_view_api.compute_all_views()
        ap_priorities = []
        for ap in aps:
            qoe_view = qoe_views.get(ap.id)
            # Lower QoE = higher priority
            priority = 1.0 - qoe_view.avg_qoe if qoe_view and qoe_view.num_clients > 0 else 0.5
            ap_priorities.append((ap, priority))
        
        # Sort APs by priority (highest first)
        ap_priorities.sort(key=lambda x: x[1], reverse=True)
        
        # Greedy channel assignment
        assigned_channels = {}
        channel_usage = {ch: [] for ch in self.allowed_channels}
        
        for ap, _ in ap_priorities:
            best_channel = None
            best_score = float('-inf')
            
            # Consider interference from already assigned APs
            for channel in self.allowed_channels:
                score = ap_channel_scores[ap.id][channel]
                
                # Penalize channels already used by nearby APs
                for other_ap_id in channel_usage[channel]:
                    other_ap = self.config_engine.aps[other_ap_id]
                    dist = compute_distance(ap.x, ap.y, other_ap.x, other_ap.y)
                    if dist < 50:  # Close proximity penalty
                        score -= 0.3
                
                if score > best_score:
                    best_score = score
                    best_channel = channel
            
            assigned_channels[ap.id] = best_channel
            channel_usage[best_channel].append(ap.id)
        
        # Build configuration
        ap_configs = []
        changes_made = False
        
        for ap in aps:
            new_channel = assigned_channels[ap.id]
            if new_channel != ap.channel:
                changes_made = True
            
            ap_configs.append(self.config_engine.build_channel_config(ap.id, new_channel))
        
        if not changes_made:
            return None
        
        return self.config_engine.build_network_config(
            ap_configs,
            metadata={'optimization': 'channel', 'step': self.last_execution}
        )
    
    def _calculate_channel_score(self, ap: AccessPoint, channel: int) -> float:
        """
        Calculate score for an AP on a specific channel.
        
        Higher score = better channel choice.
        
        Args:
            ap: AccessPoint to evaluate
            channel: Channel to evaluate
            
        Returns:
            Score (higher is better)
        """
        score = 1.0  # Base score
        
        # Factor 1: Interference from sensing (if available)
        if self.sensing_api:
            sensing_results = self.sensing_api.compute_sensing_results()
            ap_sensing = sensing_results.get(ap.id)
            
            if ap_sensing:
                # Check if major interferer is on this channel
                interferer_channel = ap_sensing.center_frequency
                # Convert frequency to channel (approximate)
                interferer_ch = self._freq_to_channel(interferer_channel)
                
                if interferer_ch == channel:
                    # Heavy penalty for interferer on same channel
                    score -= 0.5 * ap_sensing.confidence
        
        # Factor 2: Current QoE on this channel (if AP is already on it)
        if ap.channel == channel:
            qoe_views = self.client_view_api.compute_all_views()
            ap_view = qoe_views.get(ap.id)
            if ap_view and ap_view.num_clients > 0:
                # Bonus for good current QoE
                score += 0.3 * ap_view.avg_qoe
        
        # Factor 3: Avoid channels with high usage (simulated)
        # In real implementation, check neighboring AP channels
        
        return score
    
    def _freq_to_channel(self, freq_ghz: float) -> int:
        """Convert frequency to WiFi channel."""
        # 2.4 GHz band approximation
        if 2.4 <= freq_ghz <= 2.5:
            channel = int((freq_ghz - 2.407) / 0.005)
            # Map to nearest allowed channel
            if channel <= 3:
                return 1
            elif channel <= 8:
                return 6
            else:
                return 11
        return 6  # Default
    
    def optimize_power(self) -> Optional[NetworkConfig]:
        """
        Optimize transmit power for all APs.
        
        Strategy:
        1. Evaluate each power level for each AP
        2. Select power that maximizes coverage while minimizing interference
        3. Consider client QoE and signal strength
        
        Returns:
            NetworkConfig with new power settings, or None if no changes
        """
        import time
        
        aps = list(self.config_engine.aps.values())
        clients = list(self.client_view_api.clients)
        
        if len(aps) == 0:
            return None
        
        # Get current QoE
        qoe_views = self.client_view_api.compute_all_views()
        
        # Optimize power for each AP
        optimal_powers = {}
        changes_made = False
        
        for ap in aps:
            best_power = ap.tx_power
            best_score = self._evaluate_power(ap, ap.tx_power, clients, qoe_views)
            
            for power in self.power_levels:
                if power == ap.tx_power:
                    continue
                
                score = self._evaluate_power(ap, power, clients, qoe_views)
                
                if score > best_score:
                    best_score = score
                    best_power = power
            
            optimal_powers[ap.id] = best_power
            if best_power != ap.tx_power:
                changes_made = True
        
        if not changes_made:
            return None
        
        # Build configuration
        ap_configs = []
        for ap in aps:
            ap_configs.append(
                self.config_engine.build_power_config(ap.id, optimal_powers[ap.id])
            )
        
        return self.config_engine.build_network_config(
            ap_configs,
            metadata={'optimization': 'power', 'step': self.last_execution}
        )
    
    def _evaluate_power(self, ap: AccessPoint, power: float, 
                       clients: List[Client], qoe_views: Dict) -> float:
        """
        Evaluate a power level for an AP.
        
        Args:
            ap: AccessPoint to evaluate
            power: Power level to test
            clients: List of all clients
            qoe_views: Current QoE views
            
        Returns:
            Score (higher is better)
        """
        score = 0.0
        
        # Get clients associated with this AP
        ap_clients = [c for c in clients if c.associated_ap == ap.id]
        
        if len(ap_clients) == 0:
            # No clients, prefer moderate power
            return -abs(power - 20.0) / 10.0
        
        # Factor 1: Coverage (higher power = better RSSI for distant clients)
        for client in ap_clients:
            dist = compute_distance(ap.x, ap.y, client.x, client.y)
            # Estimate RSSI improvement with higher power
            # Simplified: each 5dBm power increase improves RSSI by ~5dB
            estimated_rssi_delta = (power - ap.tx_power)
            
            # Reward if it helps poor RSSI clients
            if client.rssi_dbm < -70:
                score += estimated_rssi_delta * 0.02  # Small bonus
        
        # Factor 2: Interference (higher power = more interference to others)
        # Penalty for very high power
        if power > 25:
            score -= (power - 25) * 0.05
        
        # Factor 3: Current QoE
        ap_view = qoe_views.get(ap.id)
        if ap_view and ap_view.num_clients > 0:
            # If QoE is already good, don't change much
            if ap_view.avg_qoe > 0.7:
                score -= abs(power - ap.tx_power) * 0.1
        
        return score
    
    def set_optimization_mode(self, mode: str):
        """
        Set optimization mode.
        
        Args:
            mode: "channel", "power", or "both"
        """
        if mode in ["channel", "power", "both"]:
            self.optimization_mode = mode
    
    def print_status(self):
        """Print controller status."""
        print("\n" + "="*60)
        print("SLOW LOOP CONTROLLER STATUS")
        print("="*60)
        print(f"Period: {self.period} steps")
        print(f"Last Execution: Step {self.last_execution}")
#        print(f"Optimization Mode: {self.optimization_mode}")
#        print(f"Allowed Channels: {self.allowed_channels}")
#        print(f"Power Levels: {self.power_levels} dBm")
        for i, j in self.stats.items():
            print(f"{i} Events: {j}")
        print()
