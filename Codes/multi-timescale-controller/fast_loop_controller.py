"""
Fast Loop Controller for RRMEngine.

Real-time optimization running every step:
- Client steering based on QoE thresholds
- Load balancing across APs
- Uses ClientViewAPI for QoE monitoring
- Uses PolicyEngine for role-based thresholds
"""

from typing import Dict, List, Optional, Tuple
from datatype import AccessPoint, Client
from policy_engine import PolicyEngine
from config_engine import ConfigEngine
from clientview import ClientViewAPI
from utils import compute_distance


class FastLoopController:
    """
    Fast Loop Controller for real-time network optimization.
    
    Executes every simulation step to:
    - Steer clients with poor QoE to better APs
    - Balance load across APs
    - Respond quickly to QoE degradation
    """
    
    def __init__(self,
                 policy_engine: PolicyEngine,
                 config_engine: ConfigEngine,
                 client_view_api: ClientViewAPI,
                 clients: List[Client]):
        """
        Initialize Fast Loop Controller.
        
        Args:
            policy_engine: PolicyEngine instance
            config_engine: ConfigEngine instance
            client_view_api: ClientViewAPI instance
            clients: List of all clients
        """
        self.policy_engine = policy_engine
        self.config_engine = config_engine
        self.client_view_api = client_view_api
        self.clients = clients
        
        # Steering thresholds
        self.qoe_threshold = 0.5  # Steer if QoE below this
        self.rssi_threshold = -75.0  # Steer if RSSI below this (dBm)
        self.rssi_improvement_threshold = 5.0  # Minimum RSSI gain for steering (dB)
        
        # Load balancing
        self.enable_load_balancing = True
        self.max_load_imbalance = 3  # Max client difference before balancing
        
        # Hysteresis to prevent ping-pong
        self.min_association_time = 5  # Min steps before steering again
        
        # Tracking
        self.steering_count = 0
        self.last_steering_step = {}  # client_id -> step
    
    def execute(self) -> List[Tuple[int, int, int]]:
        """
        Main execution method - runs every step.
        
        Returns:
            List of steering actions: [(client_id, old_ap, new_ap), ...]
        """
        steering_actions = []
        
        # Get current QoE for all clients
        qoe_views = self.client_view_api.compute_all_views()
        
        # Build client QoE map
        client_qoe_map = {}
        for ap_id, ap_view in qoe_views.items():
            for client_result in ap_view.client_results:
                client_qoe_map[client_result.client_id] = client_result.qoe_ap
        
        # Identify steering candidates
        candidates = self._identify_steering_candidates(client_qoe_map)
        
        # Steer candidates to better APs
        for client in candidates:
            best_ap = self._find_best_ap(client, client_qoe_map)
            
            if best_ap and best_ap != client.associated_ap:
                old_ap = client.associated_ap
                
                # Update client association
                client.associated_ap = best_ap
                client.association_time = 0.0  # Reset association time
                
                steering_actions.append((client.id, old_ap, best_ap))
                self.last_steering_step[client.id] = self.steering_count
                
        # Load balancing (if enabled and no steering happened)
        if self.enable_load_balancing and len(steering_actions) == 0:
            balance_actions = self._balance_load()
            steering_actions.extend(balance_actions)
        
        self.steering_count += len(steering_actions)
        return steering_actions
    
    def _identify_steering_candidates(self, client_qoe_map: Dict[int, float]) -> List[Client]:
        """
        Identify clients that should be considered for steering.
        
        Args:
            client_qoe_map: Mapping of client_id to QoE score
            
        Returns:
            List of Client objects that are candidates for steering
        """
        candidates = []
        
        for client in self.clients:
            # Skip if not associated
            if client.associated_ap is None:
                continue
            
            # Check hysteresis - avoid steering too frequently
            if client.association_time < self.min_association_time:
                continue
            
            # Get client's QoE
            qoe = client_qoe_map.get(client.id, 1.0)
            
            # Check role-specific thresholds from SLO catalog
            role_id = self.policy_engine.get_client_role(client.id)
            
            # Reason 1: Poor QoE
            if qoe < self.qoe_threshold:
                candidates.append(client)
                continue
            
            # Reason 2: Poor RSSI
            if client.rssi_dbm < self.rssi_threshold:
                candidates.append(client)
                continue
            
            # Reason 3: Role-specific violations
            # Get enforcement rules for this client's role
            metrics = {
                'RSSI_dBm': client.rssi_dbm,
                'Retry_pct': client.retry_rate,
            }
            actions = self.policy_engine.evaluate_client_compliance(client.id, metrics)
            
            if 'Steer' in actions:
                candidates.append(client)
        
        return candidates
    
    def _find_best_ap(self, client: Client, client_qoe_map: Dict[int, float]) -> Optional[int]:
        """
        Find the best AP for a client to associate with.
        
        Args:
            client: Client to find AP for
            client_qoe_map: Current QoE scores
            
        Returns:
            AP ID of best AP, or None if current AP is best
        """
        aps = list(self.config_engine.aps.values())
        current_ap_id = client.associated_ap
        
        best_ap_id = current_ap_id
        best_score = float('-inf')
        
        for ap in aps:
            score = self._calculate_ap_score(client, ap, client_qoe_map)
            
            if score > best_score:
                best_score = score
                best_ap_id = ap.id
        
        # Only steer if new AP is significantly better
        if best_ap_id != current_ap_id:
            current_score = self._calculate_ap_score(
                client, 
                self.config_engine.aps[current_ap_id],
                client_qoe_map
            )
            
            # Require minimum improvement to avoid unnecessary steering
            if best_score > current_score + 0.1:  # Small threshold
                return best_ap_id
        
        return None
    
    def _calculate_ap_score(self, client: Client, ap: AccessPoint, 
                           client_qoe_map: Dict[int, float]) -> float:
        """
        Calculate score for associating a client with an AP.
        
        Higher score = better choice.
        
        Args:
            client: Client to evaluate
            ap: AccessPoint to evaluate
            client_qoe_map: Current QoE scores
            
        Returns:
            Score (higher is better)
        """
        score = 0.0
        
        # Factor 1: Signal strength (RSSI estimate)
        dist = compute_distance(client.x, client.y, ap.x, ap.y)
        
        # Simplified RSSI estimation: RSSI = TxPower - PathLoss
        # PathLoss ≈ 40 + 20*log10(dist) for dist in meters
        if dist > 0:
            import math
            path_loss = 40 + 20 * math.log10(dist) if dist >= 1 else 40
            estimated_rssi = ap.tx_power - path_loss
        else:
            estimated_rssi = ap.tx_power  # Very close
        
        # Normalize RSSI to [-90, -30] → [0, 1]
        rssi_score = (estimated_rssi - (-90)) / 60.0
        rssi_score = max(0.0, min(1.0, rssi_score))
        score += 0.5 * rssi_score  # 50% weight
        
        # Factor 2: AP load (fewer clients = better)
        ap_clients = [c for c in self.clients if c.associated_ap == ap.id]
        load_score = 1.0 / (1.0 + len(ap_clients) * 0.1)  # Decays with load
        score += 0.3 * load_score  # 30% weight
        
        # Factor 3: AP's average QoE
        qoe_views = self.client_view_api.compute_all_views()
        ap_view = qoe_views.get(ap.id)
        if ap_view and ap_view.num_clients > 0:
            qoe_score = ap_view.avg_qoe
        else:
            qoe_score = 0.7  # Neutral score for empty AP
        score += 0.2 * qoe_score  # 20% weight
        
        return score
    
    def _balance_load(self) -> List[Tuple[int, int, int]]:
        """
        Balance client load across APs.
        
        Moves clients from heavily loaded APs to lightly loaded APs.
        
        Returns:
            List of steering actions
        """
        aps = list(self.config_engine.aps.values())
        
        if len(aps) < 2:
            return []  # Need at least 2 APs to balance
        
        # Calculate load per AP
        ap_loads = {}
        for ap in aps:
            ap_clients = [c for c in self.clients if c.associated_ap == ap.id]
            ap_loads[ap.id] = ap_clients
        
        # Find most and least loaded APs
        most_loaded = max(ap_loads.items(), key=lambda x: len(x[1]))
        least_loaded = min(ap_loads.items(), key=lambda x: len(x[1]))
        
        load_diff = len(most_loaded[1]) - len(least_loaded[1])
        
        # Only balance if imbalance exceeds threshold
        if load_diff <= self.max_load_imbalance:
            return []
        
        # Move one client from most loaded to least loaded
        # Choose client with best signal to least loaded AP
        most_loaded_ap = self.config_engine.aps[most_loaded[0]]
        least_loaded_ap = self.config_engine.aps[least_loaded[0]]
        
        best_client = None
        best_rssi = float('-inf')
        
        for client in most_loaded[1]:
            # Check hysteresis
            if client.association_time < self.min_association_time:
                continue
            
            dist = compute_distance(client.x, client.y, least_loaded_ap.x, least_loaded_ap.y)
            if dist > 0:
                import math
                path_loss = 40 + 20 * math.log10(dist) if dist >= 1 else 40
                estimated_rssi = least_loaded_ap.tx_power - path_loss
            else:
                estimated_rssi = least_loaded_ap.tx_power
            
            if estimated_rssi > best_rssi and estimated_rssi > self.rssi_threshold:
                best_rssi = estimated_rssi
                best_client = client
        
        if best_client:
            old_ap = best_client.associated_ap
            best_client.associated_ap = least_loaded_ap.id
            best_client.association_time = 0.0
            return [(best_client.id, old_ap, least_loaded_ap.id)]
        
        return []
    
    def set_qoe_threshold(self, threshold: float):
        """Set QoE threshold for steering."""
        self.qoe_threshold = max(0.0, min(1.0, threshold))
    
    def set_rssi_threshold(self, threshold: float):
        """Set RSSI threshold for steering (dBm)."""
        self.rssi_threshold = threshold
    
    def enable_load_balance(self, enabled: bool):
        """Enable or disable load balancing."""
        self.enable_load_balancing = enabled
    
    def get_statistics(self) -> Dict[str, any]:
        """Get controller statistics."""
        return {
            'total_steers': self.steering_count,
            'qoe_threshold': self.qoe_threshold,
            'rssi_threshold': self.rssi_threshold,
            'load_balancing_enabled': self.enable_load_balancing
        }
    
    def print_status(self):
        """Print controller status."""
        print("\n" + "="*60)
        print("FAST LOOP CONTROLLER STATUS")
        print("="*60)
        print(f"QoE Threshold: {self.qoe_threshold:.2f}")
        print(f"RSSI Threshold: {self.rssi_threshold} dBm")
        print(f"Min Association Time: {self.min_association_time} steps")
        print(f"Load Balancing: {'Enabled' if self.enable_load_balancing else 'Disabled'}")
        print(f"Max Load Imbalance: {self.max_load_imbalance} clients")
        print(f"Total Steering Actions: {self.steering_count}")
        print()
