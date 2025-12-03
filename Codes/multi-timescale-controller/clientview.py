"""
ClientView API for computing Quality of Experience (QoE) metrics.

This module provides tools for computing AP-side QoE for each client
associated with an access point, using five component scores:
- S: Signal Quality (based on RSSI)
- T: Throughput (based on allocated throughput vs max rate)
- R: Reliability (based on retry rate)
- L: Latency (based on inactivity time)
- A: Activity (based on packet counts)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datatype import AccessPoint, Client


# QoE Component Weights
W_SIGNAL = 0.28
W_THROUGHPUT = 0.32
W_RELIABILITY = 0.15
W_LATENCY = 0.15
W_ACTIVITY = 0.10

# RSSI Normalization Constants
RSSI_MIN = -90.0  # dBm
RSSI_MAX = -30.0  # dBm
RSSI_RANGE = 60.0  # MAX - MIN

# Reliability Coefficients
ALPHA = 0.6  # Weight for retry rate
BETA = 0.4   # Weight for FCS error rate (not used currently)

# Latency Threshold
LATENCY_THRESHOLD_MS = 5000.0  # milliseconds

# Activity Threshold
MAX_FRAMES = 1000  # Maximum frames for normalization


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to [min_val, max_val] range."""
    return max(min_val, min(max_val, value))


@dataclass
class QoEComponents:
    """Individual QoE component scores (each in range [0, 1])."""
    signal_quality: float  # S: Signal quality score
    throughput: float      # T: Throughput score
    reliability: float     # R: Reliability score
    latency: float         # L: Latency/Responsiveness score
    activity: float        # A: Activity score


@dataclass
class ClientQoEResult:
    """QoE result for a single client."""
    client_id: int
    components: QoEComponents
    qoe_ap: float  # Overall AP-side QoE score (0.0-1.0)


@dataclass
class APClientView:
    """Aggregated QoE results for an access point."""
    ap_id: int
    client_results: List[ClientQoEResult]
    avg_qoe: float  # Average QoE across all clients
    min_qoe: float  # Minimum QoE (worst client)
    max_qoe: float  # Maximum QoE (best client)
    num_clients: int  # Number of clients


class ClientViewAPI:
    """
    ClientView API for computing QoE metrics for each AP's clients.
    
    This API:
    1. Matches clients to their associated access points
    2. Computes QoE component scores for each client
    3. Calculates overall AP-side QoE
    4. Provides aggregated statistics per AP
    """
    
    def __init__(self, access_points: List[AccessPoint], clients: List[Client]):
        """
        Initialize the ClientView API.
        
        Args:
            access_points: List of access points
            clients: List of all clients in the network
        """
        self.access_points = access_points
        self.clients = clients
    
    @staticmethod
    def compute_signal_quality(client: Client) -> float:
        """
        Compute signal quality score (S) using RSSI.
        
        Formula: S = clamp((RSSI - (-90)) / 60, 0, 1)
        
        - RSSI ≤ -90 dBm → S = 0
        - RSSI ≥ -30 dBm → S = 1
        - Example: RSSI = -50 dBm → S = 0.67
        
        Args:
            client: Client object with rssi_dbm field
            
        Returns:
            Signal quality score in [0, 1]
        """
        rssi = client.rssi_dbm
        s = (rssi - RSSI_MIN) / RSSI_RANGE
        return clamp(s, 0.0, 1.0)
    
    @staticmethod
    def compute_throughput_score(client: Client) -> float:
        """
        Compute throughput score (T) using allocated throughput vs max rate.
        
        Original formula uses geometric mean of uplink/downlink bitrates,
        but we approximate using: T = throughput_mbps / max_rate_mbps
        
        Args:
            client: Client object with throughput_mbps and max_rate_mbps
            
        Returns:
            Throughput score in [0, 1]
        """
        if client.max_rate_mbps <= 0:
            return 0.0
        
        t = client.throughput_mbps / client.max_rate_mbps
        return clamp(t, 0.0, 1.0)
    
    @staticmethod
    def compute_reliability_score(client: Client) -> float:
        """
        Compute reliability score (R) using retry rate.
        
        Formula: R = 1 - (α × retry_rate + β × fcs_rate)
        
        Where:
        - α = 0.6
        - β = 0.4
        - retry_rate is in range [0, 1] (converted from percentage)
        - fcs_rate = 0 (not tracked currently)
        
        Args:
            client: Client object with retry_rate (0-100%)
            
        Returns:
            Reliability score in [0, 1]
        """
        # Convert retry_rate from percentage (0-100) to fraction (0-1)
        retry_fraction = client.retry_rate / 100.0
        
        # FCS error rate not tracked, assume 0
        fcs_fraction = 0.0
        
        r = 1.0 - (ALPHA * retry_fraction + BETA * fcs_fraction)
        return clamp(r, 0.0, 1.0)
    
    @staticmethod
    def compute_latency_score(client: Client) -> float:
        """
        Compute latency score (L) using inactivity timer.
        
        Formula: L = clamp(1 - inactive_msec / 5000, 0, 1)
        
        - inactive = 0 ms → L = 1.0 (best)
        - inactive = 2500 ms → L = 0.50
        - inactive ≥ 5000 ms → L = 0.0 (worst)
        
        Args:
            client: Client object with inactive_msec field
            
        Returns:
            Latency score in [0, 1]
        """
        l = 1.0 - (client.inactive_msec / LATENCY_THRESHOLD_MS)
        return clamp(l, 0.0, 1.0)
    
    @staticmethod
    def compute_activity_score(client: Client) -> float:
        """
        Compute activity score (A) using packet counts.
        
        Formula: A = clamp((tx_packets + rx_packets) / max_frames, 0, 1)
        
        Stations above max_frames saturate at A = 1.
        
        Args:
            client: Client object with tx_packets and rx_packets
            
        Returns:
            Activity score in [0, 1]
        """
        total_packets = client.tx_packets + client.rx_packets
        a = total_packets / MAX_FRAMES
        return clamp(a, 0.0, 1.0)
    
    def compute_qoe_components(self, client: Client) -> QoEComponents:
        """
        Compute all five QoE components for a client.
        
        Args:
            client: Client object
            
        Returns:
            QoEComponents with all five scores
        """
        return QoEComponents(
            signal_quality=self.compute_signal_quality(client),
            throughput=self.compute_throughput_score(client),
            reliability=self.compute_reliability_score(client),
            latency=self.compute_latency_score(client),
            activity=self.compute_activity_score(client)
        )
    
    @staticmethod
    def compute_qoe_ap(components: QoEComponents) -> float:
        """
        Calculate overall AP-side QoE using weighted formula.
        
        Formula: Q_AP = w_S×S + w_T×T + w_R×R + w_L×L + w_A×A
        
        Weights:
        - w_S = 0.28 (Signal Quality)
        - w_T = 0.32 (Throughput)
        - w_R = 0.15 (Reliability)
        - w_L = 0.15 (Latency)
        - w_A = 0.10 (Activity)
        
        Args:
            components: QoEComponents object
            
        Returns:
            Overall QoE score in [0, 1]
        """
        qoe = (
            W_SIGNAL * components.signal_quality +
            W_THROUGHPUT * components.throughput +
            W_RELIABILITY * components.reliability +
            W_LATENCY * components.latency +
            W_ACTIVITY * components.activity
        )
        return qoe
    
    def compute_client_qoe(self, client: Client) -> ClientQoEResult:
        """
        Compute complete QoE result for a single client.
        
        Args:
            client: Client object
            
        Returns:
            ClientQoEResult with components and overall QoE
        """
        components = self.compute_qoe_components(client)
        qoe_ap = self.compute_qoe_ap(components)
        
        return ClientQoEResult(
            client_id=client.id,
            components=components,
            qoe_ap=qoe_ap
        )
    
    def get_ap_clients(self, ap: AccessPoint) -> List[Client]:
        """
        Get all clients associated with a specific AP.
        
        Args:
            ap: AccessPoint object
            
        Returns:
            List of clients where associated_ap matches ap.id
        """
        return [client for client in self.clients 
                if client.associated_ap == ap.id]
    
    def compute_ap_view(self, ap: AccessPoint) -> APClientView:
        """
        Compute QoE for all clients of a specific AP.
        
        Args:
            ap: AccessPoint object
            
        Returns:
            APClientView with per-client results and aggregate statistics
        """
        # Get clients for this AP
        ap_clients = self.get_ap_clients(ap)
        
        if not ap_clients:
            # No clients associated with this AP
            return APClientView(
                ap_id=ap.id,
                client_results=[],
                avg_qoe=0.0,
                min_qoe=0.0,
                max_qoe=0.0,
                num_clients=0
            )
        
        # Compute QoE for each client
        client_results = [self.compute_client_qoe(client) 
                         for client in ap_clients]
        
        # Calculate aggregate statistics
        qoe_scores = [result.qoe_ap for result in client_results]
        avg_qoe = sum(qoe_scores) / len(qoe_scores)
        min_qoe = min(qoe_scores)
        max_qoe = max(qoe_scores)
        
        return APClientView(
            ap_id=ap.id,
            client_results=client_results,
            avg_qoe=avg_qoe,
            min_qoe=min_qoe,
            max_qoe=max_qoe,
            num_clients=len(ap_clients)
        )
    
    def compute_all_views(self) -> Dict[int, APClientView]:
        """
        Compute QoE views for all access points.
        
        Main API method that produces results for all APs.
        
        Returns:
            Dictionary mapping AP ID to APClientView
        """
        results = {}
        
        for ap in self.access_points:
            results[ap.id] = self.compute_ap_view(ap)
        
        return results
    
    def print_ap_view(self, ap_view: APClientView) -> None:
        """
        Pretty-print QoE view for a single AP.
        
        Args:
            ap_view: APClientView object
        """
        print(f"\nAccess Point {ap_view.ap_id}:")
        print(f"  Clients: {ap_view.num_clients}")
        
        if ap_view.num_clients == 0:
            print("  No clients associated")
            return
        
        print(f"  Avg QoE: {ap_view.avg_qoe:.3f}")
        print(f"  Min QoE: {ap_view.min_qoe:.3f}")
        print(f"  Max QoE: {ap_view.max_qoe:.3f}")
        
        print(f"\n  Client Details:")
        for result in ap_view.client_results:
            comp = result.components
            print(f"    Client {result.client_id}:")
            print(f"      QoE: {result.qoe_ap:.3f}")
            print(f"      Components: S={comp.signal_quality:.2f}, "
                  f"T={comp.throughput:.2f}, R={comp.reliability:.2f}, "
                  f"L={comp.latency:.2f}, A={comp.activity:.2f}")
    
    def print_all_views(self, views: Dict[int, APClientView]) -> None:
        """
        Pretty-print QoE views for all APs.
        
        Args:
            views: Dictionary of APClientView from compute_all_views()
        """
        print("\n" + "="*80)
        print("CLIENT VIEW API - QoE RESULTS")
        print("="*80)
        
        for ap_id, ap_view in sorted(views.items()):
            self.print_ap_view(ap_view)
        
        print("\n" + "="*80)
