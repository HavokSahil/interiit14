"""
QoE Monitor - Real-time Quality of Experience monitoring for RRM simulation.

This module provides QoE computation and tracking based on role-based weights
from the Policy Engine and SLO Catalog.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import statistics

from datatype import Client, AccessPoint
from policy_engine import PolicyEngine
from slo_catalog import SLOCatalog


@dataclass
class QoEComponents:
    """Individual QoE component scores (normalized 0-1)"""
    signal_quality: float = 0.0
    throughput: float = 0.0
    reliability: float = 0.0
    latency: float = 0.0
    activity: float = 0.0


@dataclass
class QoEStats:
    """Aggregate QoE statistics"""
    mean: float
    median: float
    p50: float
    p95: float
    p99: float
    min_val: float
    max_val: float
    std_dev: float
    count: int


@dataclass
class QoESnapshot:
    """QoE snapshot at a specific time"""
    step: int
    timestamp: datetime
    network_qoe: float
    per_ap_qoe: Dict[int, float] = field(default_factory=dict)
    per_client_qoe: Dict[int, float] = field(default_factory=dict)
    per_role_qoe: Dict[str, float] = field(default_factory=dict)


class QoEMonitor:
    """
    Real-time QoE monitoring and tracking.
    
    Computes weighted QoE for clients based on their assigned roles,
    tracks network-wide aggregates, and maintains time series data.
    """
    
    def __init__(self, 
                 slo_catalog: SLOCatalog,
                 policy_engine: PolicyEngine,
                 history_size: int = 1000):
        """
        Initialize QoE Monitor.
        
        Args:
            slo_catalog: SLO catalog for normalization parameters
            policy_engine: Policy engine for role-based weights
            history_size: Number of snapshots to keep in history
        """
        self.slo_catalog = slo_catalog
        self.policy_engine = policy_engine
        self.history_size = history_size
        
        # Time series storage
        self.history: List[QoESnapshot] = []
        
        # Get global normalizers
        normalizers = self.slo_catalog.get_global_normalizers()
        self.rssi_min = normalizers.get('RSSI_min', -90.0)
        self.rssi_max = normalizers.get('RSSI_max', -30.0)
        self.throughput_max = normalizers.get('max_throughput_mbps', 100.0)
        self.latency_max = normalizers.get('max_latency_ms', 100.0)
        
        # Current step tracking
        self.current_step = 0
    
    def normalize_rssi(self, rssi: float) -> float:
        """Normalize RSSI to 0-1 range"""
        if rssi == float('-inf') or rssi < self.rssi_min:
            return 0.0
        return min(1.0, max(0.0, (rssi - self.rssi_min) / (self.rssi_max - self.rssi_min)))
    
    def normalize_throughput(self, throughput: float) -> float:
        """Normalize throughput to 0-1 range"""
        return min(1.0, max(0.0, throughput / self.throughput_max))
    
    def normalize_latency(self, latency: float) -> float:
        """Normalize latency to 0-1 range (inverted - lower is better)"""
        if latency <= 0:
            return 1.0
        # Invert: high latency = low score
        return max(0.0, 1.0 - min(1.0, latency / self.latency_max))
    
    def compute_qoe_components(self, client: Client) -> QoEComponents:
        """
        Compute individual QoE components for a client.
        
        Args:
            client: Client object
            
        Returns:
            QoEComponents with normalized scores
        """
        # Signal quality (based on RSSI)
        signal = self.normalize_rssi(client.rssi_dbm)
        
        # Throughput
        throughput = self.normalize_throughput(client.throughput_mbps)
       
        # Reliability (based on retry rate - inverted)
        reliability = max(0.0, 1.0 - (client.retry_rate / 100.0))
        
        # Latency (if available, otherwise use default)
        latency_ms = getattr(client, 'latency_ms', 20.0)  # Default 20ms if not available
        latency = self.normalize_latency(latency_ms)
        
        # Activity (airtime fraction)
        activity = client.airtime_fraction
        
        return QoEComponents(
            signal_quality=signal,
            throughput=throughput,
            reliability=reliability,
            latency=latency,
            activity=activity
        )
    
    def compute_client_qoe(self, client: Client) -> Tuple[float, QoEComponents]:
        """
        Compute weighted QoE for a client based on their role.
        
        Args:
            client: Client object
            
        Returns:
            (qoe_score, components)
        """
        # Get components
        components = self.compute_qoe_components(client)
        
        # Get role-based weights
        weights = self.policy_engine.get_client_qos_weights(client.id)
        
        # Compute weighted sum
        qoe = (weights.get('ws', 0.2) * components.signal_quality +
               weights.get('wt', 0.3) * components.throughput +
               weights.get('wr', 0.2) * components.reliability +
               weights.get('wl', 0.2) * components.latency +
               weights.get('wa', 0.1) * components.activity)
        
        return qoe, components
    
    def update(self, step: int, clients: List[Client], access_points: List[AccessPoint]):
        """
        Update QoE metrics for current step.
        
        Args:
            step: Current simulation step
            clients: List of all clients
            access_points: List of all APs
        """
        self.current_step = step
        
        # Compute per-client QoE
        per_client_qoe = {}
        per_role_qoe_values = {}
        
        for client in clients:
            qoe, _ = self.compute_client_qoe(client)
            per_client_qoe[client.id] = qoe
            
            # Track by role
            role = self.policy_engine.get_client_role(client.id)
            if role not in per_role_qoe_values:
                per_role_qoe_values[role] = []
            per_role_qoe_values[role].append(qoe)
        
        # Aggregate per-role
        per_role_qoe = {
            role: statistics.mean(values) if values else 0.0
            for role, values in per_role_qoe_values.items()
        }
        
        # Compute per-AP QoE (average of associated clients)
        per_ap_qoe = {}
        for ap in access_points:
            ap_clients = [c for c in clients if c.associated_ap == ap.id]
            if ap_clients:
                ap_qoe_values = [per_client_qoe[c.id] for c in ap_clients]
                per_ap_qoe[ap.id] = statistics.mean(ap_qoe_values)
            else:
                per_ap_qoe[ap.id] = 0.0
        
        # Network-wide QoE
        if per_client_qoe:
            network_qoe = statistics.mean(per_client_qoe.values())
        else:
            network_qoe = 0.0
        
        # Create snapshot
        snapshot = QoESnapshot(
            step=step,
            timestamp=datetime.now(),
            network_qoe=network_qoe,
            per_ap_qoe=per_ap_qoe,
            per_client_qoe=per_client_qoe,
            per_role_qoe=per_role_qoe
        )
        
        # Add to history (maintain size limit)
        self.history.append(snapshot)
        if len(self.history) > self.history_size:
            self.history.pop(0)
    
    def get_network_qoe_stats(self, window: Optional[int] = None) -> QoEStats:
        """
        Get network-wide QoE statistics.
        
        Args:
            window: Number of recent steps to consider (None = all)
            
        Returns:
            QoEStats object
        """
        if not self.history:
            return QoEStats(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Get relevant history
        history = self.history[-window:] if window else self.history
        values = [s.network_qoe for s in history]
        
        if not values:
            return QoEStats(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Sort for percentiles
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return QoEStats(
            mean=statistics.mean(values),
            median=statistics.median(values),
            p50=sorted_values[int(n * 0.50)] if n > 0 else 0,
            p95=sorted_values[int(n * 0.95)] if n > 0 else 0,
            p99=sorted_values[int(n * 0.99)] if n > 0 else 0,
            min_val=min(values),
            max_val=max(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0,
            count=n
        )
    
    def get_ap_qoe_stats(self, ap_id: int, window: Optional[int] = None) -> QoEStats:
        """Get QoE statistics for a specific AP"""
        if not self.history:
            return QoEStats(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        history = self.history[-window:] if window else self.history
        values = [s.per_ap_qoe.get(ap_id, 0.0) for s in history]
        values = [v for v in values if v > 0]  # Filter out zeros
        
        if not values:
            return QoEStats(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return QoEStats(
            mean=statistics.mean(values),
            median=statistics.median(values),
            p50=sorted_values[int(n * 0.50)] if n > 0 else 0,
            p95=sorted_values[int(n * 0.95)] if n > 0 else 0,
            p99=sorted_values[int(n * 0.99)] if n > 0 else 0,
            min_val=min(values),
            max_val=max(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0,
            count=n
        )
    
    def get_time_series(self, window: Optional[int] = None) -> List[QoESnapshot]:
        """
        Get time series of QoE snapshots.
        
        Args:
            window: Number of recent steps to return (None = all)
            
        Returns:
            List of QoESnapshot objects
        """
        if window is None:
            return self.history.copy()
        return self.history[-window:].copy()
    
    def export_metrics(self, filepath: str):
        """
        Export QoE metrics to CSV file.
        
        Args:
            filepath: Output file path
        """
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['step', 'timestamp', 'network_qoe', 'num_clients', 'num_aps'])
            
            # Data
            for snapshot in self.history:
                writer.writerow([
                    snapshot.step,
                    snapshot.timestamp.isoformat(),
                    snapshot.network_qoe,
                    len(snapshot.per_client_qoe),
                    len(snapshot.per_ap_qoe)
                ])
        
        print(f"QoE metrics exported to {filepath}")
    
    def print_current_status(self):
        """Print current QoE status"""
        if not self.history:
            print("No QoE data available")
            return
        
        latest = self.history[-1]
        stats = self.get_network_qoe_stats(window=100)
        
        print("\n" + "="*60)
        print("QoE MONITOR STATUS")
        print("="*60)
        print(f"Step: {latest.step}")
        print(f"Network QoE: {latest.network_qoe:.3f}")
        print(f"\nStatistics (last 100 steps):")
        print(f"  Mean:   {stats.mean:.3f}")
        print(f"  Median: {stats.median:.3f}")
        print(f"  P95:    {stats.p95:.3f}")
        print(f"  StdDev: {stats.std_dev:.3f}")
        print(f"\nPer-Role QoE:")
        for role, qoe in latest.per_role_qoe.items():
            print(f"  {role}: {qoe:.3f}")
        print()
