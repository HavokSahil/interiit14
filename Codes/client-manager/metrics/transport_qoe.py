from typing import Optional, Dict
from dataclasses import dataclass
from db.transport_stats_db import TransportStatsDB
from db.station_db import StationDB
from model.transport_stats import TransportStats
from logger import Logger


@dataclass
class TransportQoEComponents:
    """Breakdown of Transport QoE score components."""
    latency: float = 0.0        # RTT-based score (0-100)
    reliability: float = 0.0    # Retransmission-based score (0-100)
    throughput: float = 0.0     # Bitrate-based score (0-100)
    
    @property
    def overall(self) -> float:
        """Weighted average: 30% latency + 40% reliability + 30% throughput.
        
        Returns score 0.0 to 1.0 (normalized from 0-100)
        """
        raw_score = (
            0.30 * self.latency +
            0.40 * self.reliability +
            0.30 * self.throughput
        )
        # Normalize from 0-100 to 0.0-1.0
        return raw_score / 100.0
    
    def to_dict(self) -> Dict:
        """Export as dictionary."""
        return {
            "latency": round(self.latency, 2),
            "reliability": round(self.reliability, 2),
            "throughput": round(self.throughput, 2),
            "overall": round(self.overall, 3)
        }


class TransportQoE:
    """
    Compute Transport Layer QoE based on iperf3 metrics.
    
    Scoring is based on the reference implementation:
    - Latency (RTT): Lower is better
    - Reliability (retransmissions): Lower is better
    - Throughput (Mbps): Higher is better
    
    Each component is scored 0-100, then combined with weighted average.
    """
    
    def __init__(self):
        self.transport_db = TransportStatsDB()
        self.stdb = StationDB()
        # Store computed QoE scores (sta_mac -> score 0.0-1.0)
        self._qoe_cache: Dict[str, float] = {}
        # Store component breakdowns (sta_mac -> TransportQoEComponents)
        self._component_cache: Dict[str, TransportQoEComponents] = {}
    
    @staticmethod
    def score_latency(mean_rtt_ms: Optional[float]) -> float:
        """Score RTT latency (0-100 scale).
        
        Lower latency is better:
        - <= 20ms: Excellent (100)
        - <= 40ms: Good (85)
        - <= 80ms: Fair (70)
        - <= 120ms: Poor (50)
        - <= 200ms: Bad (30)
        - > 200ms: Critical (10)
        
        Args:
            mean_rtt_ms: Mean round-trip time in milliseconds
            
        Returns:
            Score from 0 to 100
        """
        if mean_rtt_ms is None:
            return 0
        
        ms = float(mean_rtt_ms)
        if ms <= 20:
            return 100
        elif ms <= 40:
            return 85
        elif ms <= 80:
            return 70
        elif ms <= 120:
            return 50
        elif ms <= 200:
            return 30
        else:
            return 10
    
    @staticmethod
    def score_reliability(retrans_per_sec: Optional[float]) -> float:
        """Score retransmission rate (0-100 scale).
        
        Lower retransmissions are better:
        - <= 0.1: Excellent (100)
        - <= 0.5: Good (85)
        - <= 1: Fair (70)
        - <= 2: Poor (50)
        - <= 5: Bad (30)
        - > 5: Critical (10)
        
        Args:
            retrans_per_sec: Retransmissions per second
            
        Returns:
            Score from 0 to 100
        """
        if retrans_per_sec is None:
            return 0
        
        x = float(retrans_per_sec)
        if x <= 0.1:
            return 100
        elif x <= 0.5:
            return 85
        elif x <= 1:
            return 70
        elif x <= 2:
            return 50
        elif x <= 5:
            return 30
        else:
            return 10
    
    @staticmethod
    def score_throughput(mean_mbps: Optional[float]) -> float:
        """Score throughput (0-100 scale).
        
        Higher throughput is better:
        - >= 200 Mbps: Excellent (100)
        - >= 100 Mbps: Great (90)
        - >= 50 Mbps: Good (80)
        - >= 20 Mbps: Fair (60)
        - >= 5 Mbps: Poor (40)
        - < 5 Mbps: Critical (10)
        
        Args:
            mean_mbps: Mean throughput in Mbps
            
        Returns:
            Score from 0 to 100
        """
        if mean_mbps is None:
            return 0
        
        mbps = float(mean_mbps)
        if mbps >= 200:
            return 100
        elif mbps >= 100:
            return 90
        elif mbps >= 50:
            return 80
        elif mbps >= 20:
            return 60
        elif mbps >= 5:
            return 40
        else:
            return 10
    
    def compute_qoe(self, stats: TransportStats) -> TransportQoEComponents:
        """Compute Transport QoE from transport layer stats.
        
        Args:
            stats: TransportStats object with iperf3 metrics
            
        Returns:
            TransportQoEComponents with latency, reliability, throughput scores
        """
        if not stats:
            return TransportQoEComponents()
        
        components = TransportQoEComponents()
        
        try:
            components.latency = self.score_latency(stats.mean_rtt_ms)
            components.reliability = self.score_reliability(stats.retrans_per_sec)
            components.throughput = self.score_throughput(stats.mean_mbps)
            
        except Exception as e:
            Logger.log_err(f"[TransportQoE] Error computing QoE for {stats.sta_mac}: {e}")
        
        return components
    
    def update(self):
        """Update Transport QoE for all stations with transport stats.
        
        Iterates through all transport stats, computes QoE, and caches results.
        """
        all_stats = self.transport_db.all()
        Logger.log_info(f"[TransportQoE] Updating QoE for {len(all_stats)} stations")
        
        outliers = []
        
        for sta_mac, stats in all_stats.items():
            # Compute QoE components
            components = self.compute_qoe(stats)
            
            # Cache the overall score
            self._qoe_cache[sta_mac] = components.overall
            
            # Cache the components breakdown
            self._component_cache[sta_mac] = components
            
            # Log outliers (very poor or suspiciously perfect scores)
            if components.overall < 0.30 or components.overall > 0.95:
                outliers.append((sta_mac, components))
        
        # Log summary of outliers
        if outliers:
            Logger.log_info(f"[TransportQoE] Found {len(outliers)} outlier stations")
            for mac, comp in outliers[:5]:  # Log top 5 outliers
                Logger.log_info(
                    f"[TransportQoE] {mac}: overall={comp.overall:.3f} "
                    f"(lat={comp.latency:.0f}, rel={comp.reliability:.0f}, "
                    f"tput={comp.throughput:.0f})"
                )
    
    def get_qoe(self, sta_mac: str) -> Optional[float]:
        """Get Transport QoE score for a station.
        
        Args:
            sta_mac: Station MAC address
            
        Returns:
            QoE score 0.0-1.0, or None if not available
        """
        return self._qoe_cache.get(sta_mac)
    
    def get_components(self, sta_mac: str) -> Optional[TransportQoEComponents]:
        """Get detailed Transport QoE breakdown for a station.
        
        Args:
            sta_mac: Station MAC address
            
        Returns:
            TransportQoEComponents or None if not available
        """
        return self._component_cache.get(sta_mac)
    
    def all_qoe(self) -> Dict[str, float]:
        """Get all Transport QoE scores.
        
        Returns:
            Dictionary mapping MAC addresses to QoE scores
        """
        return dict(self._qoe_cache)
    
    def to_dict(self) -> Dict[str, Dict]:
        """Export complete Transport QoE state for dashboards.
        
        Returns:
            Dictionary with QoE scores and components
        """
        return {
            sta_mac: {
                "qoe": self._qoe_cache.get(sta_mac, 0.0),
                "components": self._component_cache.get(sta_mac, TransportQoEComponents()).to_dict()
            }
            for sta_mac in self._qoe_cache.keys()
        }
