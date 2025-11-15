from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from collections import deque

from db.lmrep_db import LinkMeasurementDB
from db.qoe_db import QoEDB
from db.station_db import StationDB
from model.measurement import LinkMeasurement
from model.station import Station
from logger import Logger
import math


@dataclass
class QoEComponents:
    """Breakdown of QoE score components for transparency."""
    connectivity: float = 0.0       # Connection stability (kept for API compatibility)
    signal_quality: float = 0.0     # RF signal strength/quality
    reliability: float = 0.0        # Error rates, retries
    throughput: float = 0.0         # Data rate performance
    latency: float = 0.0           # Responsiveness indicators
    activity: float = 0.0          # Recent activity level
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def overall(self) -> float:
        """Weighted average of all components based on user impact."""
        return (
            0.50 * self.signal_quality +
            0.15 * self.reliability +
            0.15 * self.throughput +
            0.15 * self.latency +
            0.05 * self.activity
        )
    
    def to_dict(self) -> Dict:
        return {
            "connectivity": round(self.connectivity, 3),
            "signal_quality": round(self.signal_quality, 3),
            "reliability": round(self.reliability, 3),
            "throughput": round(self.throughput, 3),
            "latency": round(self.latency, 3),
            "activity": round(self.activity, 3),
            "overall": round(self.overall, 3),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class QoEHistory:
    """Track QoE over time for trend analysis."""
    mac: str
    history: deque = field(default_factory=lambda: deque(maxlen=60))  # Last 60 samples
    alpha: float = 0.3  # Smoothing factor for exponential smoothing
    
    def add(self, score: float):
        self.history.append((datetime.now(), score))
    
    @property
    def average(self) -> float:
        if not self.history:
            return 0.0
        return sum(s for _, s in self.history) / len(self.history)
    
    @property
    def smoothed(self) -> float:
        """Exponential smoothing of the QoE scores."""
        if not self.history:
            return 0.0
        smoothed = self.history[0][1]
        for _, score in list(self.history)[1:]:
            smoothed = self.alpha * score + (1 - self.alpha) * smoothed
        return smoothed
    
    @property
    def trend(self) -> str:
        """Detect if QoE is improving, declining, or stable using smoothed scores."""
        if len(self.history) < 5:
            return "insufficient_data"
        
        last_half = list(self.history)[len(self.history)//2:]
        first_half = list(self.history)[:len(self.history)//2]
        
        smoothed_recent = self._smoothed_for_list(last_half)
        smoothed_older = self._smoothed_for_list(first_half)
        
        diff = smoothed_recent - smoothed_older
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        else:
            return "stable"
    
    def _smoothed_for_list(self, data: list) -> float:
        """Compute exponential smoothing for a given slice of history."""
        if not data:
            return 0.0
        smoothed = data[0][1]
        for _, score in data[1:]:
            smoothed = self.alpha * score + (1 - self.alpha) * smoothed
        return smoothed
    
    @property
    def volatility(self) -> float:
        """Measure how much QoE fluctuates (standard deviation)."""
        if len(self.history) < 2:
            return 0.0
        
        scores = [s for _, s in self.history]
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        return variance ** 0.5


class QoE:
    """
    Compute and maintain QoE (Quality of Experience) scores for WiFi stations.
    
    QoE is a multi-dimensional metric (0.0 to 1.0) that combines:
    - Signal Quality: RSSI, SNR, link margin (30%)
    - Reliability: Packet errors, retries, FCS errors (25%)
    - Throughput: Bitrates, efficiency (20%)
    - Latency: Queue backlog, inactive time (15%)
    - Activity: Recent traffic, engagement (10%)
    
    Higher scores indicate better user experience.
    """
    
    # Signal quality thresholds (dBm for RSSI, dB for SNR)
    RSSI_EXCELLENT = -50
    RSSI_GOOD = -65
    RSSI_POOR = -80
    
    SNR_EXCELLENT = 35
    SNR_GOOD = 25
    SNR_POOR = 10
    
    # Throughput thresholds (Mbps)
    BITRATE_EXCELLENT = 400
    BITRATE_GOOD = 150
    BITRATE_POOR = 20
    
    # Activity thresholds
    INACTIVE_THRESHOLD_MS = 5000  # 5 seconds
    INACTIVE_CRITICAL_MS = 20000  # 20 seconds
    
    # Reliability thresholds
    RETRY_RATE_ACCEPTABLE = 0.05  # 5%
    RETRY_RATE_POOR = 0.20  # 20%
    FCS_RATE_ACCEPTABLE = 0.01  # 1%
    FCS_RATE_POOR = 0.05  # 5%
    
    # Latency thresholds
    BACKLOG_ACCEPTABLE = 10
    BACKLOG_CRITICAL = 100
    
    # Penalty caps
    MAX_RELIABILITY_PENALTY = 0.85
    MAX_LATENCY_PENALTY = 0.70
    MAX_ACTIVITY_PENALTY = 0.80
    
    # Floor values
    SCORE_FLOOR = 0.05
    
    def __init__(self):
        self.stdb = StationDB()
        self.qoedb = QoEDB()
        self.lmdb = LinkMeasurementDB()
        self.history: Dict[str, QoEHistory] = {}
        self.component_cache: Dict[str, QoEComponents] = {}
    
    @staticmethod
    def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Ensure value is within valid range."""
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def _normalize_linear(value: Optional[float], poor: float, good: float, 
                          excellent: float, floor: float = 0.1) -> float:
        """
        Normalize a value using piecewise linear interpolation.
        
        Returns floor if value <= poor, 1.0 if value >= excellent,
        and interpolates linearly between poor-good and good-excellent.
        """
        if value is None:
            return 0.0
        
        if value >= excellent:
            return 1.0
        elif value <= poor:
            return floor
        elif value >= good:
            # Interpolate between good and excellent
            ratio = (value - good) / (excellent - good)
            return 0.7 + 0.3 * ratio
        else:
            # Interpolate between poor and good
            ratio = (value - poor) / (good - poor)
            return floor + (0.7 - floor) * ratio
    
    # ================================================================
    # QoE Component Calculators
    # ================================================================
    
    def _compute_connectivity(self, station: Station) -> float:
        """
        Basic connectivity indicator for API compatibility.
        Returns 1.0 for authorized/associated stations, 0.0 otherwise.
        Not used in overall QoE score calculation.
        """
        if "AUTHORIZED" in station.flags and "ASSOC" in station.flags:
            # Optionally factor in connection duration for more nuance
            if station.connected_sec and station.connected_sec > 60:
                return 1.0
            elif station.connected_sec:
                # New connection, not fully stable yet
                return 0.7 + 0.3 * min(station.connected_sec / 60, 1.0)
            else:
                return 0.8
        return 0.0
    
    def _compute_signal_quality(self, station: Station, lm: Optional[LinkMeasurement]) -> float:
        """
        Signal quality based on RSSI, SNR, and link margin.
        Average of available metrics, weighted equally.
        """
        scores = []
        
        factor = 0.0# RSSI from station
        if station.avg_signal:
            rssi_score = self._normalize_linear(
                station.avg_signal,
                self.RSSI_POOR,
                self.RSSI_GOOD,
                self.RSSI_EXCELLENT,
                floor=self.SCORE_FLOOR
            )
            scores.append(0.5 * rssi_score)
            factor += 0.5
        
        # Link measurement data
        if lm:
            if lm.rssi_dbm:
                rssi_score = self._normalize_linear(
                    lm.rssi_dbm,
                    self.RSSI_POOR,
                    self.RSSI_GOOD,
                    self.RSSI_EXCELLENT,
                    floor=self.SCORE_FLOOR
                )
                scores.append(0.2 * rssi_score)
                factor += 0.2
            
            if lm.rsni is not None:
                # Convert RSNI to SNR (approximate)
                snr_db = lm.rsni / 2.0 - 10
                snr_score = self._normalize_linear(
                    snr_db,
                    self.SNR_POOR,
                    self.SNR_GOOD,
                    self.SNR_EXCELLENT,
                    floor=self.SCORE_FLOOR
                )
                scores.append(0.1 * snr_score)
                factor += 0.1

            
            if lm.link_margin:  # Normalize link margin (0-30 dB range)
                margin_score = self._clamp(lm.link_margin / 30.0)
                scores.append(0.2 * margin_score)
                factor += 0.2
        
        if not scores:
            return 0.0
        
        return sum(scores) / factor
    
    def _compute_throughput(self, station: Station) -> float:
        """
        Throughput score based on actual bitrates and efficiency.
        Uses the best available rate indicator.
        """
        # Prioritize expected throughput (most accurate)
        if station.expected_throughput:
            expected_mbps = station.expected_throughput / 1000.0  # Kbps to Mbps
            return self._normalize_linear(
                expected_mbps,
                self.BITRATE_POOR,
                self.BITRATE_GOOD,
                self.BITRATE_EXCELLENT,
                floor=self.SCORE_FLOOR
            )
        
        # Fall back to actual rates
        actual_rates = []
        if station.tx_bitrate:
            actual_rates.append(station.tx_bitrate)
        if station.rx_bitrate:
            actual_rates.append(station.rx_bitrate)
        
        if actual_rates:
            avg_rate = sum(actual_rates) / len(actual_rates)
            return self._normalize_linear(
                avg_rate,
                self.BITRATE_POOR,
                self.BITRATE_GOOD,
                self.BITRATE_EXCELLENT,
                floor=self.SCORE_FLOOR
            )
        
        # Last resort: check if any supported rates exist
        if station.supported_rates:
            max_supported = max(station.supported_rates)
            # Assume 30% efficiency of max supported rate
            estimated_throughput = max_supported * 0.3
            return self._normalize_linear(
                estimated_throughput,
                self.BITRATE_POOR,
                self.BITRATE_GOOD,
                self.BITRATE_EXCELLENT,
                floor=self.SCORE_FLOOR
            )
        
        return 0.0
    
    def _compute_reliability(self, station: Station) -> float:
        """
        Reliability based on retry rates, FCS errors, and packet failures.
        Uses multiplicative penalties capped at maximum.
        """
        score = 1.0
        total_penalty = 0.0
        
        # Retry rate penalty
        if station.tx_packets and station.tx_retry_count:
            retry_rate = station.tx_retry_count / station.tx_packets
            if retry_rate > self.RETRY_RATE_ACCEPTABLE:
                # Scale penalty: 0 at acceptable, max at poor threshold
                penalty_ratio = min(
                    (retry_rate - self.RETRY_RATE_ACCEPTABLE) / 
                    (self.RETRY_RATE_POOR - self.RETRY_RATE_ACCEPTABLE),
                    1.0
                )
                total_penalty += 0.40 * penalty_ratio
        
        # FCS error penalty
        if station.rx_packets and station.fcs_error_count:
            fcs_rate = station.fcs_error_count / station.rx_packets
            if fcs_rate > self.FCS_RATE_ACCEPTABLE:
                penalty_ratio = min(
                    (fcs_rate - self.FCS_RATE_ACCEPTABLE) / 
                    (self.FCS_RATE_POOR - self.FCS_RATE_ACCEPTABLE),
                    1.0
                )
                total_penalty += 0.25 * penalty_ratio
        
        # Failed retry penalty
        if station.tx_packets and station.tx_retry_failed:
            failure_rate = station.tx_retry_failed / station.tx_packets
            if failure_rate > 0.01:  # More than 1% is concerning
                total_penalty += min(failure_rate * 10, 0.15)
        
        # Dropped packet penalty
        if station.rx_packets and station.rx_drop_misc:
            drop_rate = station.rx_drop_misc / station.rx_packets
            if drop_rate > 0.01:
                total_penalty += min(drop_rate * 10, 0.15)
        
        # Cap total penalty
        total_penalty = min(total_penalty, self.MAX_RELIABILITY_PENALTY)
        score = 1.0 - total_penalty
        
        return self._clamp(score, self.SCORE_FLOOR)
    
    def _compute_latency(self, station: Station) -> float:
        """
        Latency indicator based on backlog packets and inactive time.
        Lower is better for responsiveness.
        """
        score = 1.0
        total_penalty = 0.0
        
        # Backlog penalty (queued packets indicate congestion)
        if station.backlog_packets:
            if station.backlog_packets >= self.BACKLOG_CRITICAL:
                total_penalty += 0.50
            elif station.backlog_packets > self.BACKLOG_ACCEPTABLE:
                backlog_ratio = (station.backlog_packets - self.BACKLOG_ACCEPTABLE) / \
                               (self.BACKLOG_CRITICAL - self.BACKLOG_ACCEPTABLE)
                total_penalty += 0.50 * backlog_ratio
        
        # Inactive time penalty (but only if extreme)
        if station.inactive_msec:
            if station.inactive_msec > self.INACTIVE_CRITICAL_MS:
                # Station is nearly idle - major latency concern
                total_penalty += 0.30
            elif station.inactive_msec > self.INACTIVE_THRESHOLD_MS:
                inactive_ratio = (station.inactive_msec - self.INACTIVE_THRESHOLD_MS) / \
                                (self.INACTIVE_CRITICAL_MS - self.INACTIVE_THRESHOLD_MS)
                total_penalty += 0.30 * inactive_ratio
        
        # Cap total penalty
        total_penalty = min(total_penalty, self.MAX_LATENCY_PENALTY)
        score = 1.0 - total_penalty
        
        return self._clamp(score, self.SCORE_FLOOR)
    
    def _compute_activity(self, station: Station) -> float:
        """
        Activity level based on recent traffic and engagement.
        Inactive stations should have lower QoE as they're not actively used.
        """
        score = 0.5  # Baseline for connected but idle
        
        # Penalty for prolonged inactivity
        if station.inactive_msec:
            if station.inactive_msec > self.INACTIVE_CRITICAL_MS:
                score = self.SCORE_FLOOR  # Nearly dead connection
            elif station.inactive_msec > self.INACTIVE_THRESHOLD_MS:
                # Gradually reduce score
                inactive_ratio = (station.inactive_msec - self.INACTIVE_THRESHOLD_MS) / \
                                (self.INACTIVE_CRITICAL_MS - self.INACTIVE_THRESHOLD_MS)
                penalty = 0.40 * inactive_ratio
                score -= penalty
        
        # Bonus for active traffic
        total_packets = (station.tx_packets or 0) + (station.rx_packets or 0)
        if total_packets > 100:
            # Scale bonus: 100 packets = small boost, 10000+ packets = max boost
            traffic_bonus = min(math.log10(total_packets) / 4.0, 0.50)
            score += traffic_bonus
        
        return self._clamp(score, self.SCORE_FLOOR)
    
    # ================================================================
    # Main QoE Computation
    # ================================================================
    
    def compute_qoe(self, station: Station, lm: Optional[LinkMeasurement] = None) -> QoEComponents:
        """
        Compute comprehensive QoE for a station.
        Returns QoEComponents with all metrics and overall score.
        """
        if not station or not station.mac:
            Logger.log_err("[QoE] Invalid station object provided")
            return QoEComponents()
        
        components = QoEComponents()
        
        try:
            components.connectivity = self._compute_connectivity(station)
            components.signal_quality = self._compute_signal_quality(station, lm)
            components.reliability = self._compute_reliability(station)
            components.throughput = self._compute_throughput(station)
            components.latency = self._compute_latency(station)
            components.activity = self._compute_activity(station)
            
            # Validate all components are in valid range
            for field_name in ['connectivity', 'signal_quality', 'reliability', 'throughput', 'latency', 'activity']:
                value = getattr(components, field_name)
                if not 0.0 <= value <= 1.0:
                    Logger.log_warn(f"[QoE] {station.mac}: {field_name} out of range: {value:.3f}")
                    setattr(components, field_name, self._clamp(value))
            
        except Exception as e:
            Logger.log_err(f"[QoE] Error computing QoE for {station.mac}: {e}")
        
        return components
    
    # ================================================================
    # Update and Tracking
    # ================================================================
    
    def update(self):
        """
        Iterate through all stations and update QoE database.
        Only logs outliers to reduce noise.
        """
        stations = self.stdb.all()
        Logger.log_info(f"[QoE] Updating QoE for {len(stations)} stations")
        
        outliers = []
        
        for station in stations:
            mac = station.mac
            
            # Skip unauthorized/unassociated stations
            if "AUTHORIZED" not in station.flags or "ASSOC" not in station.flags:
                continue
            
            # Get link measurement if available
            lm = self.lmdb.get(mac)
            
            # Compute QoE components
            components = self.compute_qoe(station, lm)
            
            # Store in database
            self.qoedb.set(mac, components.overall)
            
            # Cache components for detailed queries
            self.component_cache[mac] = components
            
            # Track history
            if mac not in self.history:
                self.history[mac] = QoEHistory(mac)
            self.history[mac].add(components.overall)
            
            # Log outliers (very poor or suspiciously perfect scores)
            if components.overall < 0.30 or components.overall > 0.95:
                outliers.append((mac, components))
        
        # Log summary of outliers
        if outliers:
            Logger.log_info(f"[QoE] Found {len(outliers)} outlier stations")
            for mac, comp in outliers[:5]:  # Log top 5 outliers
                Logger.log_info(
                    f"[QoE] {mac}: overall={comp.overall:.3f} "
                    f"(conn={comp.connectivity:.2f}, sig={comp.signal_quality:.2f}, "
                    f"rel={comp.reliability:.2f}, tput={comp.throughput:.2f}, "
                    f"lat={comp.latency:.2f}, act={comp.activity:.2f})"
                )
    
    # ================================================================
    # Query and Analysis
    # ================================================================
    
    def get_components(self, mac: str) -> Optional[QoEComponents]:
        """Get detailed QoE breakdown for a station."""
        return self.component_cache.get(mac)
    
    def get_history(self, mac: str) -> Optional[QoEHistory]:
        """Get QoE history for trend analysis."""
        return self.history.get(mac)
    
    def get_worst_stations(self, n: int = 5) -> List[tuple[str, float]]:
        """Get the N stations with lowest QoE."""
        scores = [(mac, self.qoedb.get(mac) or 0.0) for mac in self.stdb.list()]
        return sorted(scores, key=lambda x: x[1])[:n]
    
    def get_best_stations(self, n: int = 5) -> List[tuple[str, float]]:
        """Get the N stations with highest QoE."""
        scores = [(mac, self.qoedb.get(mac) or 0.0) for mac in self.stdb.list()]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:n]
    
    def get_statistics(self) -> Dict:
        """Get overall QoE statistics across all stations."""
        scores = [self.qoedb.get(mac) or 0.0 for mac in self.stdb.list()]
        
        if not scores:
            return {
                "count": 0,
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std_dev": 0.0
            }
        
        avg = sum(scores) / len(scores)
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
        
        return {
            "count": len(scores),
            "average": round(avg, 3),
            "min": round(min(scores), 3),
            "max": round(max(scores), 3),
            "std_dev": round(std_dev, 3)
        }
