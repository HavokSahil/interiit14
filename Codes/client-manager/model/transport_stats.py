from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
import json


@dataclass
class TransportStats:
    """Transport layer statistics from iperf3 measurements."""
    
    sta_mac: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # RTT metrics
    mean_rtt_ms: Optional[float] = None
    min_rtt_ms: Optional[float] = None
    max_rtt_ms: Optional[float] = None
    
    # Retransmission metrics
    total_retransmits: Optional[int] = None
    duration_sec: Optional[float] = None
    retrans_per_sec: Optional[float] = None
    
    # Throughput metrics
    mean_mbps: Optional[float] = None
    bits_per_second: Optional[float] = None
    
    # Additional iperf3 metrics
    max_snd_cwnd: Optional[int] = None
    max_rtt_ms: Optional[float] = None
    min_rtt_ms: Optional[float] = None
    bytes_sent: Optional[int] = None
    
    @staticmethod
    def from_iperf3_json(json_data: Dict[str, Any], sta_mac: str) -> "TransportStats":
        """
        Parse iperf3 JSON output and create a TransportStats object.
        
        Args:
            json_data: Parsed iperf3 JSON output (from json.load or json.loads)
            sta_mac: Station MAC address for this measurement
            
        Returns:
            TransportStats object with parsed metrics
        """
        try:
            # Extract sender section from iperf3 output
            sender = json_data.get("end", {}).get("streams", [{}])[0].get("sender", {})
            
            # Extract basic metrics
            mean_rtt_us = sender.get("mean_rtt", 0)
            mean_rtt_ms = mean_rtt_us / 1000.0 if mean_rtt_us else None
            
            total_retrans = sender.get("retransmits", 0)
            duration = sender.get("seconds", 1)
            retrans_per_sec = total_retrans / duration if duration > 0 else 0
            
            bits_per_second = sender.get("bits_per_second", 0)
            mean_mbps = bits_per_second / 1e6 if bits_per_second else None
            
            # Extract additional metrics if available
            max_snd_cwnd = sender.get("max_snd_cwnd")
            max_rtt_us = sender.get("max_rtt")
            min_rtt_us = sender.get("min_rtt")
            bytes_sent = sender.get("bytes")
            
            return TransportStats(
                sta_mac=sta_mac,
                timestamp=datetime.now(),
                mean_rtt_ms=mean_rtt_ms,
                min_rtt_ms=min_rtt_us / 1000.0 if min_rtt_us else None,
                max_rtt_ms=max_rtt_us / 1000.0 if max_rtt_us else None,
                total_retransmits=total_retrans,
                duration_sec=duration,
                retrans_per_sec=retrans_per_sec,
                mean_mbps=mean_mbps,
                bits_per_second=bits_per_second,
                max_snd_cwnd=max_snd_cwnd,
                bytes_sent=bytes_sent
            )
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Invalid iperf3 JSON format: {e}")
    
    @staticmethod
    def from_iperf3_file(file_path: str, sta_mac: str) -> "TransportStats":
        """
        Load iperf3 JSON from a file and parse it.
        
        Args:
            file_path: Path to iperf3 JSON output file
            sta_mac: Station MAC address for this measurement
            
        Returns:
            TransportStats object with parsed metrics
        """
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        return TransportStats.from_iperf3_json(json_data, sta_mac)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert TransportStats to a dictionary for serialization.
        
        Returns:
            Dictionary representation of transport stats
        """
        return {
            "sta_mac": self.sta_mac,
            "timestamp": self.timestamp.isoformat(),
            "mean_rtt_ms": self.mean_rtt_ms,
            "min_rtt_ms": self.min_rtt_ms,
            "max_rtt_ms": self.max_rtt_ms,
            "total_retransmits": self.total_retransmits,
            "duration_sec": self.duration_sec,
            "retrans_per_sec": self.retrans_per_sec,
            "mean_mbps": self.mean_mbps,
            "bits_per_second": self.bits_per_second,
            "max_snd_cwnd": self.max_snd_cwnd,
            "bytes_sent": self.bytes_sent
        }
    
    def __repr__(self) -> str:
        return (
            f"<TransportStats mac={self.sta_mac} "
            f"rtt={self.mean_rtt_ms:.1f}ms "
            f"mbps={self.mean_mbps:.1f} "
            f"retrans={self.retrans_per_sec:.2f}/s>"
        )
