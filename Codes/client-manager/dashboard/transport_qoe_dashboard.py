from typing import Optional, TextIO
from db.transport_stats_db import TransportStatsDB
from metrics.transport_qoe import TransportQoE
from logger import Logger
import json
import socket


def write_stream(pipe_or_sock, text: str):
    """Write text to either a pipe (file) or a socket transparently."""
    if not pipe_or_sock:
        return
    # Socket case
    if isinstance(pipe_or_sock, socket.socket):
        try:
            pipe_or_sock.sendall((text + "\n").encode())
        except Exception:
            pass
        return
    # FIFO/file/stdout
    try:
        pipe_or_sock.write(text + "\n")
        pipe_or_sock.flush()
    except Exception:
        pass


class TransportQoEDashboard:
    """Dashboard for displaying Transport Layer QoE metrics.
    
    Shows per-station transport statistics including:
    - Transport QoE score
    - Component scores (latency, reliability, throughput)
    - Raw metrics (RTT, retransmissions, bitrate)
    """
    
    def __init__(self, transport_db: TransportStatsDB, transport_qoe: Optional[TransportQoE] = None):
        """Initialize the Transport QoE dashboard.
        
        Args:
            transport_db: TransportStatsDB instance
            transport_qoe: Optional TransportQoE instance for rich component data
        """
        self.db = transport_db
        self.qoe_engine = transport_qoe
    
    def show(self, pipe: Optional[TextIO] = None, replace: bool = False, mode: str = "full"):
        """Display the Transport QoE dashboard.
        
        Args:
            pipe: Output stream (None for stdout)
            replace: If True, clear screen before displaying
            mode: 'full' for detailed view, 'summary' for compact view
        """
        try:
            if mode == "full":
                output = self._render_full()
            else:
                output = self._render_summary()
            
            # Clear screen if requested
            if replace:
                output = "\033[H\033[J" + output
            
            if pipe:
                write_stream(pipe, output)
            else:
                print(output)
                
        except Exception as e:
            Logger.log_err(f"[TransportQoEDashboard] Error rendering dashboard: {e}")
    
    def _render_full(self) -> str:
        """Render full detailed dashboard."""
        lines = []
        lines.append("=" * 80)
        lines.append(" TRANSPORT LAYER QoE DASHBOARD ".center(80))
        lines.append("=" * 80)
        lines.append("")
        
        # Get all transport stats
        all_stats = self.db.all()
        
        if not all_stats:
            lines.append("No transport layer statistics available.".center(80))
            lines.append("=" * 80)
            return "\n".join(lines)
        
        # Header
        lines.append(f"{'MAC Address':<20} {'QoE':<8} {'RTT(ms)':<10} {'Retrans/s':<12} {'Mbps':<10} {'Status':<10}")
        lines.append("-" * 80)
        
        # Station rows
        for sta_mac, stats in sorted(all_stats.items()):
            # Get QoE score if available
            qoe_score = 0.0
            if self.qoe_engine:
                qoe_val = self.qoe_engine.get_qoe(sta_mac)
                qoe_score = qoe_val if qoe_val is not None else 0.0
            
            # Format metrics
            rtt_str = f"{stats.mean_rtt_ms:.1f}" if stats.mean_rtt_ms else "N/A"
            retrans_str = f"{stats.retrans_per_sec:.2f}" if stats.retrans_per_sec is not None else "N/A"
            mbps_str = f"{stats.mean_mbps:.1f}" if stats.mean_mbps else "N/A"
            
            # Status indicator
            if qoe_score >= 0.8:
                status = "Excellent"
            elif qoe_score >= 0.6:
                status = "Good"
            elif qoe_score >= 0.4:
                status = "Fair"
            elif qoe_score >= 0.2:
                status = "Poor"
            else:
                status = "Critical"
            
            lines.append(
                f"{sta_mac:<20} {qoe_score:>6.3f}  {rtt_str:<10} {retrans_str:<12} {mbps_str:<10} {status:<10}"
            )
        
        lines.append("-" * 80)
        
        # Component breakdown for QoE engine
        if self.qoe_engine:
            lines.append("")
            lines.append(" COMPONENT SCORES ".center(80, "-"))
            lines.append(f"{'MAC Address':<20} {'Latency':<10} {'Reliability':<12} {'Throughput':<12}")
            lines.append("-" * 80)
            
            for sta_mac in sorted(all_stats.keys()):
                components = self.qoe_engine.get_components(sta_mac)
                if components:
                    lines.append(
                        f"{sta_mac:<20} {components.latency:>8.1f}  {components.reliability:>10.1f}  {components.throughput:>10.1f}"
                    )
        
        lines.append("=" * 80)
        lines.append(f"Total stations: {len(all_stats)}")
        
        return "\n".join(lines)
    
    def _render_summary(self) -> str:
        """Render compact summary dashboard."""
        lines = []
        lines.append("=== Transport QoE Summary ===")
        
        all_stats = self.db.all()
        
        if not all_stats:
            lines.append("No data available")
            return "\n".join(lines)
        
        # Calculate averages
        total_qoe = 0.0
        count = 0
        
        if self.qoe_engine:
            for sta_mac in all_stats.keys():
                qoe = self.qoe_engine.get_qoe(sta_mac)
                if qoe is not None:
                    total_qoe += qoe
                    count += 1
        
        avg_qoe = total_qoe / count if count > 0 else 0.0
        
        lines.append(f"Stations: {len(all_stats)}")
        lines.append(f"Average QoE: {avg_qoe:.3f}")
        lines.append("=" * 30)
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Export dashboard data as JSON.
        
        Returns:
            JSON string with transport stats and QoE data
        """
        data = {
            "transport_stats": {},
            "qoe_scores": {}
        }
        
        all_stats = self.db.all()
        
        for sta_mac, stats in all_stats.items():
            data["transport_stats"][sta_mac] = stats.to_dict()
            
            if self.qoe_engine:
                qoe = self.qoe_engine.get_qoe(sta_mac)
                components = self.qoe_engine.get_components(sta_mac)
                
                data["qoe_scores"][sta_mac] = {
                    "overall": qoe if qoe is not None else 0.0,
                    "components": components.to_dict() if components else {}
                }
        
        return json.dumps(data, indent=2)
