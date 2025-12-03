import subprocess
import json
from typing import Optional, Dict
from db.transport_stats_db import TransportStatsDB
from db.station_db import StationDB
from model.transport_stats import TransportStats
from logger import Logger


class TransportStatsCollector:
    """
    Collects transport layer statistics by running iperf3 tests.
    
    This class manages iperf3 command execution to gather TCP/UDP
    performance metrics for WiFi stations.
    """
    
    # Default iperf3 parameters
    DEFAULT_DURATION = 10  # seconds
    DEFAULT_PORT = 5201
    DEFAULT_PROTOCOL = "tcp"  # or "udp"
    
    def __init__(self, iperf_server: str = "192.168.1.1", port: int = DEFAULT_PORT):
        """Initialize the transport stats collector.
        
        Args:
            iperf_server: IP address of the iperf3 server
            port: iperf3 server port (default 5201)
        """
        self.transport_db = TransportStatsDB()
        self.stdb = StationDB()
        self.iperf_server = iperf_server
        self.port = port
    
    def run_iperf3(
        self,
        target_ip: str,
        duration: int = DEFAULT_DURATION,
        protocol: str = DEFAULT_PROTOCOL,
        reverse: bool = False
    ) -> Optional[Dict]:
        """Run iperf3 test and return JSON output.
        
        Args:
            target_ip: IP address of the iperf3 server/client
            duration: Test duration in seconds
            protocol: 'tcp' or 'udp'
            reverse: If True, run reverse mode (server sends, client receives)
            
        Returns:
            Parsed JSON output from iperf3, or None on failure
        """
        # Build iperf3 command
        cmd = [
            "iperf3",
            "-c", target_ip,      # Connect to server
            "-p", str(self.port), # Port
            "-t", str(duration),  # Duration
            "-J"                   # JSON output
        ]
        
        if protocol == "udp":
            cmd.append("-u")
        
        if reverse:
            cmd.append("-R")
        
        try:
            Logger.log_info(f"[TransportCollector] Running: {' '.join(cmd)}")
            
            # Run iperf3 command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=duration + 10  # Add 10s buffer for timeout
            )
            
            if result.returncode != 0:
                # Log the actual error from iperf3
                error_msg = result.stderr.strip() if result.stderr else "(no error message)"
                Logger.log_err(
                    f"[TransportCollector] iperf3 failed (exit code {result.returncode})"
                )
                Logger.log_err(f"[TransportCollector] Error: {error_msg}")
                Logger.log_err(f"[TransportCollector] Command: {' '.join(cmd)}")
                return None
            
            # Parse JSON output
            try:
                json_data = json.loads(result.stdout)
                Logger.log_info(
                    f"[TransportCollector] iperf3 test completed successfully"
                )
                return json_data
            except json.JSONDecodeError as e:
                Logger.log_err(f"[TransportCollector] Failed to parse iperf3 JSON: {e}")
                return None
                
        except subprocess.TimeoutExpired:
            Logger.log_err(f"[TransportCollector] iperf3 timeout after {duration + 10}s")
            return None
        except FileNotFoundError:
            Logger.log_err("[TransportCollector] iperf3 not found. Please install iperf3.")
            return None
        except Exception as e:
            Logger.log_err(f"[TransportCollector] Unexpected error running iperf3: {e}")
            return None
    
    def collect_for_station(
        self,
        sta_mac: str,
        sta_ip: str,
        duration: int = DEFAULT_DURATION
    ) -> Optional[TransportStats]:
        """Collect transport stats for a specific station.
        
        Args:
            sta_mac: Station MAC address
            sta_ip: Station IP address (iperf3 target)
            duration: Test duration in seconds
            
        Returns:
            TransportStats object or None on failure
        """
        Logger.log_info(f"[TransportCollector] Collecting stats for {sta_mac} ({sta_ip})")
        
        # Run iperf3 test
        json_data = self.run_iperf3(
            target_ip=sta_ip,
            duration=duration,
            protocol="tcp"
        )
        
        if not json_data:
            Logger.log_err(f"[TransportCollector] No data for {sta_mac}")
            return None
        
        # Parse into TransportStats
        try:
            stats = TransportStats.from_iperf3_json(json_data, sta_mac)
            
            # Store in database
            self.transport_db.add(stats, sta_mac)
            
            Logger.log_info(
                f"[TransportCollector] Stored stats for {sta_mac}: "
                f"rtt={stats.mean_rtt_ms:.1f}ms, "
                f"mbps={stats.mean_mbps:.1f}, "
                f"retrans={stats.retrans_per_sec:.2f}/s"
            )
            
            return stats
            
        except ValueError as e:
            Logger.log_err(f"[TransportCollector] Failed to parse stats for {sta_mac}: {e}")
            return None
    
    def collect_all(self, duration: int = DEFAULT_DURATION):
        """Collect transport stats for all connected stations.
        
        Args:
            duration: Test duration in seconds per station
        """
        stations = self.stdb.all()
        Logger.log_info(f"[TransportCollector] Collecting stats for {len(stations)} stations")
        
        collected = 0
        failed = 0
        
        for station in stations:
            # Skip if not authorized or no IP
            if "AUTHORIZED" not in station.flags or "ASSOC" not in station.flags:
                continue
            
            # Get station IP
            sta_ip = station.ip or station.get_ip()
            
            if not sta_ip:
                Logger.log_warn(f"[TransportCollector] No IP for {station.mac}, skipping")
                failed += 1
                continue
            
            # Collect stats
            stats = self.collect_for_station(
                sta_mac=station.mac,
                sta_ip=sta_ip,
                duration=duration
            )
            
            if stats:
                collected += 1
            else:
                failed += 1
        
        Logger.log_info(
            f"[TransportCollector] Collection complete: "
            f"{collected} succeeded, {failed} failed"
        )
