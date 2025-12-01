import csv
import os
from typing import List, Optional
from datetime import datetime
from datatype import AccessPoint, Client
from metrics import APMetricsManager


class SimulationLogger:
    """Logger for wireless simulation that tracks AP state, client state, and roaming events."""
    
    def __init__(self, log_dir: str = "logs", prefix: str = "sim"):
        """
        Initialize simulation logger.
        
        Args:
            log_dir: Directory to store log files
            prefix: Prefix for log filenames
        """
        self.log_dir = log_dir
        self.prefix = prefix
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Log file paths
        self.ap_log_path = os.path.join(log_dir, f"{prefix}_ap_{self.timestamp}.csv")
        self.client_log_path = os.path.join(log_dir, f"{prefix}_client_{self.timestamp}.csv")
        self.roam_log_path = os.path.join(log_dir, f"{prefix}_roam_{self.timestamp}.csv")
        self.graph_log_path = os.path.join(log_dir, f"{prefix}_graph_{self.timestamp}.csv")
        
        # Initialize log files with headers
        self._init_ap_log()
        self._init_client_log()
        self._init_roam_log()
        self._init_graph_log()
        
        print(f"Logger initialized. Logs will be saved to: {log_dir}/")
    
    def _init_ap_log(self):
        """Initialize AP log file with headers."""
        with open(self.ap_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'ap_id', 'x', 'y', 'tx_power', 'channel', 'bandwidth',
                'noise_floor', 'max_throughput', 'allocated_throughput',
                'duty_cycle', 'num_clients', 'inc_energy_dbm'
            ])
    
    def _init_client_log(self):
        """Initialize client log file with headers."""
        with open(self.client_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'client_id', 'x', 'y', 'associated_ap', 'demand_mbps',
                'sinr_db', 'max_rate_mbps', 'throughput_mbps', 'airtime_fraction',
                'velocity', 'direction'
            ])
    
    def _init_roam_log(self):
        """Initialize roaming log file with headers."""
        with open(self.roam_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'client_id', 'from_ap', 'to_ap', 'client_x', 'client_y'
            ])
    
    def _init_graph_log(self):
        """Initialize interference graph log file with headers."""
        with open(self.graph_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'source_ap', 'dest_ap', 'weight', 'interference_dbm', 'distance'
            ])
    
    def log_ap_state(self, step: int, access_points: List[AccessPoint]):
        """
        Log AP state for current step.
        
        Args:
            step: Current simulation step
            access_points: List of access points
        """
        with open(self.ap_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for ap in access_points:
                duty_cycle = APMetricsManager.ap_duty(ap)
                inc_energy = ap.inc_energy if ap.inc_energy != float('-inf') else None
                
                writer.writerow([
                    step, ap.id, ap.x, ap.y, ap.tx_power, ap.channel, ap.bandwidth,
                    ap.noise_floor, ap.max_throughput, ap.total_allocated_throughput,
                    f"{duty_cycle:.4f}", len(ap.connected_clients), 
                    f"{inc_energy:.2f}" if inc_energy is not None else "N/A"
                ])
    
    def log_client_state(self, step: int, clients: List[Client]):
        """
        Log client state for current step.
        
        Args:
            step: Current simulation step
            clients: List of clients
        """
        with open(self.client_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for client in clients:
                sinr = client.sinr_db if client.sinr_db != float('-inf') else None
                
                writer.writerow([
                    step, client.id, f"{client.x:.2f}", f"{client.y:.2f}",
                    client.associated_ap, f"{client.demand_mbps:.2f}",
                    f"{sinr:.2f}" if sinr is not None else "N/A",
                    f"{client.max_rate_mbps:.2f}", f"{client.throughput_mbps:.2f}",
                    f"{client.airtime_fraction:.4f}", f"{client.velocity:.2f}",
                    f"{client.direction:.2f}"
                ])
    
    def log_roaming_events(self, step: int, clients: List[Client], roam_list: List[int]):
        """
        Log roaming events for current step.
        
        Args:
            step: Current simulation step
            clients: List of clients
            roam_list: List indicating which clients roamed (1 = roamed, 0 = no roam)
        """
        with open(self.roam_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for i, roamed in enumerate(roam_list):
                if roamed and i < len(clients):
                    client = clients[i]
                    writer.writerow([
                        step, client.id, client.last_assoc_ap, client.associated_ap,
                        f"{client.x:.2f}", f"{client.y:.2f}"
                    ])
    
    def log_interference_graph(self, step: int, graph):
        """Log interference graph edges and weights for current step.
        
        Args:
            step: Current simulation step
            graph: NetworkX DiGraph representing interference
        """
        with open(self.graph_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for src, dst, data in graph.edges(data=True):
                weight = data.get('weight', 0.0)
                interference_dbm = data.get('interference_dbm', 0.0)
                distance = data.get('distance', 0.0)
                
                writer.writerow([
                    step, src, dst, f"{weight:.4f}", 
                    f"{interference_dbm:.2f}", f"{distance:.2f}"
                ])
    
    def log_step(self, step: int, access_points: List[AccessPoint], 
                 clients: List[Client], roam_list: List[int], graph=None):
        """
        Log complete state for a simulation step.
        
        Args:
            step: Current simulation step
            access_points: List of access points
            clients: List of clients
            roam_list: List indicating which clients roamed
            graph: Optional interference graph to log
        """
        self.log_ap_state(step, access_points)
        self.log_client_state(step, clients)
        self.log_roaming_events(step, clients, roam_list)
        if graph is not None:
            self.log_interference_graph(step, graph)
    
    def print_summary(self):
        """Print summary of logged data."""
        print("\n=== Logging Summary ===")
        print(f"AP log: {self.ap_log_path}")
        print(f"Client log: {self.client_log_path}")
        print(f"Roaming log: {self.roam_log_path}")
        print(f"Graph log: {self.graph_log_path}")
