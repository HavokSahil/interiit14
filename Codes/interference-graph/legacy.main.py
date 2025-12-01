import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
from scipy.spatial import Voronoi
import numpy as np
from sim import *

@dataclass
class AccessPoint:
    """Represents a wireless access point."""
    id: int
    x: float
    y: float
    tx_power: float  # dBm
    channel: int = 6  # WiFi channel (1-11 for 2.4GHz)
    bandwidth: float = 20.0  # MHz
    max_throughput: float = 100.0  # Maximum throughput in Mbps
    total_allocated_throughput: float = 0.0  # Sum of allocated throughput
    airtime_utilization: float = 0.0  # Percentage of airtime used (0-1)
    is_saturated: bool = False  # Whether AP is saturated
    connected_clients: List[int] = field(default_factory=list)


@dataclass
class Client:
    """Represents a wireless client."""
    id: int
    x: float
    y: float
    load: float  # Client's traffic demand in Mbps
    associated_ap: Optional[int] = None
    velocity: float = 1.0  # Speed in units per step
    direction: float = 0.0  # Direction in radians
    sinr_db: float = 0.0  # Signal-to-Interference-plus-Noise Ratio in dB
    max_rate_mbps: float = 0.0  # Maximum achievable rate based on SINR
    throughput_mbps: float = 0.0  # Actual allocated throughput in Mbps
    airtime_fraction: float = 0.0  # Fraction of airtime allocated to this client
    is_satisfied: bool = True  # Whether demand is fully met


class Environment:
    """Bounded environment for simulation."""
    
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
    
    def is_within_bounds(self, x: float, y: float) -> bool:
        """Check if position is within bounds."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max
    
    def clip_to_bounds(self, x: float, y: float) -> Tuple[float, float]:
        """Clip position to bounds."""
        x = max(self.x_min, min(x, self.x_max))
        y = max(self.y_min, min(y, self.y_max))
        return x, y

class ClientAssociationManager:
    """Manages client-AP associations with SINR and airtime-fair throughput allocation."""
    
    def __init__(self, propagation_model: PropagationModel, noise_floor_dbm: float = -95.0):
        self.model = propagation_model
        self.noise_floor = noise_floor_dbm
    
    def _compute_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Euclidean distance."""
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def voronoi_association(self, clients: List[Client], 
                           access_points: List[AccessPoint]) -> None:
        """Initial association based on Voronoi diagram (nearest AP)."""
        for client in clients:
            min_dist = float('inf')
            nearest_ap = None
            
            for ap in access_points:
                dist = self._compute_distance(client.x, client.y, ap.x, ap.y)
                if dist < min_dist:
                    min_dist = dist
                    nearest_ap = ap.id
            
            client.associated_ap = nearest_ap
    
    def signal_strength_association(self, clients: List[Client], 
                                   access_points: List[AccessPoint]) -> None:
        """Associate clients to AP with best signal strength."""
        ap_dict = {ap.id: ap for ap in access_points}
        
        for client in clients:
            best_power = float('-inf')
            best_ap = None
            
            for ap in access_points:
                dist = self._compute_distance(client.x, client.y, ap.x, ap.y)
                rx_power = self.model.compute_received_power(ap.tx_power, dist)
                
                if rx_power > best_power:
                    best_power = rx_power
                    best_ap = ap.id
            
            client.associated_ap = best_ap
    
    def compute_sinr_and_throughput(self, clients: List[Client], 
                                    access_points: List[AccessPoint]) -> None:
        """
        Compute SINR and throughput for each client with airtime fairness.
        Each client gets equal airtime, and throughput is allocated based on:
        1. Client's max rate (from SINR)
        2. Client's demand
        3. AP's max throughput capacity
        4. Fair airtime sharing among all clients on the AP
        """
        ap_dict = {ap.id: ap for ap in access_points}
        
        # First pass: compute SINR and max rate for all clients
        for client in clients:
            if client.associated_ap is None:
                client.sinr_db = float('-inf')
                client.max_rate_mbps = 0.0
                client.throughput_mbps = 0.0
                client.airtime_fraction = 0.0
                client.is_satisfied = False
                continue
            
            # Get associated AP
            serving_ap = ap_dict[client.associated_ap]
            
            # Compute signal power from serving AP
            dist_to_ap = self._compute_distance(client.x, client.y, serving_ap.x, serving_ap.y)
            signal_power_dbm = self.model.compute_received_power(serving_ap.tx_power, dist_to_ap)
            signal_power_mw = self._dbm_to_mw(signal_power_dbm)
            
            # Compute interference from all other APs
            interference_power_mw = 0.0
            for ap in access_points:
                if ap.id == client.associated_ap:
                    continue
                
                # Only consider interference from APs on same or overlapping channels
                channel_overlap = self._compute_channel_overlap(serving_ap, ap)
                if channel_overlap > 0.01:  # Threshold for significant overlap
                    dist_to_interferer = self._compute_distance(client.x, client.y, ap.x, ap.y)
                    interferer_power_dbm = self.model.compute_received_power(ap.tx_power, dist_to_interferer)
                    interferer_power_mw = self._dbm_to_mw(interferer_power_dbm)
                    
                    # Weight interference by channel overlap
                    interference_power_mw += interferer_power_mw * channel_overlap
            
            # Add noise floor
            noise_power_mw = self._dbm_to_mw(self.noise_floor)
            
            # Compute SINR
            sinr_linear = signal_power_mw / (interference_power_mw + noise_power_mw)
            client.sinr_db = 10 * math.log10(sinr_linear) if sinr_linear > 0 else float('-inf')
            
            # Compute maximum achievable rate (Shannon capacity)
            if client.sinr_db == float('-inf'):
                client.max_rate_mbps = 0.0
            else:
                sinr_linear = 10 ** (client.sinr_db / 10.0)
                # Shannon capacity with efficiency factor
                efficiency = 0.7
                client.max_rate_mbps = efficiency * serving_ap.bandwidth * math.log2(1 + sinr_linear)
        
        # Second pass: allocate throughput with airtime fairness
        # Group clients by AP
        ap_clients = {}
        for client in clients:
            if client.associated_ap is not None:
                if client.associated_ap not in ap_clients:
                    ap_clients[client.associated_ap] = []
                ap_clients[client.associated_ap].append(client)
        
        # Allocate throughput for each AP's clients
        for ap_id, ap_client_list in ap_clients.items():
            serving_ap = ap_dict[ap_id]
            n_clients = len(ap_client_list)
            
            if n_clients == 0:
                serving_ap.airtime_utilization = 0.0
                serving_ap.total_allocated_throughput = 0.0
                serving_ap.is_saturated = False
                continue
            
            # Airtime fairness: each client gets equal airtime
            airtime_per_client = 1.0 / n_clients
            
            # Calculate throughput each client can achieve with their airtime
            client_achievable_throughputs = []
            total_demand = 0.0
            
            for client in ap_client_list:
                # With equal airtime, client achieves: max_rate * airtime_fraction
                achievable = client.max_rate_mbps * airtime_per_client
                client_achievable_throughputs.append(achievable)
                total_demand += client.load
            
            # Calculate total throughput if all demands are met
            total_needed = sum(min(client.load, achievable) 
                             for client, achievable in zip(ap_client_list, client_achievable_throughputs))
            
            # Check if AP can satisfy all demands
            if total_needed <= serving_ap.max_throughput:
                # AP is not saturated - allocate based on demand
                serving_ap.is_saturated = False
                
                for i, client in enumerate(ap_client_list):
                    # Client gets what they need (limited by their max rate with fair airtime)
                    allocated = min(client.load, client_achievable_throughputs[i])
                    client.throughput_mbps = allocated
                    client.airtime_fraction = airtime_per_client
                    client.is_satisfied = (allocated >= client.load * 0.99)  # 99% threshold
                
                serving_ap.total_allocated_throughput = total_needed
                # Calculate actual airtime used (some clients may use less than their fair share)
                serving_ap.airtime_utilization = total_needed / serving_ap.max_throughput
                
            else:
                # AP is saturated - scale down all allocations proportionally
                serving_ap.is_saturated = True
                scaling_factor = serving_ap.max_throughput / total_needed
                
                for i, client in enumerate(ap_client_list):
                    # Scale down the allocation proportionally
                    desired = min(client.load, client_achievable_throughputs[i])
                    client.throughput_mbps = desired * scaling_factor
                    client.airtime_fraction = airtime_per_client
                    client.is_satisfied = False  # No client fully satisfied when saturated
                
                serving_ap.total_allocated_throughput = serving_ap.max_throughput
                serving_ap.airtime_utilization = 1.0  # 100% utilized
    
    def _dbm_to_mw(self, dbm: float) -> float:
        """Convert dBm to milliwatts."""
        return 10 ** (dbm / 10.0)
    
    
    def _compute_channel_overlap(self, ap1: AccessPoint, ap2: AccessPoint) -> float:
        """
        Compute channel overlap factor (0-1) for 2.4 GHz WiFi.
        Adjacent channels in 2.4 GHz overlap significantly.
        """
        # Center frequency for each channel: 2407 + 5*channel MHz
        freq1 = 2407 + 5 * ap1.channel
        freq2 = 2407 + 5 * ap2.channel
        freq_separation = abs(freq1 - freq2)
        
        # Channels need ~25 MHz separation for no overlap (5 channels apart)
        # Gaussian-like overlap model
        bandwidth_sum = (ap1.bandwidth + ap2.bandwidth) / 2.0
        overlap = math.exp(-2.0 * (freq_separation / bandwidth_sum) ** 2)
        
        return overlap
    
    def update_ap_conn(self, clients: List[Client], 
                       access_points: List[AccessPoint]) -> None:
        """Update AP loads based on connected clients."""
        # Reset all AP client lists
        for ap in access_points:
            ap.connected_clients = []
        
        ap_dict = {ap.id: ap for ap in access_points}
        
        for client in clients:
            if client.associated_ap is not None:
                ap = ap_dict[client.associated_ap]
                ap.connected_clients.append(client.id)
    

class InterferenceGraphBuilder:
    """Builds interference graph from access points."""
    
    def __init__(self, propagation_model: PropagationModel, 
                 interference_threshold_dbm: float = -80.0,
                 rssi_min_dbm: float = -90.0,
                 rssi_max_dbm: float = -50.0):
        self.model = propagation_model
        self.threshold = interference_threshold_dbm
        self.rssi_min = rssi_min_dbm
        self.rssi_max = rssi_max_dbm
    
    def _compute_distance(self, ap1: AccessPoint, ap2: AccessPoint) -> float:
        """Euclidean distance between two access points."""
        return math.sqrt((ap1.x - ap2.x)**2 + (ap1.y - ap2.y)**2)
    
    def _compute_interference(self, tx_ap: AccessPoint, rx_ap: AccessPoint) -> float:
        """Compute interference power from tx_ap at rx_ap location."""
        distance = self._compute_distance(tx_ap, rx_ap)
        return self.model.compute_received_power(tx_ap.tx_power, distance)
    
    def _normalize_rssi(self, rssi_dbm: float) -> float:
        """Normalize RSSI from dBm to 0-1 scale."""
        # Clamp to range
        rssi_clamped = max(self.rssi_min, min(rssi_dbm, self.rssi_max))
        # Normalize to 0-1 (higher RSSI = stronger interference = higher value)
        return (rssi_clamped - self.rssi_min) / (self.rssi_max - self.rssi_min)
    
    def _compute_channel_overlap(self, ap1: AccessPoint, ap2: AccessPoint) -> float:
        """Compute channel overlap factor (0-1) for 2.4 GHz WiFi."""
        # Center frequency for each channel: 2407 + 5*channel MHz
        freq1 = 2407 + 5 * ap1.channel
        freq2 = 2407 + 5 * ap2.channel
        freq_separation = abs(freq1 - freq2)
        
        # Gaussian-like overlap model
        bandwidth_sum = (ap1.bandwidth + ap2.bandwidth) / 2.0
        overlap = math.exp(-2.0 * (freq_separation / bandwidth_sum) ** 2)
        
        return overlap
    
    def _compute_interference_weight(self, tx_ap: AccessPoint, rx_ap: AccessPoint, 
                                     rssi_weight: float = 0.5, 
                                     load_weight: float = 0.2,
                                     channel_weight: float = 0.3) -> float:
        """Compute normalized interference weight (0-1) from RSSI, load, and channel overlap."""
        # Get raw RSSI
        rssi_dbm = self._compute_interference(tx_ap, rx_ap)
        
        # Normalize RSSI and load
        rssi_normalized = self._normalize_rssi(rssi_dbm)
        load_normalized = tx_ap.airtime_utilization
        
        # Compute channel overlap
        overlap = self._compute_channel_overlap(tx_ap, rx_ap)
        
        # Weighted combination
        interference = overlap * (rssi_weight * rssi_normalized + 
                       load_weight * load_normalized)
        
        return interference
    
    def build_graph(self, access_points: List[AccessPoint]) -> nx.DiGraph:
        """Build directed interference graph."""
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for ap in access_points:
            G.add_node(ap.id, x=ap.x, y=ap.y, tx_power=ap.tx_power, 
                      load=ap.airtime_utilization, num_clients=len(ap.connected_clients),
                      channel=ap.channel, max_throughput=ap.max_throughput,
                      airtime_utilization=ap.airtime_utilization,
                      is_saturated=ap.is_saturated)
        
        # Add edges based on interference
        for i, tx_ap in enumerate(access_points):
            for j, rx_ap in enumerate(access_points):
                if i == j:
                    continue
                
                interference_power = self._compute_interference(tx_ap, rx_ap)
                
                if interference_power > self.threshold:
                    # Compute normalized interference weight (0-1)
                    weight = self._compute_interference_weight(tx_ap, rx_ap)
                    
                    G.add_edge(tx_ap.id, rx_ap.id, 
                             weight=weight,
                             interference_dbm=interference_power,
                             distance=self._compute_distance(tx_ap, rx_ap))        
        return G

class WirelessSimulation:
    """Main simulation coordinator."""
    
    def __init__(self, environment: "Environment", 
                 propagation_model: "PropagationModel",
                 interference_threshold_dbm: float = -80.0):
        self.env = environment
        self.prop_model = propagation_model
        self.mobility = ClientMobility(environment)
        self.assoc_manager = ClientAssociationManager(propagation_model)
        self.graph_builder = InterferenceGraphBuilder(propagation_model, 
                                                     interference_threshold_dbm)
        self.access_points = []
        self.clients = []
        self.visualizer = None
    
    def add_access_point(self, ap: "AccessPoint"):
        """Add an access point to simulation."""
        self.access_points.append(ap)
    
    def add_client(self, client: "Client"):
        """Add a client to simulation."""
        self.clients.append(client)
    
    def initialize(self):
        """Initialize simulation with Voronoi-based association."""
        self.assoc_manager.voronoi_association(self.clients, self.access_points)
        self.assoc_manager.update_ap_conn(self.clients, self.access_points)
        self.assoc_manager.compute_sinr_and_throughput(self.clients, self.access_points)
    
    def step(self):
        """Execute one simulation step."""
        # Move clients
        for client in self.clients:
            new_x, new_y = self.mobility.random_walk(client)
            client.x = new_x
            client.y = new_y
        
        # Reassociate based on signal strength
        self.assoc_manager.signal_strength_association(self.clients, self.access_points)
        self.assoc_manager.update_ap_conn(self.clients, self.access_points)
        
        # Compute SINR and throughput for all clients
        self.assoc_manager.compute_sinr_and_throughput(self.clients, self.access_points)
    
    def get_interference_graph(self) -> nx.DiGraph:
        """Get current interference graph."""
        return self.graph_builder.build_graph(self.access_points)
    
    def enable_visualization(self, width: int = 800, height: int = 800):
        """Enable pygame visualization."""
        if not PYGAME_AVAILABLE:
            print("Pygame not available. Install with: pip install pygame")
            return False
        
        self.visualizer = SimulationVisualizer(self, width, height)
        return True
    
    def run_with_visualization(self, steps: Optional[int] = None, fps: int = 10):
        """Run simulation with visualization."""
        if self.visualizer is None:
            print("Visualization not enabled. Call enable_visualization() first.")
            return
        
        self.visualizer.run(steps, fps)
    
    def print_status(self):
        """Print current simulation status."""
        print("\n=== Simulation Status ===")
        print(f"Access Points: {len(self.access_points)}")
        print(f"Clients: {len(self.clients)}")
        
        print("\nAP Status:")
        for ap in self.access_points:
            print(f"  AP {ap.id}: AT={ap.airtime_utilization:.2f}, "
                  f"Clients={len(ap.connected_clients)}, "
                  f"Channel={ap.channel}, "
                  f"Position=({ap.x:.1f}, {ap.y:.1f})")
        
        print("\nClient Status:")
        for client in self.clients:
            print(f"  Client {client.id}: AP={client.associated_ap}, "
                  f"Load={client.load:.2f}, "
                  f"Airtime={client.airtime_fraction:.2%}, "
                  f"SINR={client.sinr_db:.1f} dB, "
                  f"Throughput={client.throughput_mbps:.1f} Mbps, "
                  f"Position=({client.x:.1f}, {client.y:.1f})")
        
        # Compute statistics
        if self.clients:
            valid_clients = [c for c in self.clients if c.sinr_db != float('-inf')]
            if valid_clients:
                avg_sinr = sum(c.sinr_db for c in valid_clients) / len(valid_clients)
                avg_throughput = sum(c.throughput_mbps for c in self.clients) / len(self.clients)
                total_throughput = sum(c.throughput_mbps for c in self.clients)
                
                print(f"\n=== Network Statistics ===")
                print(f"Average SINR: {avg_sinr:.1f} dB")
                print(f"Average Client Throughput: {avg_throughput:.1f} Mbps")
                print(f"Total Network Throughput: {total_throughput:.1f} Mbps")
                
                # Per-AP statistics
                print(f"\n=== Per-AP Statistics ===")
                for ap in self.access_points:
                    ap_clients = [c for c in self.clients if c.associated_ap == ap.id]
                    if ap_clients:
                        ap_total_tput = sum(c.throughput_mbps for c in ap_clients)
                        ap_avg_sinr = sum(c.sinr_db for c in ap_clients) / len(ap_clients)
                        print(f"AP {ap.id}: {len(ap_clients)} clients, "
                              f"Avg SINR={ap_avg_sinr:.1f} dB, "
                              f"Total Throughput={ap_total_tput:.1f} Mbps")



# Example usage
if __name__ == "__main__":
    # Create environment
    env = Environment(x_min=0, x_max=50, y_min=0, y_max=50)
    
    # Create propagation model
    base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=3.0)
    fading_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
    
    # Create simulation
    sim = WirelessSimulation(env, fading_model, interference_threshold_dbm=-75.0)
    
    # Add access points with channel assignments and max throughput
    sim.add_access_point(AccessPoint(id=0, x=10, y=10, tx_power=20, channel=1, 
                                     bandwidth=20, max_throughput=150.0))
    sim.add_access_point(AccessPoint(id=1, x=25, y=20, tx_power=20, channel=1, 
                                     bandwidth=20, max_throughput=150.0))
    sim.add_access_point(AccessPoint(id=2, x=40, y=10, tx_power=20, channel=6, 
                                     bandwidth=20, max_throughput=150.0))
    sim.add_access_point(AccessPoint(id=3, x=25, y=40, tx_power=20, channel=11, 
                                     bandwidth=20, max_throughput=150.0))
    
    # Add clients with demand in Mbps
    for i in range(15):
        x = random.uniform(5, 45)
        y = random.uniform(5, 45)
        demand_mbps = random.uniform(5, 30)  # 5-30 Mbps demand
        sim.add_client(Client(id=i, x=x, y=y, load=demand_mbps))
    
    # Initialize
    sim.initialize()
    
    # Choose mode: visualization or console
    USE_VISUALIZATION = True  # Set to False for console-only mode
    
    if USE_VISUALIZATION and PYGAME_AVAILABLE:
        print("Starting visualization... (Press ESC to exit)")
        print("\nVisualization Controls:")
        print("  SPACE: Pause/Resume")
        print("  I: Toggle interference display")
        print("  A: Toggle association lines")
        print("  C: Toggle coverage areas")
        print("  ESC: Exit")
        print("\nClient Color Code:")
        print("  Green: SINR > 20 dB (Excellent)")
        print("  Yellow: SINR 10-20 dB (Good)")
        print("  Red: SINR < 10 dB (Poor)")
        print("\nClient Info: [SINR in dB] [Allocated/Demand Mbps] [Airtime %]\n")
        
        sim.enable_visualization(width=900, height=900)
        sim.run_with_visualization(steps=None, fps=10)  # None = run indefinitely
    else:
        if USE_VISUALIZATION and not PYGAME_AVAILABLE:
            print("Pygame not available. Running in console mode.")
            print("Install pygame with: pip install pygame\n")
        
        # Console mode
        sim.print_status()
        
        print("\n\n=== After 5 steps ===")
        for _ in range(5):
            sim.step()
        
        sim.print_status()
        
        # Get interference graph
        graph = sim.get_interference_graph()
        print(f"\nInterference Graph: {graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges")