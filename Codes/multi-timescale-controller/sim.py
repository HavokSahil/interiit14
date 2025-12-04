from typing import Tuple, Optional
import networkx as nx
import math
import os
from model import *
from assoc import *
from metrics import *
from logger import SimulationLogger

# Check pygame availability
try:
    import pygame
    import pygame_gui
    PYGAME_AVAILABLE = True
    from visualizer import SimulationVisualizer
except ImportError:
    PYGAME_AVAILABLE = False

class WirelessSimulation:
    """Main simulation coordinator."""
    
    def __init__(self, environment: "Environment", 
                 propagation_model: "PropagationModel",
                 interference_threshold_dbm: float = -80.0,
                 enable_logging: bool = False,
                 log_dir: str = "logs"):
        self.env = environment
        self.prop_model = propagation_model
        self.mobility = ClientMobility(environment)
        self.interferers = []
        self.assoc_manager = ClientAssociationManager([], [], propagation_model)
        self.graph_builder = InterferenceGraphBuilder(propagation_model, 
                                                     interference_threshold_dbm)
        self.ap_metrics = APMetricsManager([], [], propagation_model, self.interferers)
        self.client_metrics = ClientMetricsManager([], [], propagation_model, self.interferers)
        self.access_points = []
        self.clients = []
        self.visualizer = None
        self.logger = SimulationLogger(log_dir=log_dir) if enable_logging else None
        self.step_count = 0
    
    def add_interferer(self, interferer: "Interferer"):
        """Add an interferer to simulation."""
        self.interferers.append(interferer)
    
    def remove_interferer(self, interferer_id: int):
        """Remove an interferer from simulation."""
        self.interferers = [i for i in self.interferers if i.id != interferer_id]
    
    def add_access_point(self, ap: "AccessPoint"):
        """Add an access point to simulation."""
        self.access_points.append(ap)
        self.assoc_manager.access_points.append(ap)
        self.ap_metrics.access_points.append(ap)
        self.client_metrics.access_points.append(ap)
    
    def add_client(self, client: "Client"):
        """Add a client to simulation."""
        self.clients.append(client)
        self.assoc_manager.clients.append(client)
        self.ap_metrics.clients.append(client)
        self.client_metrics.clients.append(client)
    
    def initialize(self):
        """Initialize simulation with Voronoi-based association."""
        roam_list = self.assoc_manager.voronoi_association()
        
        # Log initial state (step 0)
        if self.logger:
            self.client_metrics.update()
            self.ap_metrics.update()
            graph = self.get_interference_graph()
            self.logger.log_step(self.step_count, self.access_points, self.clients, roam_list, graph)
    
    def step(self):
        """Execute one simulation step."""
        # Increment step counter
        self.step_count += 1
        
        # Move clients
        for client in self.clients:
            new_x, new_y = self.mobility.random_walk(client)
            client.x = new_x
            client.y = new_y
        
        # Handle interferer frequency hopping
        for interferer in self.interferers:
            if interferer.hopping_enabled and interferer.hopping_channels:
                interferer.hopping_index = (interferer.hopping_index + 1) % len(interferer.hopping_channels)
                interferer.channel = interferer.hopping_channels[interferer.hopping_index]
        
        # Reassociate based on signal strength
        roam_list = self.assoc_manager.signal_strength_association()
        
        # Update the metrics of the AP and Client
        self.ap_metrics.update()
        self.client_metrics.update()
        
        # Log this step
        if self.logger:
            graph = self.get_interference_graph()
            self.logger.log_step(self.step_count, self.access_points, self.clients, roam_list, graph)
    
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
            # Format energy for display: show per-channel or aggregate
            ch1_str = f"{ap.inc_energy_ch1:.1f}" if ap.inc_energy_ch1 != float('-inf') else "-inf"
            ch6_str = f"{ap.inc_energy_ch6:.1f}" if ap.inc_energy_ch6 != float('-inf') else "-inf"
            ch11_str = f"{ap.inc_energy_ch11:.1f}" if ap.inc_energy_ch11 != float('-inf') else "-inf"
            print(f"  AP {ap.id}: DT={APMetricsManager.ap_duty(ap):.2f}, "
                  f"Clients={len(ap.connected_clients)}, "
                  f"Channel={ap.channel}, "
                  f"Energy=[Ch1:{ch1_str}, Ch6:{ch6_str}, Ch11:{ch11_str}]dBm, "
                  f"Roam(In/Out)={ap.roam_in_rate:.2f}/{ap.roam_out_rate:.2f}, "
                  f"p95(T/R)={ap.p95_throughput:.1f}/{ap.p95_retry_rate:.1f}, "
                  f"Position=({ap.x:.1f}, {ap.y:.1f})")
        
        print("\nClient Status:")
        for client in self.clients:
            print(f"  Client {client.id}: AP={client.associated_ap}, "
                  f"Airtime={client.airtime_fraction:.2%}, "
                  f"SINR={client.sinr_db:.1f} dB, "
                  f"Throughput={client.throughput_mbps:.1f} Mbps, "
                  f"Retry={client.retry_rate:.1f}%, "
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


