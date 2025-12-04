"""
FTM RTT-enabled wireless simulation coordinator.
Extended from multi-timescale-controller/sim.py with FTM distance measurement.
"""

from typing import Tuple, Optional
import networkx as nx
import math
import random
from ftm_model import *
from ftm_assoc import *
from ftm_metrics import *
from ftm_datatype import *
from ftm_utils import trilaterate_position, is_within_bounds

# Check pygame availability
try:
    import pygame
    import pygame_gui
    PYGAME_AVAILABLE = True
    from ftm_visualizer import FTMSimulationVisualizer
except ImportError:
    PYGAME_AVAILABLE = False


class FTMWirelessSimulation:
    """Main simulation coordinator with FTM RTT support."""
    
    def __init__(self, environment: Environment, 
                 propagation_model: PropagationModel,
                 interference_threshold_dbm: float = -80.0,
                 ftm_measurement_noise_std: float = 0.5):
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
        self.step_count = 0
        self.ftm_noise_std = ftm_measurement_noise_std  # Standard deviation for FTM noise (meters)
        self.ftm_noise_std = ftm_measurement_noise_std  # Standard deviation for FTM noise (meters)
        
        # Tile-based interference detection
        self.tile_size = 3.0  # 3m x 3m tiles
        self.tiles = {}  # (grid_x, grid_y) -> Tile object
        self._initialize_tiles()
        
    def _initialize_tiles(self):
        """Initialize the grid of tiles."""
        cols = int((self.env.x_max - self.env.x_min) / self.tile_size) + 1
        rows = int((self.env.y_max - self.env.y_min) / self.tile_size) + 1
        
        tile_id = 0
        for i in range(cols):
            for j in range(rows):
                x = self.env.x_min + i * self.tile_size + self.tile_size / 2
                y = self.env.y_min + j * self.tile_size + self.tile_size / 2
                self.tiles[(i, j)] = Tile(id=tile_id, x=x, y=y, size=self.tile_size)
                tile_id += 1
    
    def add_interferer(self, interferer: Interferer):
        """Add an interferer to simulation."""
        self.interferers.append(interferer)
    
    def remove_interferer(self, interferer_id: int):
        """Remove an interferer from simulation."""
        self.interferers = [i for i in self.interferers if i.id != interferer_id]
    
    def add_access_point(self, ap: AccessPoint):
        """Add an access point to simulation."""
        self.access_points.append(ap)
        self.assoc_manager.access_points.append(ap)
        self.ap_metrics.access_points.append(ap)
        self.client_metrics.access_points.append(ap)
    
    def add_client(self, client: Client):
        """Add a client to simulation."""
        self.clients.append(client)
        self.assoc_manager.clients.append(client)
        self.ap_metrics.clients.append(client)
        self.client_metrics.clients.append(client)
    
    def update_ftm_measurements(self):
        """Update FTM RTT distance measurements for FTM-capable clients."""
        for client in self.clients:
            if not client.ftm_capable:
                continue
            
            # Clear previous measurements
            client.measured_distances.clear()
            
            # Measure distance to all APs
            for ap in self.access_points:
                # Calculate actual Euclidean distance
                actual_distance = math.sqrt((client.x - ap.x)**2 + (client.y - ap.y)**2)
                
                # Add realistic measurement noise (Gaussian)
                noise = random.gauss(0, self.ftm_noise_std)
                measured_distance = max(0.1, actual_distance + noise)  # Ensure positive
                
                # Store measured distance
                client.measured_distances[ap.id] = measured_distance
    
    def update_client_metrics(self):
        """Simulate PER and Retry Rate based on SINR."""
        for client in self.clients:
            # Simple simulation model:
            # High SINR (>20dB) -> Low PER (<1%), Low Retry (<5%)
            # Low SINR (<10dB) -> High PER (>10%), High Retry (>20%)
            
            if client.sinr_db > 20:
                client.packet_error_rate = random.uniform(0, 1)
                client.retry_rate = random.uniform(0, 5)
            elif client.sinr_db > 10:
                client.packet_error_rate = random.uniform(1, 5)
                client.retry_rate = random.uniform(5, 15)
            else:
                # Poor signal -> High errors/retries
                client.packet_error_rate = random.uniform(5, 20)
                client.retry_rate = random.uniform(15, 50)

    def collect_interference_reports(self):
        """
        Collect metrics and aggregate them into tiles.
        Detect interference hotspots based on thresholds.
        """
        # Reset tile stats for this step (or keep moving average - here we do per-step for responsiveness)
        for tile in self.tiles.values():
            tile.sample_count = 0
            tile.avg_rssi = -100.0
            tile.avg_per = 0.0
            tile.avg_retry_rate = 0.0
            tile.is_interference_hotspot = False
            
        for client in self.clients:
            if not client.ftm_capable or len(client.measured_distances) < 3:
                continue
            
            # 1. Estimate Position (Trilateration)
            ap_positions = []
            distances = []
            for ap_id, dist in client.measured_distances.items():
                ap = next((a for a in self.access_points if a.id == ap_id), None)
                if ap:
                    ap_positions.append((ap.x, ap.y))
                    distances.append(dist)
            
            estimated_pos = trilaterate_position(ap_positions, distances)
            
            if estimated_pos:
                est_x, est_y = estimated_pos
                
                # 2. Assign to Tile
                if not is_within_bounds(self.env, est_x, est_y):
                    continue
                    
                grid_x = int((est_x - self.env.x_min) / self.tile_size)
                grid_y = int((est_y - self.env.y_min) / self.tile_size)
                
                if (grid_x, grid_y) in self.tiles:
                    tile = self.tiles[(grid_x, grid_y)]
                    
                    # 3. Aggregate Metrics (Cumulative moving average for this step)
                    n = tile.sample_count
                    tile.avg_rssi = (tile.avg_rssi * n + client.rssi_dbm) / (n + 1)
                    tile.avg_per = (tile.avg_per * n + client.packet_error_rate) / (n + 1)
                    tile.avg_retry_rate = (tile.avg_retry_rate * n + client.retry_rate) / (n + 1)
                    tile.sample_count += 1
        
        # 4. Detect Interference Hotspots
        for tile in self.tiles.values():
            if tile.sample_count > 0:
                # Thresholds: RSSI > -65 dBm (Strong) AND (PER > 5% OR Retries > 20%)
                strong_signal = tile.avg_rssi > -65
                poor_performance = (tile.avg_per > 5.0) or (tile.avg_retry_rate > 20.0)
                
                if strong_signal and poor_performance:
                    tile.is_interference_hotspot = True
    
    def initialize(self):
        """Initialize simulation with Voronoi-based association."""
        # Update FTM measurements before first association
        self.update_ftm_measurements()
        
        # Perform initial association
        roam_list = self.assoc_manager.voronoi_association()
    
    def step(self):
        """Execute one simulation step."""
        # Increment step counter
        self.step_count += 1
        
        # Move clients
        for client in self.clients:
            new_x, new_y = self.mobility.random_walk(client)
            client.x = new_x
            client.y = new_y
        
        # Update FTM measurements (happens at each step for FTM-capable clients)
        self.update_ftm_measurements()
        
        # Simulate client performance metrics (PER, Retries)
        self.update_client_metrics()
        
        # Collect interference reports based on FTM positioning
        self.collect_interference_reports()
        
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
    
    def get_interference_graph(self) -> nx.DiGraph:
        """Get current interference graph."""
        return self.graph_builder.build_graph(self.access_points)
    
    def enable_visualization(self, width: int = 1920, height: int = 1080):
        """Enable pygame visualization."""
        if not PYGAME_AVAILABLE:
            print("Pygame not available. Install with: pip install pygame")
            return False
        
        self.visualizer = FTMSimulationVisualizer(self, width, height)
        return True
    
    def run_with_visualization(self, steps: Optional[int] = None, fps: int = 10):
        """Run simulation with visualization."""
        if self.visualizer is None:
            print("Visualization not enabled. Call enable_visualization() first.")
            return
        
        self.visualizer.run(steps, fps)
    
    def print_status(self):
        """Print current simulation status."""
        print("\n=== FTM RTT Simulation Status ===")
        print(f"Access Points: {len(self.access_points)}")
        print(f"Clients: {len(self.clients)}")
        
        # FTM statistics
        ftm_capable_count = sum(1 for c in self.clients if c.ftm_capable)
        print(f"FTM-Capable Clients: {ftm_capable_count}/{len(self.clients)} ({ftm_capable_count/len(self.clients)*100:.1f}%)")
        
        print("\nAP Status:")
        for ap in self.access_points:
            ch1_str = f"{ap.inc_energy_ch1:.1f}" if ap.inc_energy_ch1 != float('-inf') else "-inf"
            ch6_str = f"{ap.inc_energy_ch6:.1f}" if ap.inc_energy_ch6 != float('-inf') else "-inf"
            ch11_str = f"{ap.inc_energy_ch11:.1f}" if ap.inc_energy_ch11 != float('-inf') else "-inf"
            print(f"  AP {ap.id}: Clients={len(ap.connected_clients)}, "
                  f"Channel={ap.channel}, "
                  f"Position=({ap.x:.1f}, {ap.y:.1f})")
        
        print("\nClient Status:")
        for client in self.clients:
            ftm_status = "FTM✓" if client.ftm_capable else "FTM✗"
            dist_info = ""
            if client.ftm_capable and client.measured_distances:
                # Show distance to associated AP
                if client.associated_ap is not None and client.associated_ap in client.measured_distances:
                    dist = client.measured_distances[client.associated_ap]
                    dist_info = f", Dist={dist:.1f}m"
            
            print(f"  Client {client.id} [{ftm_status}]: AP={client.associated_ap}, "
                  f"SINR={client.sinr_db:.1f} dB{dist_info}, "
                  f"Position=({client.x:.1f}, {client.y:.1f})")
