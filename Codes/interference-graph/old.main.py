import math
import random
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
from scipy.spatial import Voronoi
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


@dataclass
class AccessPoint:
    """Represents a wireless access point."""
    id: int
    x: float
    y: float
    tx_power: float  # dBm
    channel: int = 1  # WiFi channel (1-11 for 2.4 GHz)
    bandwidth: int = 20  # Channel bandwidth in MHz (20 or 40)
    load: float = 0.0  # Computed from connected clients
    connected_clients: List[int] = field(default_factory=list)


@dataclass
class Client:
    """Represents a wireless client."""
    id: int
    x: float
    y: float
    load: float  # Client's traffic load (0.0 to 1.0)
    associated_ap: Optional[int] = None
    velocity: float = 1.0  # Speed in units per step
    direction: float = 0.0  # Direction in radians
    sinr: float = 0.0  # Signal-to-Interference-plus-Noise Ratio (dB)
    throughput: float = 0.0  # Throughput (Mbps)


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


class PropagationModel(ABC):
    """Abstract base class for propagation models."""
    
    @abstractmethod
    def compute_received_power(self, tx_power: float, distance: float) -> float:
        """Compute received power in dBm."""
        pass


class PathLossModel(PropagationModel):
    """Free space path loss model."""
    
    def __init__(self, frequency_mhz: float = 2400, path_loss_exp: float = 2.0):
        self.freq = frequency_mhz
        self.n = path_loss_exp
    
    def compute_received_power(self, tx_power: float, distance: float) -> float:
        if distance == 0:
            return tx_power
        fspl = 20 * math.log10(distance) + 20 * math.log10(self.freq) - 27.55
        additional_loss = 10 * (self.n - 2) * math.log10(distance) if self.n != 2 else 0
        return tx_power - fspl - additional_loss


class MultipathFadingModel(PropagationModel):
    """Path loss with Rayleigh fading."""
    
    def __init__(self, base_model: PropagationModel, fading_margin_db: float = 10.0):
        self.base_model = base_model
        self.fading_margin = fading_margin_db
    
    def compute_received_power(self, tx_power: float, distance: float) -> float:
        base_power = self.base_model.compute_received_power(tx_power, distance)
        return base_power - self.fading_margin


class ClientMobility:
    """Handles client movement with velocity and direction."""
    
    def __init__(self, environment: Environment, 
                 max_velocity: float = 2.0, 
                 min_velocity: float = 0.5,
                 velocity_change_rate: float = 0.1,
                 direction_change_rate: float = math.pi / 8):
        self.env = environment
        self.max_velocity = max_velocity
        self.min_velocity = min_velocity
        self.velocity_change_rate = velocity_change_rate
        self.direction_change_rate = direction_change_rate
    
    def random_walk(self, client: Client) -> Tuple[float, float]:
        """
        Update client position based on velocity and direction.
        Velocity and direction change gradually with random variations.
        Returns new (x, y) position within bounds.
        """
        # Randomly adjust direction (slightly left or right)
        direction_change = random.uniform(-self.direction_change_rate, self.direction_change_rate)
        client.direction += direction_change
        
        # Keep direction in [0, 2π) range
        client.direction = client.direction % (2 * math.pi)
        
        # Randomly adjust velocity
        velocity_change = random.uniform(-self.velocity_change_rate, self.velocity_change_rate)
        client.velocity += velocity_change
        
        # Clamp velocity to bounds
        client.velocity = max(self.min_velocity, min(self.max_velocity, client.velocity))
        
        # Compute new position
        new_x = client.x + client.velocity * math.cos(client.direction)
        new_y = client.y + client.velocity * math.sin(client.direction)
        
        # Handle boundary collisions with reflection
        if new_x < self.env.x_min or new_x > self.env.x_max:
            # Reflect horizontally
            client.direction = math.pi - client.direction
            new_x = max(self.env.x_min, min(new_x, self.env.x_max))
        
        if new_y < self.env.y_min or new_y > self.env.y_max:
            # Reflect vertically
            client.direction = -client.direction
            new_y = max(self.env.y_min, min(new_y, self.env.y_max))
        
        # Normalize direction after reflections
        client.direction = client.direction % (2 * math.pi)
        
        return new_x, new_y


class ClientAssociationManager:
    """Manages client-AP associations."""
    
    def __init__(self, propagation_model: PropagationModel):
        self.model = propagation_model
    
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
    
    def update_ap_loads(self, clients: List[Client], 
                       access_points: List[AccessPoint]) -> None:
        """Update AP loads based on connected clients."""
        # Reset all AP loads and client lists
        for ap in access_points:
            ap.load = 0.0
            ap.connected_clients = []
        
        # Aggregate client loads
        ap_dict = {ap.id: ap for ap in access_points}
        
        for client in clients:
            if client.associated_ap is not None:
                ap = ap_dict[client.associated_ap]
                ap.load += client.load
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
    
    def _normalize_load(self, load: float, max_load: float = 10.0) -> float:
        """Normalize load to 0-1 scale."""
        # Clamp load and normalize
        return min(1.0, load / max_load)
    
    def _compute_channel_overlap(self, ap1: AccessPoint, ap2: AccessPoint) -> float:
        """Compute channel overlap factor (0-1) for 2.4 GHz WiFi.
        
        Args:
            ap1: First access point
            ap2: Second access point
            
        Returns:
            Overlap factor from 0.0 (no overlap) to 1.0 (same channel)
        """
        # 2.4 GHz channel center frequencies: 2407 + 5*channel_number MHz
        freq1 = 2407 + 5 * ap1.channel
        freq2 = 2407 + 5 * ap2.channel
        
        # Frequency separation
        freq_separation = abs(freq1 - freq2)
        
        # Effective bandwidth (half-power bandwidth)
        bw1 = ap1.bandwidth
        bw2 = ap2.bandwidth
        
        # Maximum overlap occurs when channels are the same
        # Overlap decreases as frequency separation increases
        # For 20 MHz channels: significant overlap up to ~25 MHz separation
        # For adjacent channels (5 MHz apart): high overlap
        
        # Use Gaussian-like overlap function
        # bandwidth_sum represents the combined spectral spread
        bandwidth_sum = (bw1 + bw2) / 2.0
        
        # Overlap factor using exponential decay
        # When separation = 0: overlap = 1.0
        # When separation = bandwidth_sum: overlap ≈ 0.37
        # When separation = 2*bandwidth_sum: overlap ≈ 0.14
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
        load_normalized = self._normalize_load(tx_ap.load)
        
        # Compute channel overlap
        overlap = self._compute_channel_overlap(tx_ap, rx_ap)
        
        # Weighted combination
        interference = overlap * (rssi_weight * rssi_normalized + 
                       load_weight * load_normalized)
        
        return interference
    
    def build_graph(self, access_points: List[AccessPoint]) -> nx.DiGraph:
        """Build directed interference graph with normalized edge weights (0-1)."""
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for ap in access_points:
            G.add_node(ap.id, x=ap.x, y=ap.y, tx_power=ap.tx_power, 
                      load=ap.load, num_clients=len(ap.connected_clients),
                      channel=ap.channel, bandwidth=ap.bandwidth)
        
        # Add edges based on interference
        for i, tx_ap in enumerate(access_points):
            for j, rx_ap in enumerate(access_points):
                if i == j:
                    continue
                
                # Get raw RSSI for threshold check
                interference_power = self._compute_interference(tx_ap, rx_ap)
                
                if interference_power > self.threshold:
                    # Compute normalized interference weight (0-1)
                    weight = self._compute_interference_weight(tx_ap, rx_ap)
                    
                    G.add_edge(tx_ap.id, rx_ap.id, 
                             weight=weight,
                             interference_dbm=interference_power,  # Keep for reference
                             distance=self._compute_distance(tx_ap, rx_ap))
        
        return G


class WirelessSimulation:
    """Main simulation coordinator."""
    
    def __init__(self, environment: Environment, 
                 propagation_model: PropagationModel,
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
        self.noise_floor_dbm = -90.0  # Thermal noise floor
    
    def _dbm_to_watts(self, dbm: float) -> float:
        """Convert dBm to Watts."""
        return 10 ** ((dbm - 30) / 10)
    
    def _watts_to_dbm(self, watts: float) -> float:
        """Convert Watts to dBm."""
        if watts <= 0: return -100.0
        return 10 * math.log10(watts) + 30

    def update_metrics(self):
        """Update SINR and throughput for all clients."""
        noise_watts = self._dbm_to_watts(self.noise_floor_dbm)
        
        for client in self.clients:
            if client.associated_ap is None:
                client.sinr = 0.0
                client.throughput = 0.0
                continue
            
            # Find associated AP
            assoc_ap = next(ap for ap in self.access_points if ap.id == client.associated_ap)
            
            # 1. Calculate Signal Power (S)
            dist = self.assoc_manager._compute_distance(client.x, client.y, assoc_ap.x, assoc_ap.y)
            signal_dbm = self.prop_model.compute_received_power(assoc_ap.tx_power, dist)
            signal_watts = self._dbm_to_watts(signal_dbm)
            
            # 2. Calculate Interference Power (I)
            interference_watts = 0.0
            for ap in self.access_points:
                if ap.id == client.associated_ap:
                    continue
                
                # Distance to interfering AP
                int_dist = self.assoc_manager._compute_distance(client.x, client.y, ap.x, ap.y)
                int_dbm = self.prop_model.compute_received_power(ap.tx_power, int_dist)
                int_watts = self._dbm_to_watts(int_dbm)
                
                # Channel overlap factor
                overlap = self.graph_builder._compute_channel_overlap(assoc_ap, ap)
                
                interference_watts += int_watts * overlap
            
            # 3. Calculate SINR
            # SINR = S / (I + N)
            sinr_linear = signal_watts / (interference_watts + noise_watts)
            client.sinr = 10 * math.log10(sinr_linear) if sinr_linear > 0 else -10.0
            
            # 4. Calculate Throughput (Shannon Capacity)
            # C = B * log2(1 + SINR)
            # Bandwidth in Hz (MHz * 1e6)
            bandwidth_hz = assoc_ap.bandwidth * 1e6
            capacity_bps = bandwidth_hz * math.log2(1 + sinr_linear)
            client.throughput = capacity_bps / 1e6  # Convert to Mbps
    
    def add_access_point(self, ap: AccessPoint):
        """Add an access point to simulation."""
        self.access_points.append(ap)
    
    def add_client(self, client: Client):
        """Add a client to simulation."""
        self.clients.append(client)
    
    def initialize(self):
        """Initialize simulation with Voronoi-based association."""
        self.assoc_manager.voronoi_association(self.clients, self.access_points)
        self.assoc_manager.update_ap_loads(self.clients, self.access_points)
    
    def step(self):
        """Execute one simulation step."""
        # Move clients
        for client in self.clients:
            new_x, new_y = self.mobility.random_walk(client)
            client.x = new_x
            client.y = new_y
        
        # Reassociate based on signal strength
        # Reassociate based on signal strength
        self.assoc_manager.signal_strength_association(self.clients, self.access_points)
        self.assoc_manager.update_ap_loads(self.clients, self.access_points)
        
        # Update metrics (SINR, Throughput)
        self.update_metrics()
    
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
            print(f"  AP {ap.id}: Load={ap.load:.2f}, "
                  f"Clients={len(ap.connected_clients)}, "
                  f"Position=({ap.x:.1f}, {ap.y:.1f})")
        
        print("\nClient Associations:")
        for client in self.clients:
            print(f"  Client {client.id}: AP={client.associated_ap}, "
                  f"Load={client.load:.2f}, Position=({client.x:.1f}, {client.y:.1f}), "
                  f"SINR={client.sinr:.1f} dB, T-put={client.throughput:.1f} Mbps")


class SimulationVisualizer:
    """Pygame-based visualization for wireless simulation."""
    
    def __init__(self, simulation: WirelessSimulation, width: int = 800, height: int = 800):
        pygame.init()
        self.sim = simulation
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Wireless Network Simulation")
        
        # Colors
        self.BG_COLOR = (240, 240, 245)
        self.AP_COLOR = (0, 120, 215)
        self.CLIENT_COLORS = [
            (220, 50, 50), (50, 220, 50), (220, 220, 50),
            (220, 50, 220), (50, 220, 220), (255, 140, 0)
        ]
        self.INTERFERENCE_COLOR = (255, 100, 100, 80)
        self.ASSOCIATION_COLOR = (100, 100, 100, 100)
        self.TEXT_COLOR = (20, 20, 20)
        
        # Fonts
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 16)
        
        # Scaling
        self.scale_x = width / (self.sim.env.x_max - self.sim.env.x_min)
        self.scale_y = height / (self.sim.env.y_max - self.sim.env.y_min)
        
        # Visualization options
        self.show_interference = True
        self.show_associations = True
        self.show_coverage = False
        self.paused = False
        self.view_mode = 'simulation'  # 'simulation' or 'graph'
    
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int((x - self.sim.env.x_min) * self.scale_x)
        screen_y = int((y - self.sim.env.y_min) * self.scale_y)
        return screen_x, screen_y
    
    def draw_interference_edges(self):
        """Draw interference graph edges."""
        if not self.show_interference:
            return
        
        graph = self.sim.get_interference_graph()
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        for u, v, data in graph.edges(data=True):
            ap_u = next(ap for ap in self.sim.access_points if ap.id == u)
            ap_v = next(ap for ap in self.sim.access_points if ap.id == v)
            
            pos_u = self.world_to_screen(ap_u.x, ap_u.y)
            pos_v = self.world_to_screen(ap_v.x, ap_v.y)
            
            # Use normalized weight (0-1)
            weight = data.get('weight', 0.0)
            
            # Line thickness based on interference weight
            thickness = int(1 + weight * 3)
            
            # Color gradient from yellow (low) to red (high)
            red = 255
            green = int(255 * (1 - weight))
            blue = 0
            color = (*( red, green, blue), 80)
            
            pygame.draw.line(surface, color, pos_u, pos_v, thickness)
        
        self.screen.blit(surface, (0, 0))
    
    def draw_association_lines(self):
        """Draw client-AP association lines."""
        if not self.show_associations:
            return
        
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        for client in self.sim.clients:
            if client.associated_ap is None:
                continue
            
            ap = next(ap for ap in self.sim.access_points if ap.id == client.associated_ap)
            client_pos = self.world_to_screen(client.x, client.y)
            ap_pos = self.world_to_screen(ap.x, ap.y)
            
            pygame.draw.line(surface, self.ASSOCIATION_COLOR, client_pos, ap_pos, 1)
        
        self.screen.blit(surface, (0, 0))
    
    def draw_coverage_areas(self):
        """Draw AP coverage circles."""
        if not self.show_coverage:
            return
        
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        for i, ap in enumerate(self.sim.access_points):
            pos = self.world_to_screen(ap.x, ap.y)
            color = (*self.CLIENT_COLORS[i % len(self.CLIENT_COLORS)], 30)
            
            # Coverage radius based on tx_power (simplified)
            coverage_radius = int(ap.tx_power * self.scale_x * 0.5)
            pygame.draw.circle(surface, color, pos, coverage_radius)
        
        self.screen.blit(surface, (0, 0))
    
    def draw_access_points(self):
        """Draw access points."""
        for i, ap in enumerate(self.sim.access_points):
            pos = self.world_to_screen(ap.x, ap.y)
            
            # Draw AP
            pygame.draw.circle(self.screen, self.AP_COLOR, pos, 12)
            pygame.draw.circle(self.screen, (255, 255, 255), pos, 8)
            
            # Draw ID
            id_text = self.small_font.render(str(ap.id), True, self.TEXT_COLOR)
            self.screen.blit(id_text, (pos[0] - 4, pos[1] - 5))
            
            # Draw load info
            load_text = self.small_font.render(
                f"Load: {ap.load:.1f} ({len(ap.connected_clients)} clients)",
                True, self.TEXT_COLOR
            )
            self.screen.blit(load_text, (pos[0] + 15, pos[1] - 8))
    
    def draw_clients(self):
        """Draw clients."""
        for client in self.sim.clients:
            pos = self.world_to_screen(client.x, client.y)
            
            # Color based on associated AP
            if client.associated_ap is not None:
                color = self.CLIENT_COLORS[client.associated_ap % len(self.CLIENT_COLORS)]
            else:
                color = (150, 150, 150)
            
            # Draw client
            pygame.draw.circle(self.screen, color, pos, 6)
            pygame.draw.circle(self.screen, (255, 255, 255), pos, 3)
    
    def draw_interference_graph(self):
        """Draw interference graph with labeled nodes."""
        graph = self.sim.get_interference_graph()
        
        if graph.number_of_nodes() == 0:
            # No graph to draw
            text = self.font.render("No interference graph (no APs)", True, self.TEXT_COLOR)
            self.screen.blit(text, (self.width // 2 - 100, self.height // 2))
            return
        
        # Use spring layout for graph visualization
        pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
        
        # Scale positions to screen
        margin = 80
        min_x = min(p[0] for p in pos.values())
        max_x = max(p[0] for p in pos.values())
        min_y = min(p[1] for p in pos.values())
        max_y = max(p[1] for p in pos.values())
        
        # Prevent division by zero
        x_range = max_x - min_x if max_x - min_x > 0 else 1
        y_range = max_y - min_y if max_y - min_y > 0 else 1
        
        screen_pos = {}
        for node, (x, y) in pos.items():
            screen_x = margin + int((x - min_x) / x_range * (self.width - 2 * margin))
            screen_y = margin + int((y - min_y) / y_range * (self.height - 2 * margin))
            screen_pos[node] = (screen_x, screen_y)
        
        # Draw edges with color based on interference strength
        for u, v, data in graph.edges(data=True):
            if u not in screen_pos or v not in screen_pos:
                continue
            
            pos_u = screen_pos[u]
            pos_v = screen_pos[v]
            
            # Use normalized weight (0-1)
            weight = data.get('weight', 0.0)
            
            # Gradient from yellow (weak) to red (strong)
            red = 255
            green = int(255 * (1 - weight))
            blue = 0
            color = (red, green, blue)
            
            # Line thickness based on strength
            thickness = int(2 + weight * 4)
            
            pygame.draw.line(self.screen, color, pos_u, pos_v, thickness)
            
            # Draw interference weight on edge (0-1 scale)
            # Offset label perpendicular to edge to avoid overlap with reverse edge
            dx = pos_v[0] - pos_u[0]
            dy = pos_v[1] - pos_u[1]
            edge_length = math.sqrt(dx*dx + dy*dy)
            
            if edge_length > 0:
                # Perpendicular offset (to the right of the direction from u to v)
                offset_distance = 10
                perp_x = -dy / edge_length * offset_distance
                perp_y = dx / edge_length * offset_distance
                
                mid_x = (pos_u[0] + pos_v[0]) // 2 + int(perp_x)
                mid_y = (pos_u[1] + pos_v[1]) // 2 + int(perp_y)
            else:
                mid_x = (pos_u[0] + pos_v[0]) // 2
                mid_y = (pos_u[1] + pos_v[1]) // 2
            
            edge_label = self.small_font.render(f"{u}->{v}: {weight:.2f}", True, (60, 60, 60))
            self.screen.blit(edge_label, (mid_x - 20, mid_y - 6))
        
        # Draw nodes
        node_radius = 25
        for node in graph.nodes():
            if node not in screen_pos:
                continue
            
            pos = screen_pos[node]
            node_data = graph.nodes[node]
            
            # Draw node circle
            pygame.draw.circle(self.screen, self.AP_COLOR, pos, node_radius)
            pygame.draw.circle(self.screen, (255, 255, 255), pos, node_radius - 4)
            
            # Draw node label (AP ID)
            id_text = self.font.render(str(node), True, self.TEXT_COLOR)
            text_rect = id_text.get_rect(center=pos)
            self.screen.blit(id_text, text_rect)
            
            # Draw node info below
            info_lines = [
                f"Load: {node_data.get('load', 0):.1f}",
                f"Clients: {node_data.get('num_clients', 0)}",
                f"Ch: {node_data.get('channel', 1)}"
            ]
            
            y_offset = pos[1] + node_radius + 5
            for line in info_lines:
                info_text = self.small_font.render(line, True, self.TEXT_COLOR)
                info_rect = info_text.get_rect(center=(pos[0], y_offset))
                self.screen.blit(info_text, info_rect)
                y_offset += 16
        
        # Draw legend
        legend_x = self.width - 200
        legend_y = 40
        
        legend_title = self.small_font.render("Edge Weights:", True, self.TEXT_COLOR)
        self.screen.blit(legend_title, (legend_x, legend_y))
        
        # Weak interference (0.0)
        pygame.draw.line(self.screen, (255, 255, 0), (legend_x, legend_y + 20), (legend_x + 40, legend_y + 20), 3)
        weak_text = self.small_font.render("Low (0.0)", True, self.TEXT_COLOR)
        self.screen.blit(weak_text, (legend_x + 45, legend_y + 13))
        
        # Strong interference (1.0)
        pygame.draw.line(self.screen, (255, 0, 0), (legend_x, legend_y + 40), (legend_x + 40, legend_y + 40), 6)
        strong_text = self.small_font.render("High (1.0)", True, self.TEXT_COLOR)
        self.screen.blit(strong_text, (legend_x + 45, legend_y + 33))
        
        # Graph statistics
        stats_y = legend_y + 70
        stats = [
            f"Nodes: {graph.number_of_nodes()}",
            f"Edges: {graph.number_of_edges()}",
            f"Avg Degree: {2 * graph.number_of_edges() / graph.number_of_nodes():.1f}" if graph.number_of_nodes() > 0 else "Avg Degree: 0"
        ]
        
        for stat in stats:
            stat_text = self.small_font.render(stat, True, self.TEXT_COLOR)
            self.screen.blit(stat_text, (legend_x, stats_y))
            stats_y += 18
    
    def draw_ui(self, step_count: int):
        """Draw UI elements."""
        # Title
        mode_text = "Graph View" if self.view_mode == 'graph' else "Simulation View"
        title = self.font.render(
            f"Wireless Network - {mode_text} - Step {step_count}",
            True, self.TEXT_COLOR
        )
        self.screen.blit(title, (10, 10))
        
        # Statistics
        avg_throughput = 0.0
        if self.sim.clients:
            avg_throughput = sum(c.throughput for c in self.sim.clients) / len(self.sim.clients)

        stats = [
            f"APs: {len(self.sim.access_points)}",
            f"Clients: {len(self.sim.clients)}",
            f"Avg Load: {sum(ap.load for ap in self.sim.access_points) / len(self.sim.access_points):.2f}" if self.sim.access_points else "N/A",
            f"Avg T-put: {avg_throughput:.1f} Mbps"
        ]
        
        y_offset = 35
        for stat in stats:
            text = self.small_font.render(stat, True, self.TEXT_COLOR)
            self.screen.blit(text, (10, y_offset))
            y_offset += 20
        
        # Controls
        controls = [
            "SPACE: Pause/Resume",
            "G: Toggle Graph View",
            "I: Toggle Interference",
            "A: Toggle Associations",
            "C: Toggle Coverage",
            "ESC: Exit"
        ]
        
        y_offset = self.height - 110
        for control in controls:
            text = self.small_font.render(control, True, self.TEXT_COLOR)
            self.screen.blit(text, (10, y_offset))
            y_offset += 18
        
        # Pause indicator
        if self.paused:
            pause_text = self.font.render("PAUSED", True, (255, 0, 0))
            self.screen.blit(pause_text, (self.width - 100, 10))
    
    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_g:
                    self.view_mode = 'graph' if self.view_mode == 'simulation' else 'simulation'
                elif event.key == pygame.K_i:
                    self.show_interference = not self.show_interference
                elif event.key == pygame.K_a:
                    self.show_associations = not self.show_associations
                elif event.key == pygame.K_c:
                    self.show_coverage = not self.show_coverage
        
        return True
    
    def run(self, max_steps: Optional[int] = None, fps: int = 10):
        """Run the visualization loop."""
        clock = pygame.time.Clock()
        step_count = 0
        running = True
        
        while running:
            # Handle events
            running = self.handle_events()
            
            # Update simulation
            if not self.paused:
                self.sim.step()
                step_count += 1
                
                if max_steps is not None and step_count >= max_steps:
                    running = False
            
            # Draw
            self.screen.fill(self.BG_COLOR)
            
            if self.view_mode == 'simulation':
                self.draw_coverage_areas()
                self.draw_interference_edges()
                self.draw_association_lines()
                self.draw_access_points()
                self.draw_clients()
            else:  # graph view
                self.draw_interference_graph()
            
            self.draw_ui(step_count)
            
            pygame.display.flip()
            clock.tick(fps)
        
        pygame.quit()


# Example usage
if __name__ == "__main__":
    # Create environment
    env = Environment(x_min=0, x_max=50, y_min=0, y_max=50)
    
    # Create propagation model
    base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=3.0)
    fading_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
    
    # Create simulation
    sim = WirelessSimulation(env, fading_model, interference_threshold_dbm=-75.0)
    
    # Add access points with different channels
    # Using common non-overlapping channels: 1, 6, 11
    sim.add_access_point(AccessPoint(id=0, x=10, y=10, tx_power=20, channel=1))
    sim.add_access_point(AccessPoint(id=1, x=25, y=20, tx_power=25, channel=6))
    sim.add_access_point(AccessPoint(id=2, x=40, y=10, tx_power=20, channel=11))
    sim.add_access_point(AccessPoint(id=3, x=25, y=40, tx_power=20, channel=1))  # Same as AP 0
    
    # Add clients
    for i in range(10):
        x = random.uniform(5, 45)
        y = random.uniform(5, 45)
        load = random.uniform(0.1, 0.5)
        velocity = random.uniform(0.5, 2.0)
        direction = random.uniform(0, 2 * math.pi)
        sim.add_client(Client(id=i, x=x, y=y, load=load, velocity=velocity, direction=direction))
    
    # Initialize
    sim.initialize()
    
    # Choose mode: visualization or console
    USE_VISUALIZATION = True  # Set to False for console-only mode
    
    if USE_VISUALIZATION and PYGAME_AVAILABLE:
        print("Starting visualization... (Press ESC to exit)")
        sim.enable_visualization(width=800, height=800)
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