from typing import Tuple, Optional
import networkx as nx
import math
import os
from model import *
from assoc import *
from metrics import *
from logger import SimulationLogger

# GNN imports (optional - only if model exists)
try:
    import torch
    import numpy as np
    from torch_geometric.data import Data
    from gnn_model import InterferenceGNN
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
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
        self.assoc_manager = ClientAssociationManager([], [], propagation_model)
        self.graph_builder = InterferenceGraphBuilder(propagation_model, 
                                                     interference_threshold_dbm)
        self.ap_metrics = APMetricsManager([], [], propagation_model)
        self.client_metrics = ClientMetricsManager([], [], propagation_model)
        self.access_points = []
        self.clients = []
        self.visualizer = None
        self.logger = SimulationLogger(log_dir=log_dir) if enable_logging else None
        self.step_count = 0
    
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
            energy_str = f"{ap.inc_energy:.1f}" if ap.inc_energy != float('-inf') else "-inf"
            print(f"  AP {ap.id}: DT={APMetricsManager.ap_duty(ap):.2f}, "
                  f"Clients={len(ap.connected_clients)}, "
                  f"Channel={ap.channel}, "
                  f"IncEnergy={energy_str}dBm, "
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


class SimulationVisualizer:
    """Pygame-based visualization for wireless simulation."""
    
    def __init__(self, simulation: 'WirelessSimulation', width: int = 800, height: int = 800):
        pygame.init()
        self.sim = simulation
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Wireless Network Simulation - SINR & Airtime")
        
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
        self.show_predicted_graph = False  # Toggle for GNN predictions
        
        # Load GNN model if available
        self.gnn_model = None
        self.gnn_device = 'cpu'
        if GNN_AVAILABLE:
            self._load_gnn_model()
    
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int((x - self.sim.env.x_min) * self.scale_x)
        screen_y = int((y - self.sim.env.y_min) * self.scale_y)
        return screen_x, screen_y
    
    def _load_gnn_model(self):
        """Load trained GNN model if it exists."""
        model_path = './models/best_model.pt'
        if not os.path.exists(model_path):
            print(f"GNN model not found at {model_path}. Train a model first.")
            return
        
        try:
            # Load model
            checkpoint = torch.load(model_path, map_location=self.gnn_device, weights_only=False)
            self.gnn_model = InterferenceGNN(
                in_channels=9,  # 9 node features
                hidden_channels=32,
                num_layers=3,  # EdgeConv
                dropout=0.2
            )
            self.gnn_model.load_state_dict(checkpoint['model_state_dict'])
            self.gnn_model.to(self.gnn_device)
            self.gnn_model.eval()
            print(f"Loaded GNN model from {model_path}")
            print("  Press 'P' during simulation to toggle predicted interference graph")
        except Exception as e:
            print(f"Error loading GNN model: {e}")
            self.gnn_model = None
    
    def _create_graph_snapshot(self) -> Optional[Data]:
        """Create PyG Data object from current simulation state."""
        if not GNN_AVAILABLE or self.gnn_model is None:
            return None
        
        # Create client map for fast lookup
        client_map = {c.id: c for c in self.sim.clients}
        
        # Extract features for each AP
        ap_features = []
        for ap in self.sim.access_points:
            # 1. Incoming energy (raw)
            inc_energy = ap.inc_energy if ap.inc_energy != float('-inf') else -100.0
            
            # 2. Throughput
            # connected_clients contains IDs, so we look up the objects
            total_throughput = 0.0
            for client_id in ap.connected_clients:
                if client_id in client_map:
                    total_throughput += client_map[client_id].throughput_mbps
            
            # 3. Number of clients
            num_clients = len(ap.connected_clients)
            
            # 4. Duty cycle
            duty_cycle = APMetricsManager.ap_duty(ap) # Using the existing helper
            
            # 5. Roaming events (from simulation stats)
            # Assuming self.sim.roam_events exists and is updated by the simulation
            roam_in = self.sim.roam_events.get(ap.id, {}).get('in', 0) if hasattr(self.sim, 'roam_events') else 0
            roam_out = self.sim.roam_events.get(ap.id, {}).get('out', 0) if hasattr(self.sim, 'roam_events') else 0
            
            # 6. Channel (new)
            channel = float(ap.channel)
            
            # 7. Bandwidth (new)
            bandwidth = float(ap.bandwidth)
            
            # 8. Tx Power (new)
            tx_power = float(ap.tx_power)
            
            ap_features.append([
                inc_energy,
                total_throughput,
                float(num_clients),
                duty_cycle,
                float(roam_in),
                float(roam_out),
                channel,
                bandwidth,
                tx_power
            ])
        x = torch.tensor(ap_features, dtype=torch.float)
        
        # Apply z-score normalization with stats from training
        stats_path = 'models/norm_stats.pt'
        if os.path.exists(stats_path):
            try:
                stats = torch.load(stats_path, weights_only=False)
                mean = torch.tensor(stats['mean'], dtype=torch.float)
                std = torch.tensor(stats['std'], dtype=torch.float)
                x = (x - mean) / std
                
                # Validate normalized features
                if torch.isnan(x).any():
                    raise ValueError("Normalized features contain NaN values")
                if torch.isinf(x).any():
                    raise ValueError("Normalized features contain inf values")
            except Exception as e:
                raise RuntimeError(f"Error applying normalization: {e}")
        else:
            raise FileNotFoundError(
                f"Normalization stats not found at {stats_path}. "
                "Please train the model first to generate normalization statistics."
            )
        
        # Create fully connected edge index for message passing
        num_nodes = len(self.sim.access_points)
        src_list = []
        dst_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    src_list.append(i)
                    dst_list.append(j)
                    
        if len(src_list) > 0:
            fc_edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            fc_edge_index = torch.tensor([[], []], dtype=torch.long)
        
        data = Data(x=x, edge_index=fc_edge_index, num_nodes=num_nodes)
        return data
    
    def _predict_interference_graph(self) -> Optional[nx.DiGraph]:
        """Predict interference graph using GNN."""
        if not GNN_AVAILABLE or self.gnn_model is None:
            return None
        
        try:
            # Create graph snapshot
            data = self._create_graph_snapshot()
            if data is None:
                return None
            
            data = data.to(self.gnn_device)
            
            # Predict edges
            with torch.no_grad():
                edge_index, edge_probs = self.gnn_model.predict_all_edges(data)
            
            # Filter by threshold (0.1 for regression)
            edge_probs = edge_probs.cpu().numpy().flatten()
            edge_index = edge_index.cpu().numpy()

            threshold = 0.05
            mask = edge_probs >= threshold
            predicted_edges = edge_index[:, mask]
            filtered_probs = edge_probs[mask]
            
            # Create NetworkX graph
            G = nx.DiGraph()
            for i, ap in enumerate(self.sim.access_points):
                G.add_node(ap.id)
            
            for (src, dst), prob in zip(predicted_edges.T, filtered_probs):
                src_id = self.sim.access_points[src].id
                dst_id = self.sim.access_points[dst].id
                G.add_edge(src_id, dst_id, weight=float(prob))
            
            return G
            
        except Exception as e:
            print(f"Error predicting graph: {e}")
            return None
    
    def draw_interference_edges(self):
        """Draw interference graph edges (actual or predicted)."""
        if not self.show_interference:
            return
        
        # Get graph (actual or predicted)
        if self.show_predicted_graph and GNN_AVAILABLE and self.gnn_model is not None:
            graph = self._predict_interference_graph()
            if graph is None:
                return
        else:
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
            
            # Color: Blue for predicted, Red-Yellow for actual
            if self.show_predicted_graph:
                # Blue gradient for predictions
                red = 100
                green = 100
                blue = int(150 + 105 * weight)
                color = (red, green, blue, 100)
            else:
                # Red-Yellow gradient for actual
                red = 255
                green = int(255 * (1 - weight))
                blue = 0
                color = (red, green, blue, 80)
            
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
            
            # Draw AP info
            energy_str = f"{ap.inc_energy:.1f}" if ap.inc_energy != float('-inf') else "-inf"
            info_text = self.small_font.render(
                f"Ch{ap.channel} DUTY:{APMetricsManager.ap_duty(ap):.2f} C:{len(ap.connected_clients)} E:{energy_str}",
                True, self.TEXT_COLOR
            )
            self.screen.blit(info_text, (pos[0] + 15, pos[1] - 8))
    
    def draw_clients(self):
        """Draw clients with SINR color coding."""
        for client in self.sim.clients:
            pos = self.world_to_screen(client.x, client.y)
            
            # Color based on SINR quality
            # Good SINR (> 20 dB): Green
            # Medium SINR (10-20 dB): Yellow
            # Poor SINR (< 10 dB): Red
            if client.sinr_db > 20:
                color = (50, 220, 50)  # Green
            elif client.sinr_db > 10:
                color = (220, 220, 50)  # Yellow
            else:
                color = (220, 50, 50)  # Red
            
            # Draw client
            pygame.draw.circle(self.screen, color, pos, 6)
            pygame.draw.circle(self.screen, (255, 255, 255), pos, 3)
            
            # Draw client info
            info_text = self.small_font.render(
                f"{client.sinr_db:.0f}dB {client.throughput_mbps:.0f}Mbps A:{client.airtime_fraction:.0%}",
                True, self.TEXT_COLOR
            )
            self.screen.blit(info_text, (pos[0] + 8, pos[1] - 8))

    def draw_interference_graph(self):
        """Draw interference graph with labeled nodes (actual or predicted)."""
        # Get graph (actual or predicted)
        if self.show_predicted_graph and GNN_AVAILABLE and self.gnn_model is not None:
            graph = self._predict_interference_graph()
            print(graph)
            if graph is None:
                return
        else:
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
            
            # Color: Blue for predicted, Red-Yellow for actual
            if self.show_predicted_graph:
                # Blue gradient for predictions
                red = 100
                green = 100
                blue = int(150 + 105 * weight)
                color = (red, green, blue)
            else:
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
        title = self.font.render(
            f"Wireless Network Simulation - Step {step_count}",
            True, self.TEXT_COLOR
        )
        self.screen.blit(title, (10, 10))
        
        # Statistics
        valid_clients = [c for c in self.sim.clients if c.sinr_db != float('-inf')]
        avg_sinr = sum(c.sinr_db for c in valid_clients) / len(valid_clients) if valid_clients else 0
        total_tput = sum(c.throughput_mbps for c in self.sim.clients)
        
        stats = [
            f"APs: {len(self.sim.access_points)}  Clients: {len(self.sim.clients)}",
            f"Avg SINR: {avg_sinr:.1f} dB",
            f"Total Throughput: {total_tput:.1f} Mbps"
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
            "P: Toggle Predicted Graph" if GNN_AVAILABLE and self.gnn_model else "P: (No GNN model)",
            "ESC: Exit"
        ]
        
        y_offset = self.height - 140
        for control in controls:
            text = self.small_font.render(control, True, self.TEXT_COLOR)
            self.screen.blit(text, (10, y_offset))
            y_offset += 18
        
        # Status indicators
        if self.paused:
            pause_text = self.font.render("PAUSED", True, (255, 0, 0))
            self.screen.blit(pause_text, (self.width - 100, 10))
        
        if self.show_predicted_graph:
            pred_text = self.font.render("PREDICTED GRAPH", True, (100, 100, 255))
            self.screen.blit(pred_text, (self.width - 200, 35))
    
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
                elif event.key == pygame.K_p:
                    if GNN_AVAILABLE and self.gnn_model is not None:
                        self.show_predicted_graph = not self.show_predicted_graph
                        status = "ON" if self.show_predicted_graph else "OFF"
                        print(f"Predicted interference graph: {status}")
                    else:
                        print("GNN model not available. Train a model first.")
        
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
