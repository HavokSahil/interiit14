"""
Pygame-based visualization for wireless simulation.
"""
from typing import Tuple, Optional
import networkx as nx
import math
import os
import pygame
import pygame_gui

from datatype import AccessPoint, Client, Interferer
from metrics import APMetricsManager

# GNN imports (optional)
try:
    import torch
    import numpy as np
    from torch_geometric.data import Data
    from gnn_model import InterferenceGNN
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False


class SimulationVisualizer:
    """Pygame-based visualization for wireless simulation."""
    
    def __init__(self, simulation: 'WirelessSimulation', width: int = 1920, height: int = 1080):
        pygame.init()
        self.sim = simulation
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Wireless Network Simulation - Advanced Controller")
        
        # GUI Manager
        self.ui_manager = pygame_gui.UIManager((width, height))
        
        # Layout
        self.SIDEBAR_WIDTH = 400
        self.MAP_WIDTH = width - self.SIDEBAR_WIDTH
        self.MAP_HEIGHT = height
        
        # Colors
        self.BG_COLOR = (15, 23, 42)
        self.SIDEBAR_BG = (30, 41, 59)
        self.TEXT_COLOR = (226, 232, 240)
        self.ACCENT_COLOR = (56, 189, 248)
        self.AP_COLOR = (99, 102, 241)
        self.CLIENT_COLORS = [(34, 197, 94), (234, 179, 8), (239, 68, 68)]
        self.INTERFERENCE_COLOR = (244, 63, 94, 60)
        self.ASSOCIATION_COLOR = (148, 163, 184, 80)
        self.SELECTION_COLOR = (255, 255, 255)
        self.INTERFERER_COLOR = (255, 64, 129)
        
        # Fonts
        self.title_font = pygame.font.SysFont("Arial", 24, bold=True)
        self.header_font = pygame.font.SysFont("Arial", 20, bold=True)
        self.font = pygame.font.SysFont("Arial", 16)
        self.small_font = pygame.font.SysFont("Arial", 14)
        
        # Scaling
        self.scale_x = self.MAP_WIDTH / (self.sim.env.x_max - self.sim.env.x_min)
        self.scale_y = self.MAP_HEIGHT / (self.sim.env.y_max - self.sim.env.y_min)
        
        # State
        self.selected_entity = None
        self.dragging = False
        self.show_interference = True
        self.show_associations = True
        self.show_coverage = False
        self.paused = False
        self.view_mode = 'simulation'
        self.show_predicted_graph = False
        self.interferer_mode = False
        self.next_interferer_id = 0
        
        # Setup GUI Elements
        self._setup_gui()
        
        # Load GNN
        self.gnn_model = None
        self.gnn_device = 'cpu'
        if GNN_AVAILABLE:
            self._load_gnn_model()

    def _setup_gui(self):
        """Create pygame_gui elements."""
        # Control Panel in Sidebar
        y_start = self.height - 250
        x_start = self.MAP_WIDTH + 20
        btn_width = 170
        btn_height = 35
        
        self.btn_pause = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((x_start, y_start), (btn_width, btn_height)),
            text='Pause/Resume (SPC)',
            manager=self.ui_manager
        )
        
        self.btn_interf = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((x_start + btn_width + 10, y_start), (btn_width, btn_height)),
            text='Toggle Interference (I)',
            manager=self.ui_manager
        )
        
        self.btn_assoc = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((x_start, y_start + 45), (btn_width, btn_height)),
            text='Toggle Assoc (A)',
            manager=self.ui_manager
        )
        
        self.btn_cover = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((x_start + btn_width + 10, y_start + 45), (btn_width, btn_height)),
            text='Toggle Coverage (C)',
            manager=self.ui_manager
        )
        
        self.btn_graph = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((x_start, y_start + 90), (btn_width, btn_height)),
            text='Graph View (G)',
            manager=self.ui_manager
        )
        
        self.btn_pred = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((x_start + btn_width + 10, y_start + 90), (btn_width, btn_height)),
            text='Pred. Graph (P)',
            manager=self.ui_manager
        )
        
        self.btn_interferer = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((x_start, y_start + 135), (btn_width, btn_height)),
            text='Add Interferer (X)',
            manager=self.ui_manager
        )

        # AP Controls (Middle - Hidden by default)
        ctrl_y = y_start - 220
        
        self.lbl_ap_ctrl = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((x_start, ctrl_y), (350, 30)),
            text="AP Controls",
            manager=self.ui_manager
        )
        
        # Channel
        self.btn_ch_dec = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((x_start, ctrl_y + 35), (40, 30)), text='-', manager=self.ui_manager)
        self.lbl_ch = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((x_start + 45, ctrl_y + 35), (100, 30)), text='CH: 1', manager=self.ui_manager)
        self.btn_ch_inc = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((x_start + 150, ctrl_y + 35), (40, 30)), text='+', manager=self.ui_manager)

        # Bandwidth
        self.btn_bw_cycle = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((x_start + 210, ctrl_y + 35), (140, 30)), text='BW: 20 MHz', manager=self.ui_manager)

        # Tx Power
        self.btn_tx_dec = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((x_start, ctrl_y + 75), (40, 30)), text='-', manager=self.ui_manager)
        self.lbl_tx = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((x_start + 45, ctrl_y + 75), (100, 30)), text='TX: 20.0', manager=self.ui_manager)
        self.btn_tx_inc = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((x_start + 150, ctrl_y + 75), (40, 30)), text='+', manager=self.ui_manager)

        # OBSS PD Threshold
        self.btn_obss_dec = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((x_start, ctrl_y + 115), (40, 30)), text='-', manager=self.ui_manager)
        self.lbl_obss = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((x_start + 45, ctrl_y + 115), (100, 30)), text='PD: -82', manager=self.ui_manager)
        self.btn_obss_inc = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((x_start + 150, ctrl_y + 115), (40, 30)), text='+', manager=self.ui_manager)

        self.ap_controls = [
            self.lbl_ap_ctrl,
            self.btn_ch_dec, self.lbl_ch, self.btn_ch_inc,
            self.btn_bw_cycle,
            self.btn_tx_dec, self.lbl_tx, self.btn_tx_inc,
            self.btn_obss_dec, self.lbl_obss, self.btn_obss_inc
        ]
        
        for c in self.ap_controls:
            c.hide()
        
        # Interferer Controls (Hidden by default)
        self.lbl_interf_ctrl = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((x_start, ctrl_y), (350, 30)),
            text="Interferer Controls",
            manager=self.ui_manager
        )
        
        # Channel
        self.btn_interf_ch_dec = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((x_start, ctrl_y + 35), (40, 30)), text='-', manager=self.ui_manager)
        self.lbl_interf_ch = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((x_start + 45, ctrl_y + 35), (100, 30)), text='CH: 1', manager=self.ui_manager)
        self.btn_interf_ch_inc = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((x_start + 150, ctrl_y + 35), (40, 30)), text='+', manager=self.ui_manager)
        
        # Bandwidth (text input)
        self.lbl_interf_bw = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((x_start + 210, ctrl_y + 35), (40, 30)), text='BW:', manager=self.ui_manager)
        self.txt_interf_bw = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((x_start + 255, ctrl_y + 35), (95, 30)), manager=self.ui_manager)
        self.txt_interf_bw.set_text("20")
        
        # Tx Power
        self.btn_interf_tx_dec = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((x_start, ctrl_y + 75), (40, 30)), text='-', manager=self.ui_manager)
        self.lbl_interf_tx = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((x_start + 45, ctrl_y + 75), (100, 30)), text='TX: 20.0', manager=self.ui_manager)
        self.btn_interf_tx_inc = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((x_start + 150, ctrl_y + 75), (40, 30)), text='+', manager=self.ui_manager)
        
        # Duty Cycle
        self.btn_interf_duty_dec = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((x_start + 210, ctrl_y + 75), (40, 30)), text='-', manager=self.ui_manager)
        self.lbl_interf_duty = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((x_start + 255, ctrl_y + 75), (95, 30)), text='Duty: 100%', manager=self.ui_manager)
        self.btn_interf_duty_inc = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((x_start + 315, ctrl_y + 75), (35, 30)), text='+', manager=self.ui_manager)
        
        # Frequency Hopping
        self.btn_interf_hop_toggle = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((x_start, ctrl_y + 115), (100, 30)), text='Hopping: OFF', manager=self.ui_manager)
        self.lbl_interf_hop_ch = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((x_start + 110, ctrl_y + 115), (35, 30)), text='CH:', manager=self.ui_manager)
        self.txt_interf_hop_ch = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((x_start + 150, ctrl_y + 115), (200, 30)), manager=self.ui_manager)
        self.txt_interf_hop_ch.set_text("1,6,11")
        
        # Delete button
        self.btn_interf_delete = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((x_start, ctrl_y + 155), (350, 30)), text='Delete', manager=self.ui_manager)
        
        self.interferer_controls = [
            self.lbl_interf_ctrl,
            self.btn_interf_ch_dec, self.lbl_interf_ch, self.btn_interf_ch_inc,
            self.lbl_interf_bw, self.txt_interf_bw,
            self.btn_interf_tx_dec, self.lbl_interf_tx, self.btn_interf_tx_inc,
            self.btn_interf_duty_dec, self.lbl_interf_duty, self.btn_interf_duty_inc,
            self.btn_interf_hop_toggle, self.lbl_interf_hop_ch, self.txt_interf_hop_ch,
            self.btn_interf_delete
        ]
        
        for c in self.interferer_controls:
            c.hide()
    
    def update_ui_state(self):
        """Update visibility and text of UI elements."""
        if isinstance(self.selected_entity, AccessPoint):
            for c in self.ap_controls: c.show()
            for c in self.interferer_controls: c.hide()
            self.lbl_ch.set_text(f"CH: {self.selected_entity.channel}")
            self.btn_bw_cycle.set_text(f"BW: {self.selected_entity.bandwidth}")
            self.lbl_tx.set_text(f"TX: {self.selected_entity.tx_power:.1f}")
            self.lbl_obss.set_text(f"PD: {self.selected_entity.obss_pd_threshold:.0f}")
        elif isinstance(self.selected_entity, Interferer):
            for c in self.ap_controls: c.hide()
            for c in self.interferer_controls: c.show()
            self.lbl_interf_ch.set_text(f"CH: {self.selected_entity.channel}")
            self.txt_interf_bw.set_text(str(self.selected_entity.bandwidth))
            self.lbl_interf_tx.set_text(f"TX: {self.selected_entity.tx_power:.1f}")
            self.lbl_interf_duty.set_text(f"Duty: {self.selected_entity.duty_cycle*100:.0f}%")
            hop_status = "ON" if self.selected_entity.hopping_enabled else "OFF"
            self.btn_interf_hop_toggle.set_text(f"Hopping: {hop_status}")
            if self.selected_entity.hopping_channels:
                hop_ch_str = ','.join(map(str, self.selected_entity.hopping_channels))
                self.txt_interf_hop_ch.set_text(hop_ch_str)
        else:
            for c in self.ap_controls: c.hide()
            for c in self.interferer_controls: c.hide()
    
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int((x - self.sim.env.x_min) * self.scale_x)
        screen_y = int((y - self.sim.env.y_min) * self.scale_y)
        return screen_x, screen_y
    
    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        x = screen_x / self.scale_x + self.sim.env.x_min
        y = screen_y / self.scale_y + self.sim.env.y_min
        return x, y

    def _load_gnn_model(self):
        """Load trained GNN model if it exists."""
        model_path = './models/best_model.pt'
        if not os.path.exists(model_path):
            return
        
        try:
            checkpoint = torch.load(model_path, map_location=self.gnn_device, weights_only=False)
            self.gnn_model = InterferenceGNN(in_channels=11, hidden_channels=32, num_layers=3, dropout=0.2)
            self.gnn_model.load_state_dict(checkpoint['model_state_dict'])
            self.gnn_model.to(self.gnn_device)
            self.gnn_model.eval()
            print(f"Loaded GNN model from {model_path}")
        except Exception as e:
            print(f"Error loading GNN model: {e}")
            self.gnn_model = None
    
    def _create_graph_snapshot(self) -> Optional[Data]:
        # ... (Keep existing implementation, it's fine)
        # For brevity in this replacement, I'm assuming the previous implementation of _create_graph_snapshot is preserved 
        # or I need to copy it. Since I'm replacing the whole class, I MUST copy it.
        # To save space and avoid errors, I will use the previous implementation logic.
        if not GNN_AVAILABLE or self.gnn_model is None:
            return None
        
        client_map = {c.id: c for c in self.sim.clients}
        ap_features = []
        for ap in self.sim.access_points:
            inc_energy_ch1 = ap.inc_energy_ch1 if ap.inc_energy_ch1 != float('-inf') else -100.0
            inc_energy_ch6 = ap.inc_energy_ch6 if ap.inc_energy_ch6 != float('-inf') else -100.0
            inc_energy_ch11 = ap.inc_energy_ch11 if ap.inc_energy_ch11 != float('-inf') else -100.0
            
            total_throughput = 0.0
            for client_id in ap.connected_clients:
                if client_id in client_map:
                    total_throughput += client_map[client_id].throughput_mbps
            
            num_clients = len(ap.connected_clients)
            duty_cycle = APMetricsManager.ap_duty(ap)
            roam_in = ap.roam_in
            roam_out = ap.roam_out
            
            ap_features.append([
                inc_energy_ch1, inc_energy_ch6, inc_energy_ch11,
                total_throughput, float(num_clients), duty_cycle,
                float(roam_in), float(roam_out),
                float(ap.channel), float(ap.bandwidth), float(ap.tx_power)
            ])
        
        x = torch.tensor(ap_features, dtype=torch.float)
        
        # Normalization (simplified for this view)
        stats_path = 'models/norm_stats.pt'
        if os.path.exists(stats_path):
            try:
                stats = torch.load(stats_path, weights_only=False)
                mean = torch.tensor(stats['mean'], dtype=torch.float)
                std = torch.tensor(stats['std'], dtype=torch.float)
                x = (x - mean) / std
            except: pass

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
            
        return Data(x=x, edge_index=fc_edge_index, num_nodes=num_nodes)

    def _predict_interference_graph(self) -> Optional[nx.DiGraph]:
        # ... (Keep existing implementation logic)
        if not GNN_AVAILABLE or self.gnn_model is None: return None
        try:
            data = self._create_graph_snapshot()
            if data is None: return None
            data = data.to(self.gnn_device)
            with torch.no_grad():
                edge_index, edge_probs = self.gnn_model.predict_all_edges(data)
            
            edge_probs = edge_probs.cpu().numpy().flatten()
            edge_index = edge_index.cpu().numpy()
            mask = edge_probs >= 0.05
            predicted_edges = edge_index[:, mask]
            filtered_probs = edge_probs[mask]
            
            G = nx.DiGraph()
            for ap in self.sim.access_points:
                G.add_node(ap.id, x=ap.x, y=ap.y, 
                          load=APMetricsManager.ap_duty(ap), 
                          num_clients=len(ap.connected_clients),
                          channel=ap.channel)
            for (src, dst), prob in zip(predicted_edges.T, filtered_probs):
                src_id = self.sim.access_points[src].id
                dst_id = self.sim.access_points[dst].id
                G.add_edge(src_id, dst_id, weight=float(prob))
            return G
        except: return None

    def draw_sidebar(self):
        """Draw the informative sidebar."""
        # Background
        sidebar_rect = pygame.Rect(self.MAP_WIDTH, 0, self.SIDEBAR_WIDTH, self.height)
        pygame.draw.rect(self.screen, self.SIDEBAR_BG, sidebar_rect)
        pygame.draw.line(self.screen, (50, 60, 80), (self.MAP_WIDTH, 0), (self.MAP_WIDTH, self.height), 2)
        
        x_start = self.MAP_WIDTH + 20
        y = 20
        
        # Title
        title = self.title_font.render("Network Controller", True, self.ACCENT_COLOR)
        self.screen.blit(title, (x_start, y))
        y += 40
        
        # Global Stats
        self.draw_text("Global Statistics", self.header_font, x_start, y)
        y += 30
        
        valid_clients = [c for c in self.sim.clients if c.sinr_db != float('-inf')]
        avg_sinr = sum(c.sinr_db for c in valid_clients) / len(valid_clients) if valid_clients else 0
        total_tput = sum(c.throughput_mbps for c in self.sim.clients)
        avg_retry = sum(c.retry_rate for c in self.sim.clients) / len(self.sim.clients) if self.sim.clients else 0
        
        stats = [
            f"APs: {len(self.sim.access_points)}",
            f"Clients: {len(self.sim.clients)}",
            f"Avg SINR: {avg_sinr:.1f} dB",
            f"Total T-put: {total_tput:.1f} Mbps",
            f"Avg Retry: {avg_retry:.1f}%",
            f"Step: {self.sim.step_count}"
        ]
        
        for stat in stats:
            self.draw_text(stat, self.font, x_start, y)
            y += 25
            
        y += 20
        pygame.draw.line(self.screen, (50, 60, 80), (x_start, y), (self.width - 20, y), 1)
        y += 20
        
        # Selected Entity Details
        if self.selected_entity:
            if isinstance(self.selected_entity, AccessPoint):
                self.draw_ap_details(self.selected_entity, x_start, y)
            elif isinstance(self.selected_entity, Client):
                self.draw_client_details(self.selected_entity, x_start, y)
            elif isinstance(self.selected_entity, Interferer):
                self.draw_interferer_details(self.selected_entity, x_start, y)
        else:
            msg = "Add Interferer Mode" if self.interferer_mode else "Select an AP/Client/Interferer"
            self.draw_text(msg, self.header_font, x_start, y, (100, 110, 130))
            if not self.interferer_mode:
                self.draw_text("to view details", self.font, x_start, y + 25, (100, 110, 130))
            else:
                self.draw_text("Click to place interferer", self.font, x_start, y + 25, (100, 110, 130))
            
        # Control Panel Header
        y_bottom = self.height - 280
        self.draw_text("Control Panel", self.header_font, x_start, y_bottom)

    def draw_ap_details(self, ap: AccessPoint, x: int, y: int):
        self.draw_text(f"Access Point {ap.id}", self.header_font, x, y, self.ACCENT_COLOR)
        y += 30
        
        # Operating Channel Energy
        op_energy = float('-inf')
        if ap.channel == 1: op_energy = ap.inc_energy_ch1
        elif ap.channel == 6: op_energy = ap.inc_energy_ch6
        elif ap.channel == 11: op_energy = ap.inc_energy_ch11
        energy_str = f"{op_energy:.1f} dBm" if op_energy != float('-inf') else "-inf"
        
        details = [
            f"Channel: {ap.channel}",
            f"Tx Power: {ap.tx_power:.1f} dBm",
            f"Load (Duty): {APMetricsManager.ap_duty(ap):.2f}",
            f"Clients: {len(ap.connected_clients)}",
            f"Op. Energy: {energy_str}",
            f"CCA Busy: {ap.cca_busy_percentage:.1f}%",
            f"OBSS PD: {ap.obss_pd_threshold:.0f} dBm",
            f"Roam In/Out: {ap.roam_in_rate:.2f} / {ap.roam_out_rate:.2f}",
            f"p95 T-put: {ap.p95_throughput:.1f} Mbps",
            f"p95 Retry: {ap.p95_retry_rate:.1f}%"
        ]
        
        for detail in details:
            self.draw_text(detail, self.font, x, y)
            y += 25

    def draw_client_details(self, client: Client, x: int, y: int):
        self.draw_text(f"Client {client.id}", self.header_font, x, y, self.ACCENT_COLOR)
        y += 30
        
        assoc_str = f"AP {client.associated_ap}" if client.associated_ap is not None else "None"
        
        details = [
            f"Associated: {assoc_str}",
            f"Demand: {client.demand_mbps:.1f} Mbps",
            f"Throughput: {client.throughput_mbps:.1f} Mbps",
            f"SINR: {client.sinr_db:.1f} dB",
            f"RSSI: {client.rssi_dbm:.1f} dBm",
            f"Retry Rate: {client.retry_rate:.1f}%",
            f"Airtime: {client.airtime_fraction:.1%}",
            f"Velocity: {client.velocity:.2f}"
        ]
        
        for detail in details:
            self.draw_text(detail, self.font, x, y)
            y += 25
    
    def draw_interferer_details(self, interferer: Interferer, x: int, y: int):
        self.draw_text(f"Interferer {interferer.id}", self.header_font, x, y, self.ACCENT_COLOR)
        y += 30
        
        details = [
            f"Channel: {interferer.channel}",
            f"Bandwidth: {interferer.bandwidth} MHz",
            f"Tx Power: {interferer.tx_power:.1f} dBm",
            f"Duty Cycle: {interferer.duty_cycle*100:.0f}%",
            f"Position: ({interferer.x:.1f}, {interferer.y:.1f})"
        ]
        
        for detail in details:
            self.draw_text(detail, self.font, x, y)
            y += 25

    def draw_text(self, text: str, font, x: int, y: int, color=None):
        if color is None: color = self.TEXT_COLOR
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))

    def draw_access_points(self):
        for ap in self.sim.access_points:
            pos = self.world_to_screen(ap.x, ap.y)
            
            # Selection Highlight
            if self.selected_entity == ap:
                pygame.draw.circle(self.screen, self.SELECTION_COLOR, pos, 16, 2)
                # Glow effect
                s = pygame.Surface((40, 40), pygame.SRCALPHA)
                pygame.draw.circle(s, (255, 255, 255, 50), (20, 20), 18)
                self.screen.blit(s, (pos[0]-20, pos[1]-20))

            # Draw AP
            pygame.draw.circle(self.screen, self.AP_COLOR, pos, 12)
            pygame.draw.circle(self.screen, (255, 255, 255), pos, 8)
            
            # ID only (details in sidebar)
            id_text = self.small_font.render(str(ap.id), True, (20, 20, 20))
            text_rect = id_text.get_rect(center=pos)
            self.screen.blit(id_text, text_rect)

    def draw_clients(self):
        for client in self.sim.clients:
            pos = self.world_to_screen(client.x, client.y)
            
            # Color based on SINR
            if client.sinr_db > 20: color = self.CLIENT_COLORS[0]
            elif client.sinr_db > 10: color = self.CLIENT_COLORS[1]
            else: color = self.CLIENT_COLORS[2]
            
            # Selection Highlight
            if self.selected_entity == client:
                pygame.draw.circle(self.screen, self.SELECTION_COLOR, pos, 10, 2)
            
            pygame.draw.circle(self.screen, color, pos, 6)
    
    def draw_interferers(self):
        for interferer in self.sim.interferers:
            pos = self.world_to_screen(interferer.x, interferer.y)
            
            # Selection Highlight
            if self.selected_entity == interferer:
                pygame.draw.circle(self.screen, self.SELECTION_COLOR, pos, 16, 2)
                # Glow effect
                s = pygame.Surface((40, 40), pygame.SRCALPHA)
                pygame.draw.circle(s, (255, 255, 255, 50), (20, 20), 18)
                self.screen.blit(s, (pos[0]-20, pos[1]-20))
            
            # Draw interferer as X shape
            pygame.draw.circle(self.screen, self.INTERFERER_COLOR, pos, 10)
            # Draw X
            pygame.draw.line(self.screen, (255, 255, 255), (pos[0]-6, pos[1]-6), (pos[0]+6, pos[1]+6), 2)
            pygame.draw.line(self.screen, (255, 255, 255), (pos[0]-6, pos[1]+6), (pos[0]+6, pos[1]-6), 2)
            
            # ID
            id_text = self.small_font.render(f"I{interferer.id}", True, self.TEXT_COLOR)
            self.screen.blit(id_text, (pos[0]+12, pos[1]-8))

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

    def draw_interference_graph(self):
        """Draw interference graph with labeled nodes (actual or predicted)."""
        # Get graph (actual or predicted)
        if self.show_predicted_graph and GNN_AVAILABLE and self.gnn_model is not None:
            graph = self._predict_interference_graph()
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
            screen_x = margin + int((x - min_x) / x_range * (self.MAP_WIDTH - 2 * margin))
            screen_y = margin + int((y - min_y) / y_range * (self.MAP_HEIGHT - 2 * margin))
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
            
            edge_label = self.small_font.render(f"{u}->{v}: {weight:.2f}", True, self.TEXT_COLOR)
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
            id_text = self.font.render(str(node), True, (20, 20, 20))
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

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
            
            # Pass event to UI Manager
            self.ui_manager.process_events(event)
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return False
                elif event.key == pygame.K_SPACE: self.paused = not self.paused
                elif event.key == pygame.K_g: self.view_mode = 'graph' if self.view_mode == 'simulation' else 'simulation'
                elif event.key == pygame.K_i: self.show_interference = not self.show_interference
                elif event.key == pygame.K_a: self.show_associations = not self.show_associations
                elif event.key == pygame.K_c: self.show_coverage = not self.show_coverage
                elif event.key == pygame.K_p: 
                    if GNN_AVAILABLE and self.gnn_model: self.show_predicted_graph = not self.show_predicted_graph
                elif event.key == pygame.K_x:
                    self.interferer_mode = not self.interferer_mode
                    if self.interferer_mode:
                        self.selected_entity = None
                        self.update_ui_state()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    mx, my = pygame.mouse.get_pos()
                    if mx < self.MAP_WIDTH:
                        if self.interferer_mode:
                            # Place interferer
                            wx, wy = self.screen_to_world(mx, my)
                            new_interferer = Interferer(
                                id=self.next_interferer_id,
                                x=wx, y=wy,
                                tx_power=20.0,
                                channel=1,
                                bandwidth=20.0,
                                duty_cycle=1.0
                            )
                            self.sim.add_interferer(new_interferer)
                            self.selected_entity = new_interferer
                            self.next_interferer_id += 1
                            self.interferer_mode = False  # Exit interferer mode after placing
                            self.update_ui_state()
                        else:
                            self.select_entity(mx, my)
                            if self.selected_entity and (isinstance(self.selected_entity, AccessPoint) or isinstance(self.selected_entity, Interferer)):
                                self.dragging = True
            
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
            
            if event.type == pygame.MOUSEMOTION:
                if self.dragging and self.selected_entity:
                    if isinstance(self.selected_entity, AccessPoint) or isinstance(self.selected_entity, Interferer):
                        mx, my = pygame.mouse.get_pos()
                        # Clamp to map
                        mx = min(max(0, mx), self.MAP_WIDTH)
                        my = min(max(0, my), self.MAP_HEIGHT)
                        
                        wx, wy = self.screen_to_world(mx, my)
                        self.selected_entity.x = wx
                        self.selected_entity.y = wy
            
            # Handle GUI Button Events
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.btn_pause:
                    self.paused = not self.paused
                elif event.ui_element == self.btn_interf:
                    self.show_interference = not self.show_interference
                elif event.ui_element == self.btn_assoc:
                    self.show_associations = not self.show_associations
                elif event.ui_element == self.btn_cover:
                    self.show_coverage = not self.show_coverage
                elif event.ui_element == self.btn_graph:
                    self.view_mode = 'graph' if self.view_mode == 'simulation' else 'simulation'
                elif event.ui_element == self.btn_pred:
                    if GNN_AVAILABLE and self.gnn_model:
                        self.show_predicted_graph = not self.show_predicted_graph
                elif event.ui_element == self.btn_interferer:
                    self.interferer_mode = not self.interferer_mode
                    if self.interferer_mode:
                        self.selected_entity = None
                        self.update_ui_state()
                
                # AP Controls
                if isinstance(self.selected_entity, AccessPoint):
                    ap = self.selected_entity
                    if event.ui_element == self.btn_ch_dec:
                        ap.channel = max(1, ap.channel - 1)
                    elif event.ui_element == self.btn_ch_inc:
                        ap.channel = min(14, ap.channel + 1)
                    elif event.ui_element == self.btn_bw_cycle:
                        bws = [20, 40, 80, 160]
                        try:
                            idx = bws.index(ap.bandwidth)
                            ap.bandwidth = bws[(idx + 1) % len(bws)]
                        except: ap.bandwidth = 20
                    elif event.ui_element == self.btn_tx_dec:
                        ap.tx_power -= 1.0
                    elif event.ui_element == self.btn_tx_inc:
                        ap.tx_power += 1.0
                    elif event.ui_element == self.btn_obss_dec:
                        ap.obss_pd_threshold -= 1.0
                    elif event.ui_element == self.btn_obss_inc:
                        ap.obss_pd_threshold += 1.0
                    
                    self.update_ui_state()
                
                # Interferer Controls
                if isinstance(self.selected_entity, Interferer):
                    interferer = self.selected_entity
                    if event.ui_element == self.btn_interf_ch_dec:
                        interferer.channel = max(1, interferer.channel - 1)
                    elif event.ui_element == self.btn_interf_ch_inc:
                        interferer.channel = min(14, interferer.channel + 1)
                    elif event.ui_element == self.btn_interf_tx_dec:
                        interferer.tx_power -= 1.0
                    elif event.ui_element == self.btn_interf_tx_inc:
                        interferer.tx_power += 1.0
                    elif event.ui_element == self.btn_interf_duty_dec:
                        interferer.duty_cycle = max(0.0, interferer.duty_cycle - 0.1)
                    elif event.ui_element == self.btn_interf_duty_inc:
                        interferer.duty_cycle = min(1.0, interferer.duty_cycle + 0.1)
                    elif event.ui_element == self.btn_interf_hop_toggle:
                        interferer.hopping_enabled = not interferer.hopping_enabled
                        if interferer.hopping_enabled:
                            # Parse hopping channels from text
                            try:
                                ch_text = self.txt_interf_hop_ch.get_text().strip()
                                channels = [int(ch.strip()) for ch in ch_text.split(',') if ch.strip()]
                                if channels:
                                    interferer.hopping_channels = channels
                                    interferer.hopping_index = 0
                                    interferer.channel = channels[0]
                                else:
                                    interferer.hopping_enabled = False
                            except ValueError:
                                interferer.hopping_enabled = False
                        else:
                            interferer.hopping_channels = []
                    elif event.ui_element == self.btn_interf_delete:
                        self.sim.remove_interferer(interferer.id)
                        self.selected_entity = None
                    
                    self.update_ui_state()
            
            # Handle text entry changes
            if event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
                if isinstance(self.selected_entity, Interferer):
                    interferer = self.selected_entity
                    if event.ui_element == self.txt_interf_bw:
                        try:
                            bw = float(self.txt_interf_bw.get_text())
                            if bw > 0:
                                interferer.bandwidth = bw
                            else:
                                self.txt_interf_bw.set_text(str(interferer.bandwidth))
                        except ValueError:
                            self.txt_interf_bw.set_text(str(interferer.bandwidth))
                    elif event.ui_element == self.txt_interf_hop_ch:
                        # Update hopping channels if hopping is enabled
                        if interferer.hopping_enabled:
                            try:
                                ch_text = self.txt_interf_hop_ch.get_text().strip()
                                channels = [int(ch.strip()) for ch in ch_text.split(',') if ch.strip()]
                                if channels:
                                    interferer.hopping_channels = channels
                                    interferer.hopping_index = 0
                                    interferer.channel = channels[0]
                            except ValueError:
                                pass
        
        return True

    def select_entity(self, mx: int, my: int):
        # Find nearest AP
        nearest_ap = None
        min_dist_ap = float('inf')
        for ap in self.sim.access_points:
            px, py = self.world_to_screen(ap.x, ap.y)
            dist = math.sqrt((px-mx)**2 + (py-my)**2)
            if dist < min_dist_ap:
                min_dist_ap = dist
                nearest_ap = ap
        
        # Find nearest Client
        nearest_client = None
        min_dist_client = float('inf')
        for c in self.sim.clients:
            px, py = self.world_to_screen(c.x, c.y)
            dist = math.sqrt((px-mx)**2 + (py-my)**2)
            if dist < min_dist_client:
                min_dist_client = dist
                nearest_client = c
        
        # Find nearest Interferer
        nearest_interferer = None
        min_dist_interferer = float('inf')
        for i in self.sim.interferers:
            px, py = self.world_to_screen(i.x, i.y)
            dist = math.sqrt((px-mx)**2 + (py-my)**2)
            if dist < min_dist_interferer:
                min_dist_interferer = dist
                nearest_interferer = i
        
        # Select the closest one within threshold
        threshold = 30
        min_dist = min(min_dist_ap, min_dist_client, min_dist_interferer)
        
        if min_dist > threshold:
            self.selected_entity = None
        elif min_dist == min_dist_interferer:
            self.selected_entity = nearest_interferer
        elif min_dist == min_dist_client:
            self.selected_entity = nearest_client
        else:
            self.selected_entity = nearest_ap
        
        # Update UI visibility
        self.update_ui_state()

    def run(self, max_steps: Optional[int] = None, fps: int = 10):
        clock = pygame.time.Clock()
        running = True
        while running:
            time_delta = clock.tick(fps)/1000.0
            running = self.handle_events()
            
            # Update GUI
            self.ui_manager.update(time_delta)
            
            if not self.paused:
                self.sim.step()
                if max_steps and self.sim.step_count >= max_steps: running = False
            
            self.screen.fill(self.BG_COLOR)
            
            if self.view_mode == 'simulation':
                if self.show_coverage: self.draw_coverage_areas()
                if self.show_interference: self.draw_interference_edges()
                if self.show_associations: self.draw_association_lines()
                self.draw_access_points()
                self.draw_clients()
                self.draw_interferers()
            else:
                self.draw_interference_graph()
            
            self.draw_sidebar()
            
            # Draw GUI
            self.ui_manager.draw_ui(self.screen)
            
            pygame.display.flip()
        pygame.quit()

    # Helper methods for drawing graph/coverage/etc need to be preserved or re-added if I overwrote them.
    # Since I replaced the WHOLE class, I need to make sure I included everything.
    # I missed: draw_coverage_areas, draw_interference_edges, draw_association_lines, draw_interference_graph
    # I will add them back in the next step to ensure the class is complete.
    # The previous replacement was truncated/incomplete because I didn't include those methods in the ReplacementContent.
    # I will perform a second replacement to add the missing methods.

