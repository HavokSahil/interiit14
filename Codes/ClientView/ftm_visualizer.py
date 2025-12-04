"""
FTM RTT Visualizer - extends base visualization with FTM-specific displays.

This file creates a custom visualizer by adapting the original SimulationVisualizer
with FTM distance display, heatmap overlay, and client capability indication.
"""

import pygame
import pygame_gui
import math
from typing import Tuple, Optional
from ftm_datatype import Client, AccessPoint

# Import base visualizer but rename to avoid conflicts
import sys
import os

# Temporarily modify the imports in the copied base file
sys.path.insert(0, os.path.dirname(__file__))

# We'll manually define FTMSimulationVisualizer based on the core features we need
# rather than trying to modify the large file programmatically

class FTMSimulationVisualizer:
    """FTM-enhanced pygame visualization."""
    
    def __init__(self, simulation: 'FTMWirelessSimulation', width: int = 1920, height: int = 1080):
        pygame.init()
        self.sim = simulation
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("FTM RTT Wireless Simulation")
        
        # GUI Manager
        self.ui_manager = pygame_gui.UIManager((width, height))
        
        # Layout
        self.SIDEBAR_WIDTH = 450  # Wider for FTM stats
        self.MAP_WIDTH = width - self.SIDEBAR_WIDTH
        self.MAP_HEIGHT = height
        
        # Colors
        self.BG_COLOR = (15, 23, 42)
        self.SIDEBAR_BG = (30, 41, 59)
        self.TEXT_COLOR = (226, 232, 240)
        self.ACCENT_COLOR = (56, 189, 248)
        self.AP_COLOR = (99, 102, 241)
        self.CLIENT_COLORS = [(34, 197, 94), (234, 179, 8), (239, 68, 68)]
        self.FTM_INDICATOR_COLOR = (0, 150, 255)
        self.SELECTION_COLOR = (255, 255, 255)
        self.DISTANCE_LINE_COLOR = (100, 100, 255, 100)
        
        # Fonts
        self.title_font = pygame.font.SysFont("Arial", 24, bold=True)
        self.header_font = pygame.font.SysFont("Arial", 20, bold=True)
        self.font = pygame.font.SysFont("Arial", 16)
        self.small_font = pygame.font.SysFont("Arial", 14)
        self.tiny_font = pygame.font.SysFont("Arial", 12)
        
        # Scaling
        self.scale_x = self.MAP_WIDTH / (self.sim.env.x_max - self.sim.env.x_min)
        self.scale_y = self.MAP_HEIGHT / (self.sim.env.y_max - self.sim.env.y_min)
        
        # State
        self.selected_entity = None
        self.selected_ap = None
        self.dragging = False
        self.show_ftm_distances = False  # FTM: Toggle for showing distance lines
        self.show_distance_heatmap = False  # FTM: Toggle for distance heatmap overlay
        self.show_interference_hotspots = False  # FTM: Toggle for interference hotspot heatmap
        self.show_associations = True
        self.paused = False
        
        # Interactive creation modes
        self.add_ap_mode = False
        self.add_ftm_client_mode = False
        self.add_non_ftm_client_mode = False
        self.next_ap_id = len(simulation.access_points)
        self.next_client_id = len(simulation.clients)
        
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
    
    def draw_sidebar(self):
        """Draw the informative sidebar."""
        # Background
        sidebar_rect = pygame.Rect(self.MAP_WIDTH, 0, self.SIDEBAR_WIDTH, self.height)
        pygame.draw.rect(self.screen, self.SIDEBAR_BG, sidebar_rect)
        pygame.draw.line(self.screen, (50, 60, 80), (self.MAP_WIDTH, 0), (self.MAP_WIDTH, self.height), 2)
        
        x_start = self.MAP_WIDTH + 20
        y = 20
        
        # Title
        title = self.title_font.render("FTM RTT Simulation", True, self.ACCENT_COLOR)
        self.screen.blit(title, (x_start, y))
        y += 40
        
        # FTM Statistics
        font = self.header_font
        self.screen.blit(font.render("FTM Statistics", True, self.ACCENT_COLOR), (x_start, y))
        y += 30
        
        ftm_capable_count = sum(1 for c in self.sim.clients if c.ftm_capable)
        ftm_percent = ftm_capable_count / len(self.sim.clients) * 100 if self.sim.clients else 0
        
        stats = [
            f"FTM Capable: {ftm_capable_count}/{len(self.sim.clients)} ({ftm_percent:.0f}%)",
            f"Total APs: {len(self.sim.access_points)}",
            f"Total Clients: {len(self.sim.clients)}",
            f"Step: {self.sim.step_count}"
        ]
        
        for stat in stats:
            text = self.font.render(stat, True, self.TEXT_COLOR)
            self.screen.blit(text, (x_start, y))
            y += 25
        
        y += 20
        pygame.draw.line(self.screen, (50, 60, 80), (x_start, y), (self.width - 20, y), 1)
        y += 20
        
        # Controls
        self.screen.blit(self.header_font.render("Controls", True, self.ACCENT_COLOR), (x_start, y))
        y += 30
        
        controls = [
            "SPACE: Pause/Resume",
            "A: Toggle Associations",
            "D: Toggle FTM Distance Lines",
            "H: Trilateration Heatmap",
            "   (select FTM client first)",
            "I: Toggle Interference Hotspots",
            "   (aggregated low SINR zones)",
            "X: Add AP Mode",
            "F: Add FTM Client Mode",
            "N: Add Non-FTM Client Mode",
            "S: Save Heatmap to File",
            "Click: Select/Place Entity",
            "ESC: Exit"
        ]
        
        for ctrl in controls:
            text = self.small_font.render(ctrl, True, self.TEXT_COLOR)
            self.screen.blit(text, (x_start, y))
            y += 22
        
        y += 20
        pygame.draw.line(self.screen, (50, 60, 80), (x_start, y), (self.width - 20, y), 1)
        y += 20
        
        # Selected Entity Details
        if self.selected_entity:
            if isinstance(self.selected_entity, AccessPoint):
                self.draw_ap_details(self.selected_entity, x_start, y)
            elif isinstance(self.selected_entity, Client):
                self.draw_client_details(self.selected_entity, x_start, y)
        else:
            msg = "Select an AP or Client"
            self.screen.blit(self.font.render(msg, True, (100, 110, 130)), (x_start, y))
    
    def draw_ap_details(self, ap: AccessPoint, x: int, y: int):
        """Draw AP details in sidebar."""
        self.screen.blit(self.header_font.render(f"Access Point {ap.id}", True, self.ACCENT_COLOR), (x, y))
        y += 30
        
        details = [
            f"Channel: {ap.channel}",
            f"Tx Power: {ap.tx_power:.1f} dBm",
            f"Clients: {len(ap.connected_clients)}",
            f"Position: ({ap.x:.1f}, {ap.y:.1f})"
        ]
        
        for detail in details:
            text = self.font.render(detail, True, self.TEXT_COLOR)
            self.screen.blit(text, (x, y))
            y += 25
    
    def draw_client_details(self, client: Client, x: int, y: int):
        """Draw client details in sidebar."""
        ftm_status = "✓ FTM-Capable" if client.ftm_capable else "✗ No FTM"
        title_color = (0, 255, 100) if client.ftm_capable else (255, 100, 100)
        
        self.screen.blit(self.header_font.render(f"Client {client.id}", True, self.ACCENT_COLOR), (x, y))
        y += 30
        self.screen.blit(self.font.render(ftm_status, True, title_color), (x, y))
        y += 30
        
        assoc_str = f"AP {client.associated_ap}" if client.associated_ap is not None else "None"
        
        details = [
            f"Associated: {assoc_str}",
            f"SINR: {client.sinr_db:.1f} dB",
            f"Position: ({client.x:.1f}, {client.y:.1f})"
        ]
        
        # Show FTM distances if available
        if client.ftm_capable and client.measured_distances:
            details.append("")
            details.append("FTM Distances:")
            for ap_id, dist in sorted(client.measured_distances.items()):
                details.append(f"  AP {ap_id}: {dist:.1f}m")
        
        for detail in details:
            text = self.font.render(detail, True, self.TEXT_COLOR)
            self.screen.blit(text, (x, y))
            y += 22
    
    def draw_access_points(self):
        """Draw APs on map."""
        for ap in self.sim.access_points:
            pos = self.world_to_screen(ap.x, ap.y)
            
            # Selection highlight
            if self.selected_entity == ap:
                pygame.draw.circle(self.screen, self.SELECTION_COLOR, pos, 16, 2)
            
            # Draw AP
            pygame.draw.circle(self.screen, self.AP_COLOR, pos, 12)
            pygame.draw.circle(self.screen, (255, 255, 255), pos, 8)
            
            # ID
            id_text = self.small_font.render(str(ap.id), True, (20, 20, 20))
            text_rect = id_text.get_rect(center=pos)
            self.screen.blit(id_text, text_rect)
    
    def draw_clients(self):
        """Draw clients on map with FTM indication."""
        for client in self.sim.clients:
            pos = self.world_to_screen(client.x, client.y)
            
            # Color based on SINR
            if client.sinr_db > 20:
                color = self.CLIENT_COLORS[0]
            elif client.sinr_db > 10:
                color = self.CLIENT_COLORS[1]
            else:
                color = self.CLIENT_COLORS[2]
            
            # Selection highlight
            if self.selected_entity == client:
                pygame.draw.circle(self.screen, self.SELECTION_COLOR, pos, 10, 2)
            
            # Draw client differently based on FTM capability
            if client.ftm_capable:
                # FTM-capable: filled circle with blue indicator
                pygame.draw.circle(self.screen, color, pos, 7)
                pygame.draw.circle(self.screen, self.FTM_INDICATOR_COLOR, pos, 3)
            else:
                # Non-FTM: hollow circle
                pygame.draw.circle(self.screen, color, pos, 7, 2)
    
    def draw_association_lines(self):
        """Draw client-AP association lines."""
        if not self.show_associations:
            return
        
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        for client in self.sim.clients:
            if client.associated_ap is None:
                continue
            
            ap = next((ap for ap in self.sim.access_points if ap.id == client.associated_ap), None)
            if ap is None:
                continue
            
            client_pos = self.world_to_screen(client.x, client.y)
            ap_pos = self.world_to_screen(ap.x, ap.y)
            
            pygame.draw.line(surface, (148, 163, 184, 80), client_pos, ap_pos, 1)
        
        self.screen.blit(surface, (0, 0))
    
    def draw_ftm_distance_lines(self):
        """Draw FTM distance measurement lines."""
        if not self.show_ftm_distances:
            return
        
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        for client in self.sim.clients:
            if not client.ftm_capable or not client.measured_distances:
                continue
            
            client_pos = self.world_to_screen(client.x, client.y)
            
            for ap_id, distance in client.measured_distances.items():
                ap = next((ap for ap in self.sim.access_points if ap.id == ap_id), None)
                if ap is None:
                    continue
                
                ap_pos = self.world_to_screen(ap.x, ap.y)
                
                # Draw semi-transparent line
                pygame.draw.line(surface, self.DISTANCE_LINE_COLOR, ap_pos, client_pos, 1)
                
                # Draw distance value at midpoint
                mx, my = (ap_pos[0] + client_pos[0]) // 2, (ap_pos[1] + client_pos[1]) // 2
                dist_text = self.tiny_font.render(f'{distance:.1f}m', True, (0, 0, 200))
                surface.blit(dist_text, (mx, my))
        
        self.screen.blit(surface, (0, 0))
    
    def draw_distance_heatmap(self):
        """Draw trilateration heatmap showing positioning from all APs combined."""
        if not self.show_distance_heatmap:
            return
        
        # Need a selected FTM client to show positioning
        if not isinstance(self.selected_entity, Client) or not self.selected_entity.ftm_capable:
            return
        
        client = self.selected_entity
        
        # Get measured distances from all APs
        if not client.measured_distances:
            return
        
        # Create overlay surface
        overlay = pygame.Surface((self.MAP_WIDTH, self.MAP_HEIGHT), pygame.SRCALPHA)
        
        # Grid resolution
        step = 10
        
        # For each grid point, calculate positioning error
        for x in range(0, self.MAP_WIDTH, step):
            for y in range(0, self.MAP_HEIGHT, step):
                world_x, world_y = self.screen_to_world(x, y)
                
                # Calculate total positioning error for this point
                total_error = 0
                count = 0
                
                for ap_id, measured_dist in client.measured_distances.items():
                    ap = next((a for a in self.sim.access_points if a.id == ap_id), None)
                    if ap is None:
                        continue
                    
                    # Calculate actual distance from this grid point to AP
                    actual_dist = math.sqrt((world_x - ap.x)**2 + (world_y - ap.y)**2)
                    
                    # Error = difference between measured and actual distance
                    error = abs(actual_dist - measured_dist)
                    total_error += error
                    count += 1
                
                if count == 0:
                    continue
                
                # Average error across all APs
                avg_error = total_error / count
                
                # Map error to color
                # Low error (good match) = green, high error (poor match) = red
                max_error = 15  # meters
                error_ratio = min(avg_error / max_error, 1.0)
                
                # Green to red gradient
                red = int(255 * error_ratio)
                green = int(255 * (1 - error_ratio))
                blue = 0
                alpha = 140
                
                color = (red, green, blue, alpha)
                pygame.draw.rect(overlay, color, (x, y, step, step))
        
        self.screen.blit(overlay, (0, 0))
        
        # Draw distance circles from each AP to show trilateration visually
        surface = pygame.Surface((self.MAP_WIDTH, self.MAP_HEIGHT), pygame.SRCALPHA)
        
        for ap_id, measured_dist in client.measured_distances.items():
            ap = next((a for a in self.sim.access_points if a.id == ap_id), None)
            if ap is None:
                continue
            
            # Draw circle representing measured distance
            ap_pos = self.world_to_screen(ap.x, ap.y)
            radius_pixels = int(measured_dist * self.scale_x)
            
            # Semi-transparent circle outline
            pygame.draw.circle(surface, (255, 255, 255, 80), ap_pos, radius_pixels, 2)
        
        self.screen.blit(surface, (0, 0))
        
        # Draw actual client position for comparison
        client_pos = self.world_to_screen(client.x, client.y)
        pygame.draw.circle(self.screen, (255, 255, 0), client_pos, 10, 3)
    
    def draw_interference_hotspots(self):
        """Draw interference hotspots based on aggregated client reports."""
        if not self.show_interference_hotspots:
            return
        
        reports = self.sim.interference_reports
        if not reports:
            return
        
        # Create overlay surface
        overlay = pygame.Surface((self.MAP_WIDTH, self.MAP_HEIGHT), pygame.SRCALPHA)
        
        # Grid resolution
        step = 10
        
        # Aggregate reports into grid cells
        grid_sinr = {}  # (grid_x, grid_y) -> [sinr_values]
        
        for r_x, r_y, sinr in reports:
            # Map report position to grid cell
            screen_x, screen_y = self.world_to_screen(r_x, r_y)
            grid_x = (screen_x // step) * step
            grid_y = (screen_y // step) * step
            
            if (grid_x, grid_y) not in grid_sinr:
                grid_sinr[(grid_x, grid_y)] = []
            grid_sinr[(grid_x, grid_y)].append(sinr)
        
        # Draw grid cells
        for (gx, gy), sinr_list in grid_sinr.items():
            if not sinr_list:
                continue
            
            avg_sinr = sum(sinr_list) / len(sinr_list)
            
            # Map SINR to color
            # High SINR (>20) = Transparent/Green
            # Low SINR (<10) = Red (Interference Hotspot)
            
            if avg_sinr > 20:
                continue  # Don't draw good areas to keep map clean
            
            # Calculate intensity of interference (0 to 1)
            # SINR 20 -> 0 (No interference shown)
            # SINR 0 -> 1 (Max interference shown)
            intensity = max(0.0, min(1.0, (20 - avg_sinr) / 20.0))
            
            red = 255
            green = int(255 * (1 - intensity))
            blue = 0
            alpha = int(150 * intensity)  # More intense = more opaque
            
            color = (red, green, blue, alpha)
            pygame.draw.rect(overlay, color, (gx, gy, step, step))
            
        self.screen.blit(overlay, (0, 0))
    
    def save_heatmap_to_file(self):
        """Save current trilateration heatmap to a PNG file."""
        if not isinstance(self.selected_entity, Client) or not self.selected_entity.ftm_capable:
            print("Please select an FTM-capable client first to save trilateration heatmap!")
            return
        
        import pygame
        
        client = self.selected_entity
        
        if not client.measured_distances:
            print("No FTM distance measurements available for this client!")
            return
        
        # Create a surface for the heatmap
        heatmap_surface = pygame.Surface((self.MAP_WIDTH, self.MAP_HEIGHT))
        heatmap_surface.fill(self.BG_COLOR)
        
        # Grid resolution - higher for saved image
        step = 3
        max_error = 15
        
        # Calculate trilateration heatmap
        for x in range(0, self.MAP_WIDTH, step):
            for y in range(0, self.MAP_HEIGHT, step):
                world_x, world_y = self.screen_to_world(x, y)
                
                total_error = 0
                count = 0
                
                for ap_id, measured_dist in client.measured_distances.items():
                    ap = next((a for a in self.sim.access_points if a.id == ap_id), None)
                    if ap is None:
                        continue
                    
                    actual_dist = math.sqrt((world_x - ap.x)**2 + (world_y - ap.y)**2)
                    error = abs(actual_dist - measured_dist)
                    total_error += error
                    count += 1
                
                if count == 0:
                    continue
                
                avg_error = total_error / count
                error_ratio = min(avg_error / max_error, 1.0)
                
                red = int(255 * error_ratio)
                green = int(255 * (1 - error_ratio))
                blue = 0
                
                color = (red, green, blue)
                pygame.draw.rect(heatmap_surface, color, (x, y, step, step))
        
        # Draw distance circles from each AP
        for ap_id, measured_dist in client.measured_distances.items():
            ap = next((a for a in self.sim.access_points if a.id == ap_id), None)
            if ap is None:
                continue
            
            ap_pos = self.world_to_screen(ap.x, ap.y)
            radius_pixels = int(measured_dist * self.scale_x)
            pygame.draw.circle(heatmap_surface, (255, 255, 255), ap_pos, radius_pixels, 2)
            
            # Label AP
            label = self.small_font.render(f'AP{ap.id}', True, (255, 255, 255))
            heatmap_surface.blit(label, (ap_pos[0] + 15, ap_pos[1] - 10))
        
        # Draw all APs
        for ap in self.sim.access_points:
            pos = self.world_to_screen(ap.x, ap.y)
            pygame.draw.circle(heatmap_surface, self.AP_COLOR, pos, 12)
            pygame.draw.circle(heatmap_surface, (255, 255, 255), pos, 8)
        
        # Draw actual client position (yellow circle)
        client_pos = self.world_to_screen(client.x, client.y)
        pygame.draw.circle(heatmap_surface, (255, 255, 0), client_pos, 12, 3)
        
        # Save to file
        filename = f"trilateration_client{client.id}_step{self.sim.step_count}.png"
        pygame.image.save(heatmap_surface, filename)
        print(f"✓ Saved trilateration heatmap to: {filename}")
    
    def handle_events(self) -> bool:
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            self.ui_manager.process_events(event)
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_a:
                    self.show_associations = not self.show_associations
                elif event.key == pygame.K_d:
                    self.show_ftm_distances = not self.show_ftm_distances
                elif event.key == pygame.K_h:
                    self.show_distance_heatmap = not self.show_distance_heatmap
                elif event.key == pygame.K_i:
                    self.show_interference_hotspots = not self.show_interference_hotspots
                    print(f"Interference Hotspots: {'ON' if self.show_interference_hotspots else 'OFF'}")
                elif event.key == pygame.K_x:
                    # Toggle AP creation mode
                    self.add_ap_mode = not self.add_ap_mode
                    self.add_ftm_client_mode = False
                    self.add_non_ftm_client_mode = False
                    mode = "ON" if self.add_ap_mode else "OFF"
                    print(f"Add AP Mode: {mode}")
                elif event.key == pygame.K_f:
                    # Toggle FTM client creation mode
                    self.add_ftm_client_mode = not self.add_ftm_client_mode
                    self.add_ap_mode = False
                    self.add_non_ftm_client_mode = False
                    mode = "ON" if self.add_ftm_client_mode else "OFF"
                    print(f"Add FTM Client Mode: {mode}")
                elif event.key == pygame.K_n:
                    # Toggle Non-FTM client creation mode
                    self.add_non_ftm_client_mode = not self.add_non_ftm_client_mode
                    self.add_ap_mode = False
                    self.add_ftm_client_mode = False
                    mode = "ON" if self.add_non_ftm_client_mode else "OFF"
                    print(f"Add Non-FTM Client Mode: {mode}")
                elif event.key == pygame.K_s:
                    # Save heatmap to file
                    self.save_heatmap_to_file()
            
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Left click
                mx, my = event.pos
                if mx < self.MAP_WIDTH:  # Click on map
                    if self.add_ap_mode or self.add_ftm_client_mode or self.add_non_ftm_client_mode:
                        self.add_entity_at_position(mx, my)
                    else:
                        self.select_entity(mx, my)
        
        return True
    
    def add_entity_at_position(self, mx: int, my: int):
        """Add AP or client at clicked position."""
        import random
        from ftm_datatype import AccessPoint, Client
        
        world_x, world_y = self.screen_to_world(mx, my)
        
        if self.add_ap_mode:
            # Add new AP
            new_ap = AccessPoint(
                id=self.next_ap_id,
                x=world_x,
                y=world_y,
                tx_power=random.uniform(20, 25),
                channel=random.choice([1, 6, 11]),
                bandwidth=20,
                max_throughput=150.0
            )
            self.sim.add_access_point(new_ap)
            print(f"Added AP {self.next_ap_id} at ({world_x:.1f}, {world_y:.1f})")
            self.next_ap_id += 1
            self.add_ap_mode = False  # Auto-disable after adding
            
        elif self.add_ftm_client_mode:
            # Add FTM-capable client
            new_client = Client(
                id=self.next_client_id,
                x=world_x,
                y=world_y,
                demand_mbps=random.uniform(5, 30),
                velocity=random.uniform(0.5, 2.0),
                ftm_capable=True  # FTM-capable
            )
            self.sim.add_client(new_client)
            print(f"Added FTM Client {self.next_client_id} at ({world_x:.1f}, {world_y:.1f})")
            self.next_client_id += 1
            self.add_ftm_client_mode = False  # Auto-disable after adding
            
        elif self.add_non_ftm_client_mode:
            # Add non-FTM client
            new_client = Client(
                id=self.next_client_id,
                x=world_x,
                y=world_y,
                demand_mbps=random.uniform(5, 30),
                velocity=random.uniform(0.5, 2.0),
                ftm_capable=False  # Non-FTM
            )
            self.sim.add_client(new_client)
            print(f"Added Non-FTM Client {self.next_client_id} at ({world_x:.1f}, {world_y:.1f})")
            self.next_client_id += 1
            self.add_non_ftm_client_mode = False  # Auto-disable after adding
        
        return True
    
    def select_entity(self, mx: int, my: int):
        """Select AP or client at mouse position."""
        # Check APs first
        for ap in self.sim.access_points:
            pos = self.world_to_screen(ap.x, ap.y)
            dist = math.sqrt((mx - pos[0])**2 + (my - pos[1])**2)
            if dist < 15:
                self.selected_entity = ap
                return
        
        # Check clients
        for client in self.sim.clients:
            pos = self.world_to_screen(client.x, client.y)
            dist = math.sqrt((mx - pos[0])**2 + (my - pos[1])**2)
            if dist < 10:
                self.selected_entity = client
                return
        
        # No entity selected
        self.selected_entity = None
    
    def run(self, max_steps: Optional[int] = None, fps: int = 10):
        """Run the visualization loop."""
        clock = pygame.time.Clock()
        running = True
        step_count = 0
        
        while running:
            time_delta = clock.tick(fps) / 1000.0
            
            # Handle events
            running = self.handle_events()
            
            # Update simulation
            if not self.paused:
                self.sim.step()
                step_count += 1
                if max_steps is not None and step_count >= max_steps:
                    running = False
            
            # Draw everything
            self.screen.fill(self.BG_COLOR)
            
            # Draw map elements
            self.draw_association_lines()
            self.draw_distance_heatmap()  # FTM: Heatmap overlay
            self.draw_interference_hotspots()  # FTM: Interference Hotspots
            self.draw_access_points()
            self.draw_clients()
            self.draw_ftm_distance_lines()  # FTM: Distance lines
            
            # Draw sidebar
            self.draw_sidebar()
            
            # Update UI
            self.ui_manager.update(time_delta)
            self.ui_manager.draw_ui(self.screen)
            
            # Flip display
            pygame.display.flip()
        
        pygame.quit()
