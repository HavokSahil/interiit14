"""
Quick test of A/B testing with visualization - 1000 steps only for testing.
"""

import time
import numpy as np
import pygame
from datatype import Environment, AccessPoint, Client
from model import PathLossModel, MultipathFadingModel
from sim import WirelessSimulation
from ab_testing import ABTestFramework, ABTestConfig
from qoe_monitor import QoEMonitor
from qoe_dashboard import QoEDashboard
from slo_catalog import SLOCatalog
from policy_engine import PolicyEngine
from enhanced_rrm_engine import EnhancedRRMEngine


def create_network():
    """Create test network topology"""
    # Environment
    env = Environment(x_min=0, x_max=200, y_min=0, y_max=200)
    
    # Propagation model
    base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=3.0)
    prop_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
    
    # Access Points
    aps = [
        AccessPoint(id=0, x=50, y=50, tx_power=25.0, channel=1),
        AccessPoint(id=1, x=150, y=50, tx_power=25.0, channel=6),
        AccessPoint(id=2, x=50, y=150, tx_power=25.0, channel=11),
        AccessPoint(id=3, x=150, y=150, tx_power=25.0, channel=1),
    ]
    
    # Clients (mix of roles)
    clients = []
    for i in range(20):
        x = np.random.uniform(10, 190)
        y = np.random.uniform(10, 190)
        demand_mbps = np.random.uniform(5, 20)
        clients.append(Client(id=i, x=x, y=y, demand_mbps=demand_mbps))
    
    return env, prop_model, aps, clients


def main():
    """Quick visualization test"""
    print("Testing visualization with Fast Loop enabled")
    print("===============================================")
    
    # Create network
    env, prop_model, aps, clients = create_network()
    
    # Create simulation
    sim = WirelessSimulation(env, prop_model, enable_logging=False)
    
    for ap in aps:
        sim.add_access_point(ap)
    for client in clients:
        sim.add_client(client)
    
    sim.initialize()
    
    # Enable visualization
    if sim.enable_visualization(width=1920, height=1080):
        print("✓ Visualization enabled!")
        print("\nControls:")
        print("  P - Pause/Resume")
        print("  ESC - Quit")
        print("  Click APs/Clients to see details")
    
    # Create SLO catalog and policy engine
    slo_catalog = SLOCatalog("slo_catalog.yml")
    policy_engine = PolicyEngine(slo_catalog, default_role="BE")
    
    # Create RRM engine (Fast Loop enabled)
    rrm = EnhancedRRMEngine(
        access_points=aps,
        clients=clients,
        interferers=[],
        prop_model=prop_model,
        slo_catalog_path="slo_catalog.yml",
        default_role="BE",
        fast_loop_period=60,
        slow_loop_period=300
    )
    
    # Create QoE monitor
    qoe_monitor = QoEMonitor(slo_catalog, policy_engine)
    
    # Run with visualization
    steps = 2000  # Run for 2000 steps
    print(f"\nRunning {steps} steps with visualization...")
    print("Watch the simulation window!")
    
    clock = pygame.time.Clock()
    running = True
    fps = 10
    actions_taken = 0
    
    target_step = sim.step_count + steps
    
    while running and sim.step_count < target_step:
        # Process events using visualizer's handler (handles UI buttons)
        running = sim.visualizer.handle_events()
        
        if not running:
            break
        
        if not sim.visualizer.paused:
            # Simulation step
            sim.step()
            current_step = sim.step_count
            
            # RRM
            result = rrm.execute(current_step)
            if result.get('fast_loop_actions'):
                actions_taken += len(result['fast_loop_actions'])
                print(f"Step {current_step}: Fast Loop took action!")
            
            # QoE monitoring
            if current_step % 10 == 0:
                qoe_monitor.update(current_step, sim.clients, sim.access_points)
            
            # Progress
            if current_step % 200 == 0:
                stats = qoe_monitor.get_network_qoe_stats()
                print(f"Step {current_step}/{target_step}: QoE={stats.mean:.4f}, Actions={actions_taken}")
        
        # Draw
        sim.visualizer.screen.fill((20, 20, 30))
        sim.visualizer.draw_coverage_areas()
        sim.visualizer.draw_association_lines()
        sim.visualizer.draw_access_points()
        sim.visualizer.draw_clients()
        sim.visualizer.draw_interferers()
        sim.visualizer.draw_sidebar()
        
        # Update GUI
        time_delta = clock.tick(fps) / 1000.0
        sim.visualizer.ui_manager.update(time_delta)
        sim.visualizer.ui_manager.draw_ui(sim.visualizer.screen)
        
        pygame.display.flip()
    
    pygame.quit()
    
    # Final stats
    final_stats = qoe_monitor.get_network_qoe_stats()
    print(f"\nCompleted!")
    print(f"Final QoE: {final_stats.mean:.4f}")
    print(f"Total Actions: {actions_taken}")


if __name__ == "__main__":
    main()
