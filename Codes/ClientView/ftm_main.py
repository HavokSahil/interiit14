"""
FTM RTT Simulation - Main Entry Point

Real-time pygame visualization of wireless network with FTM RTT distance measurement.
Demonstrates 802.11mc where APs can measure distance to all clients in range.
"""

import random
from ftm_datatype import AccessPoint, Client, Environment
from ftm_model import PathLossModel, MultipathFadingModel
from ftm_sim import FTMWirelessSimulation
from ftm_generate_training_data import create_grid_topology, create_random_topology


def main():
    """Main entry point for FTM RTT simulation."""
    
    # Create environment
    env = Environment(x_min=0, x_max=50, y_min=0, y_max=50)
    
    # Create propagation model
    base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=3.5)
    fading_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
    
    # Create simulation
    sim = FTMWirelessSimulation(env, fading_model, 
                                interference_threshold_dbm=-75.0,
                                ftm_measurement_noise_std=0.5)
    
    # Create APs in grid
    N_ap = 6
    ap_positions = create_grid_topology(N_ap, env)
    for i, (x, y) in enumerate(ap_positions):
        channel = random.choice([1, 6, 11])
        tx_power = random.uniform(20, 25)
        sim.add_access_point(AccessPoint(
            id=i, x=x, y=y, tx_power=tx_power, 
            channel=channel, bandwidth=20, max_throughput=150.0
        ))
    
    # Create clients with mixed FTM capability
    N_client = 20
    client_positions = create_random_topology(N_client, env)
    ftm_support_rate = 0.7  # 70% of clients support FTM
    
    for i, (x, y) in enumerate(client_positions):
        demand_mbps = random.uniform(5, 30)
        velocity = random.uniform(0.5, 2.0)
        ftm_capable = random.random() < ftm_support_rate
        
        sim.add_client(Client(
            id=i, x=x, y=y, 
            demand_mbps=demand_mbps, 
            velocity=velocity,
            ftm_capable=ftm_capable
        ))
    
    # Initialize simulation
    sim.initialize()
    
    # Print initial status
    print("\n" + "="*70)
    print("FTM RTT Wireless Simulation")
    print("="*70)
    sim.print_status()
    
    # Visualization controls
    USE_VISUALIZATION = True
    
    if USE_VISUALIZATION:
        try:
            import pygame
            PYGAME_AVAILABLE = True
        except ImportError:
            PYGAME_AVAILABLE = False
            USE_VISUALIZATION = False
    
    if USE_VISUALIZATION and PYGAME_AVAILABLE:
        print("\n" + "-"*70)
        print("Starting visualization... (Press ESC to exit)")
        print("-"*70)
        print("\nVisualization Controls:")
        print("  SPACE: Pause/Resume")
        print("  A: Toggle association lines")
        print("  D: Toggle FTM distance lines (shows distances from APs to FTM-capable clients)")
        print("  H: Toggle TRILATERATION heatmap (SELECT AN FTM CLIENT FIRST!)")
        print("     Shows positioning based on ALL APs working together")
        print("  I: Toggle INTERFERENCE HOTSPOTS (Aggregated from client reports)")
        print("     Red areas = High interference (Low SINR) reported by clients")
        print("  X: Add AP Mode (click to place new AP)")
        print("  F: Add FTM Client Mode (click to place FTM-capable client)")
        print("  N: Add Non-FTM Client Mode (click to place non-FTM client)")
        print("  S: Save trilateration heatmap to PNG file (select FTM client first)")
        print("  Left Click: Select AP/Client (or place entity in add mode)")
        print("  ESC: Exit")
        print("\nClient Visual Indicators:")
        print("  Filled circle with blue center: FTM-capable client")
        print("  Hollow circle: Non-FTM client")
        print("\nClient Color Code (based on SINR):")
        print("  Green: SINR > 20 dB (Excellent)")
        print("  Yellow: SINR 10-20 dB (Good)")
        print("  Red: SINR < 10 dB (Poor)")
        print()
        
        sim.enable_visualization(width=1280, height=720)
        sim.run_with_visualization(steps=None, fps=10)  # None = run indefinitely
    else:
        if USE_VISUALIZATION and not PYGAME_AVAILABLE:
            print("\nPygame not available. Install with: pip install pygame\n")
        
        # Console mode
        print("\n\n=== Running in console mode ===")
        print("Simulating 10 steps...\n")
        
        for step in range(10):
            sim.step()
            if step % 5 == 4:
                print(f"\n--- Step {sim.step_count} ---")
                # Show FTM measurement example
                ftm_clients = [c for c in sim.clients if c.ftm_capable]
                if ftm_clients:
                    client = ftm_clients[0]
                    print(f"Client {client.id} (FTM-capable) distances:")
                    for ap_id, dist in client.measured_distances.items():
                        print(f"  to AP {ap_id}: {dist:.2f}m")
        
        print("\n" + "="*70)
        print("Simulation complete!")
        print("="*70)


if __name__ == "__main__":
    main()
