"""
Generate large, diverse training dataset for GNN.
Creates multiple simulation scenarios with varying:
- Number of APs (3-7)
- Number of clients (10-50)
- AP topologies (grid, random, clustered)
- Channels and power levels
"""

from ftm_assoc import *
from ftm_datatype import *
from ftm_metrics import *
from ftm_sim import *
from ftm_utils import *
import numpy as np


def create_grid_topology(num_aps: int, env: Environment) -> List[Tuple[float, float]]:
    """Create grid-based AP positions."""
    grid_size = int(np.ceil(np.sqrt(num_aps)))
    positions = []
    
    x_spacing = (env.x_max - env.x_min - 10) / (grid_size + 1)
    y_spacing = (env.y_max - env.y_min - 10) / (grid_size + 1)
    
    for i in range(num_aps):
        row = i // grid_size
        col = i % grid_size
        x = env.x_min + 5 + (col + 1) * x_spacing
        y = env.y_min + 5 + (row + 1) * y_spacing
        positions.append((x, y))
    
    return positions


def create_random_topology(num_aps: int, env: Environment) -> List[Tuple[float, float]]:
    """Create random AP positions."""
    positions = []
    for _ in range(num_aps):
        x = random.uniform(env.x_min + 5, env.x_max - 5)
        y = random.uniform(env.y_min + 5, env.y_max - 5)
        positions.append((x, y))
    return positions


def create_clustered_topology(num_aps: int, env: Environment) -> List[Tuple[float, float]]:
    """Create clustered AP positions (APs grouped in hotspots)."""
    num_clusters = max(2, num_aps // 3)
    cluster_centers = []
    
    # Generate cluster centers
    for _ in range(num_clusters):
        cx = random.uniform(env.x_min + 10, env.x_max - 10)
        cy = random.uniform(env.y_min + 10, env.y_max - 10)
        cluster_centers.append((cx, cy))
    
    # Assign APs to clusters
    positions = []
    for i in range(num_aps):
        cluster_idx = i % num_clusters
        cx, cy = cluster_centers[cluster_idx]
        
        # Add some spread around cluster center
        x = cx + random.uniform(-8, 8)
        y = cy + random.uniform(-8, 8)
        
        # Clip to bounds
        x = max(env.x_min + 5, min(x, env.x_max - 5))
        y = max(env.y_min + 5, min(y, env.y_max - 5))
        
        positions.append((x, y))
    
    return positions


def create_linear_topology(num_aps: int, env: Environment) -> List[Tuple[float, float]]:
    """Create linear AP positions (corridor scenario)."""
    positions = []
    spacing = (env.x_max - env.x_min - 10) / (num_aps + 1)
    y_center = (env.y_min + env.y_max) / 2
    
    for i in range(num_aps):
        x = env.x_min + 5 + (i + 1) * spacing
        y = y_center + random.uniform(-5, 5)  # Small y variation
        positions.append((x, y))
    
    return positions


def run_scenario(scenario_id: int, 
                num_aps: int,
                num_clients: int,
                topology_type: str,
                num_steps: int,
                log_dir: str = "logs") -> None:
    """
    Run a single simulation scenario.
    
    Args:
        scenario_id: Unique scenario identifier
        num_aps: Number of access points
        num_clients: Number of clients
        topology_type: Type of topology ('grid', 'random', 'clustered', 'linear')
        num_steps: Number of simulation steps
        log_dir: Directory for logs
    """
    print(f"\n{'='*70}")
    print(f"Scenario {scenario_id}: {num_aps} APs, {num_clients} clients, {topology_type} topology")
    print(f"{'='*70}")
    
    # Environment
    env = Environment(x_min=0, x_max=50, y_min=0, y_max=50)
    
    # Propagation model
    base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=3.0)
    fading_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
    
    # Create simulation
    sim = WirelessSimulation(
        env, 
        fading_model, 
        interference_threshold_dbm=-75.0, 
        enable_logging=True,
        log_dir=log_dir
    )
    
    # Generate AP topology
    if topology_type == 'grid':
        ap_positions = create_grid_topology(num_aps, env)
    elif topology_type == 'random':
        ap_positions = create_random_topology(num_aps, env)
    elif topology_type == 'clustered':
        ap_positions = create_clustered_topology(num_aps, env)
    elif topology_type == 'linear':
        ap_positions = create_linear_topology(num_aps, env)
    else:
        ap_positions = create_random_topology(num_aps, env)
    
    # Add APs with varied configurations
    channels = [1, 6, 11]  # Common non-overlapping channels
    for i, (x, y) in enumerate(ap_positions):
        channel = channels[i % len(channels)]
        tx_power = random.uniform(20, 30)  # Vary transmit power
        
        sim.add_access_point(AccessPoint(
            id=i, 
            x=x, 
            y=y, 
            tx_power=tx_power, 
            channel=channel,
            bandwidth=20, 
            max_throughput=150.0
        ))
    
    # Add clients with random positions and demands
    for i in range(num_clients):
        x = random.uniform(env.x_min + 5, env.x_max - 5)
        y = random.uniform(env.y_min + 5, env.y_max - 5)
        demand_mbps = random.uniform(5, 35)
        velocity = random.uniform(0.5, 2.0)  # Varied mobility
        
        sim.add_client(Client(
            id=i, 
            x=x, 
            y=y, 
            demand_mbps=demand_mbps,
            velocity=velocity
        ))
    
    # Initialize
    sim.initialize()
    
    # Run simulation
    print(f"Running {num_steps} steps... ", end="", flush=True)
    for step in range(num_steps):
        sim.step()
        if (step + 1) % 25 == 0:
            print(f"{step+1}...", end=" ", flush=True)
    print("Done!")
    
    # Print stats
    if sim.logger:
        print(f"Logs saved to {log_dir}/")


def generate_diverse_dataset(
    num_scenarios: int = 20,
    steps_per_scenario: int = 100,
    log_dir: str = "logs"
) -> None:
    """
    Generate diverse dataset with multiple scenarios.
    
    Args:
        num_scenarios: Number of different scenarios to simulate
        steps_per_scenario: Number of timesteps per scenario
        log_dir: Directory for logs
    """
    print("="*70)
    print(f"GENERATING DIVERSE GNN TRAINING DATASET")
    print(f"  Scenarios: {num_scenarios}")
    print(f"  Steps per scenario: {steps_per_scenario}")
    print(f"  Total timesteps: {num_scenarios * steps_per_scenario}")
    print("="*70)
    
    topologies = ['grid', 'random', 'clustered', 'linear']
    
    for i in range(num_scenarios):
        # Vary parameters
        num_aps = random.randint(3, 7)
        num_clients = random.randint(10, 50)
        topology = topologies[i % len(topologies)]
        
        # Run scenario
        run_scenario(
            scenario_id=i + 1,
            num_aps=num_aps,
            num_clients=num_clients,
            topology_type=topology,
            num_steps=steps_per_scenario,
            log_dir=log_dir
        )
    
    print("\n" + "="*70)
    print("DATASET GENERATION COMPLETE!")
    print("="*70)
    print(f"\nTotal scenarios: {num_scenarios}")
    print(f"Total timesteps: {num_scenarios * steps_per_scenario}")
    print(f"\nVariations:")
    print(f"  APs: 3-7")
    print(f"  Clients: 10-50")
    print(f"  Topologies: grid, random, clustered, linear")
    print(f"  Channels: 1, 6, 11 (varied)")
    print(f"  Tx Power: 20-30 dBm (varied)")
    print(f"\nLogs saved to: {log_dir}/")
    print("\nNext steps:")
    print("  1. Run: python train_gnn.py")
    print("  2. Then: python evaluate_gnn.py")
    print("="*70)


if __name__ == "__main__":
    # Configuration options
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate diverse GNN training data')
    parser.add_argument('--scenarios', type=int, default=20, 
                       help='Number of different scenarios (default: 20)')
    parser.add_argument('--steps', type=int, default=100,
                       help='Steps per scenario (default: 100)')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Log directory (default: logs)')
    
    args = parser.parse_args()
    
    # Generate dataset
    generate_diverse_dataset(
        num_scenarios=args.scenarios,
        steps_per_scenario=args.steps,
        log_dir=args.log_dir
    )
