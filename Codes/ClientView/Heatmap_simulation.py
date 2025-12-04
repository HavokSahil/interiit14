"""
FTM RTT Distance Heatmap Simulation

Simulates 802.11mc FTM (Fine Timing Measurement) RTT capability where APs can
measure the distance to all clients within range, regardless of association status.
Generates heatmap visualizations showing distance relationships.
"""

import sys
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from typing import List, Tuple

# Add parent directory to path to import from multi-timescale-controller
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'multi-timescale-controller'))

from datatype import AccessPoint, Client, Environment
from model import PathLossModel, MultipathFadingModel
from generate_training_data import create_grid_topology, create_random_topology


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def simulate_ftm_rtt_distance(actual_distance: float, noise_std: float = 0.5) -> float:
    """
    Simulate FTM RTT distance measurement with realistic noise.
    
    Args:
        actual_distance: True Euclidean distance
        noise_std: Standard deviation of measurement noise in meters
    
    Returns:
        Measured distance with added noise
    """
    # Add Gaussian noise to simulate measurement uncertainty
    noise = random.gauss(0, noise_std)
    measured_distance = max(0.1, actual_distance + noise)  # Ensure positive distance
    return measured_distance


def create_distance_matrix(aps: List[AccessPoint], clients: List[Client], 
                          use_ftm_noise: bool = True) -> np.ndarray:
    """
    Create a distance matrix showing distance from each AP to each client.
    
    Args:
        aps: List of access points
        clients: List of clients
        use_ftm_noise: Whether to add FTM measurement noise
    
    Returns:
        2D numpy array of shape (num_aps, num_clients) with distances
    """
    distance_matrix = np.zeros((len(aps), len(clients)))
    
    for i, ap in enumerate(aps):
        for j, client in enumerate(clients):
            actual_dist = calculate_distance(ap.x, ap.y, client.x, client.y)
            if use_ftm_noise:
                distance_matrix[i, j] = simulate_ftm_rtt_distance(actual_dist)
            else:
                distance_matrix[i, j] = actual_dist
    
    return distance_matrix


def plot_per_ap_distance_heatmap(aps: List[AccessPoint], clients: List[Client], 
                                 env: Environment, prop_model: PathLossModel):
    """
    Generate a heatmap showing distance gradients from each AP.
    Each subplot shows one AP's distance field.
    """
    num_aps = len(aps)
    cols = min(3, num_aps)
    rows = (num_aps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if num_aps == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if num_aps > 1 else [axes]
    
    # Create grid for heatmap
    x_grid = np.linspace(env.x_min, env.x_max, 100)
    y_grid = np.linspace(env.y_min, env.y_max, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    for i, ap in enumerate(aps):
        ax = axes[i]
        
        # Calculate distance from this AP to every point in the grid
        distances = np.sqrt((X - ap.x)**2 + (Y - ap.y)**2)
        
        # Create contour plot
        contour = ax.contourf(X, Y, distances, levels=20, cmap='RdYlGn_r', alpha=0.7)
        plt.colorbar(contour, ax=ax, label='Distance (m)')
        
        # Draw the AP
        ax.plot(ap.x, ap.y, 'b^', markersize=15, label=f'AP {ap.id}', 
                markeredgecolor='black', markeredgewidth=2)
        
        # Draw all clients
        for client in clients:
            dist = calculate_distance(ap.x, ap.y, client.x, client.y)
            ax.plot(client.x, client.y, 'ro', markersize=8, alpha=0.7)
            # Annotate distance
            ax.annotate(f'{dist:.1f}m', (client.x, client.y), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.8)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'AP {ap.id} Distance Field\n(Ch {ap.channel}, Tx {ap.tx_power:.1f} dBm)')
        ax.set_xlim(env.x_min, env.x_max)
        ax.set_ylim(env.y_min, env.y_max)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    # Hide unused subplots
    for i in range(num_aps, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('ap_distance_heatmaps.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: ap_distance_heatmaps.png")
    return fig


def plot_closest_ap_heatmap(aps: List[AccessPoint], clients: List[Client], env: Environment):
    """
    Generate a heatmap showing which AP is closest at each point (Voronoi-like).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create grid
    x_grid = np.linspace(env.x_min, env.x_max, 200)
    y_grid = np.linspace(env.y_min, env.y_max, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # For each grid point, find the closest AP
    closest_ap_map = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x, y = X[i, j], Y[i, j]
            min_dist = float('inf')
            closest_ap = 0
            
            for ap_idx, ap in enumerate(aps):
                dist = calculate_distance(x, y, ap.x, ap.y)
                if dist < min_dist:
                    min_dist = dist
                    closest_ap = ap_idx
            
            closest_ap_map[i, j] = closest_ap
    
    # Plot Voronoi regions
    cmap = plt.cm.get_cmap('tab10', len(aps))
    im = ax.contourf(X, Y, closest_ap_map, levels=len(aps), cmap=cmap, alpha=0.4)
    
    # Draw APs
    for ap in aps:
        ax.plot(ap.x, ap.y, 'b^', markersize=15, 
                markeredgecolor='black', markeredgewidth=2)
        ax.annotate(f'AP {ap.id}', (ap.x, ap.y), 
                   xytext=(5, -15), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # Draw clients
    for client in clients:
        ax.plot(client.x, client.y, 'ro', markersize=10, alpha=0.8,
                markeredgecolor='darkred', markeredgewidth=1.5)
        ax.annotate(f'C{client.id}', (client.x, client.y),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, color='darkred')
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('Closest AP Regions (Voronoi Diagram)\nBased on FTM RTT Distance', fontsize=14)
    ax.set_xlim(env.x_min, env.x_max)
    ax.set_ylim(env.y_min, env.y_max)
    ax.grid(True, alpha=0.3)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', label='Access Points'),
        Patch(facecolor='red', edgecolor='darkred', label='Clients')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('closest_ap_heatmap.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: closest_ap_heatmap.png")
    return fig


def plot_rssi_heatmap(aps: List[AccessPoint], clients: List[Client], 
                     env: Environment, prop_model: PathLossModel):
    """
    Generate a heatmap showing RSSI (Received Signal Strength) based on distance.
    """
    num_aps = len(aps)
    cols = min(3, num_aps)
    rows = (num_aps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if num_aps == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if num_aps > 1 else [axes]
    
    # Create grid
    x_grid = np.linspace(env.x_min, env.x_max, 100)
    y_grid = np.linspace(env.y_min, env.y_max, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    for i, ap in enumerate(aps):
        ax = axes[i]
        
        # Calculate RSSI from this AP to every point
        distances = np.sqrt((X - ap.x)**2 + (Y - ap.y)**2)
        rssi = np.zeros_like(distances)
        
        for row in range(distances.shape[0]):
            for col in range(distances.shape[1]):
                rssi[row, col] = prop_model.compute_received_power(ap.tx_power, distances[row, col])
        
        # Create contour plot
        levels = np.linspace(-90, -30, 20)
        contour = ax.contourf(X, Y, rssi, levels=levels, cmap='RdYlGn', alpha=0.7)
        plt.colorbar(contour, ax=ax, label='RSSI (dBm)')
        
        # Draw contour lines at key RSSI thresholds
        ax.contour(X, Y, rssi, levels=[-80, -70, -60, -50], colors='black', 
                  linewidths=1, linestyles='dashed', alpha=0.5)
        
        # Draw the AP
        ax.plot(ap.x, ap.y, 'b^', markersize=15, 
                markeredgecolor='black', markeredgewidth=2)
        
        # Draw clients with RSSI values
        for client in clients:
            dist = calculate_distance(ap.x, ap.y, client.x, client.y)
            client_rssi = prop_model.compute_received_power(ap.tx_power, dist)
            ax.plot(client.x, client.y, 'ko', markersize=8)
            ax.annotate(f'{client_rssi:.1f}dBm', (client.x, client.y),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, bbox=dict(boxstyle='round,pad=0.2', 
                                            facecolor='white', alpha=0.8))
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'AP {ap.id} RSSI Field\n(Ch {ap.channel}, Tx {ap.tx_power:.1f} dBm)')
        ax.set_xlim(env.x_min, env.x_max)
        ax.set_ylim(env.y_min, env.y_max)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_aps, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('rssi_heatmaps.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: rssi_heatmaps.png")
    return fig


def plot_client_distance_matrix(distance_matrix: np.ndarray, aps: List[AccessPoint], 
                                clients: List[Client]):
    """
    Plot the AP-to-Client distance matrix as a heatmap table.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(distance_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(clients)))
    ax.set_yticks(np.arange(len(aps)))
    
    # Label with client and AP IDs
    ax.set_xticklabels([f'C{c.id}' for c in clients])
    ax.set_yticklabels([f'AP{ap.id}' for ap in aps])
    
    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add color bar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Distance (m)', rotation=270, labelpad=20, fontsize=12)
    
    # Annotate each cell with the distance value
    for i in range(len(aps)):
        for j in range(len(clients)):
            text = ax.text(j, i, f'{distance_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_xlabel('Clients', fontsize=12)
    ax.set_ylabel('Access Points', fontsize=12)
    ax.set_title('FTM RTT Distance Matrix\n(AP-to-Client Distances in meters)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('distance_matrix.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: distance_matrix.png")
    return fig


def main():
    """Main simulation and heatmap generation."""
    print("\n" + "="*60)
    print("FTM RTT Distance Heatmap Simulation")
    print("="*60 + "\n")
    
    # Create environment
    env = Environment(x_min=0, x_max=50, y_min=0, y_max=50)
    print(f"Environment: {env.x_max}m × {env.y_max}m")
    
    # Create propagation model (without fading for cleaner visualization)
    prop_model = PathLossModel(frequency_mhz=2400, path_loss_exp=3.5)
    print(f"Propagation: Path Loss Model (f={prop_model.freq} MHz, n={prop_model.n})")
    
    # Create APs in a grid pattern
    N_ap = 6
    ap_positions = create_grid_topology(N_ap, env)
    aps = []
    for i, (x, y) in enumerate(ap_positions):
        channel = random.choice([1, 6, 11])
        tx_power = random.uniform(20, 25)
        aps.append(AccessPoint(id=i, x=x, y=y, tx_power=tx_power, 
                              channel=channel, bandwidth=20, max_throughput=150.0))
    
    print(f"\nCreated {len(aps)} Access Points:")
    for ap in aps:
        print(f"  AP {ap.id}: Position=({ap.x:.1f}, {ap.y:.1f}), "
              f"Channel={ap.channel}, Tx Power={ap.tx_power:.1f} dBm")
    
    # Create clients at random positions
    N_client = 15
    client_positions = create_random_topology(N_client, env)
    clients = []
    for i, (x, y) in enumerate(client_positions):
        demand_mbps = random.uniform(5, 30)
        velocity = random.uniform(0.5, 2.0)
        clients.append(Client(id=i, x=x, y=y, demand_mbps=demand_mbps, velocity=velocity))
    
    print(f"\nCreated {len(clients)} Clients:")
    for client in clients[:5]:  # Show first 5
        print(f"  Client {client.id}: Position=({client.x:.1f}, {client.y:.1f})")
    if len(clients) > 5:
        print(f"  ... and {len(clients) - 5} more clients")
    
    # Calculate FTM RTT distance matrix
    print("\n" + "-"*60)
    print("Calculating FTM RTT Distances...")
    print("-"*60)
    distance_matrix = create_distance_matrix(aps, clients, use_ftm_noise=True)
    
    print("\nDistance Matrix (AP × Client):")
    print("AP/Client", end="  ")
    for client in clients:
        print(f"C{client.id:2d}", end="  ")
    print()
    
    for i, ap in enumerate(aps):
        print(f"AP{ap.id:2d}      ", end="  ")
        for j in range(len(clients)):
            print(f"{distance_matrix[i, j]:4.1f}", end=" ")
        print()
    
    print(f"\nDistance Statistics:")
    print(f"  Min Distance: {distance_matrix.min():.2f} m")
    print(f"  Max Distance: {distance_matrix.max():.2f} m")
    print(f"  Mean Distance: {distance_matrix.mean():.2f} m")
    
    # Find closest AP for each client
    print("\nClosest AP for each Client:")
    for j, client in enumerate(clients):
        closest_ap_idx = np.argmin(distance_matrix[:, j])
        closest_dist = distance_matrix[closest_ap_idx, j]
        print(f"  Client {client.id}: AP {aps[closest_ap_idx].id} "
              f"(distance: {closest_dist:.1f} m)")
    
    # Generate visualizations
    print("\n" + "-"*60)
    print("Generating Heatmap Visualizations...")
    print("-"*60 + "\n")
    
    plot_per_ap_distance_heatmap(aps, clients, env, prop_model)
    plot_closest_ap_heatmap(aps, clients, env)
    plot_rssi_heatmap(aps, clients, env, prop_model)
    plot_client_distance_matrix(distance_matrix, aps, clients)
    
    print("\n" + "="*60)
    print("✓ All heatmaps generated successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. ap_distance_heatmaps.png - Distance gradients for each AP")
    print("  2. closest_ap_heatmap.png   - Voronoi diagram of AP coverage")
    print("  3. rssi_heatmaps.png        - RSSI field for each AP")
    print("  4. distance_matrix.png      - AP-to-Client distance table")
    print("\nNote: These visualizations demonstrate the FTM RTT capability")
    print("      where APs can measure distance to ALL clients in range,")
    print("      not just those currently associated.")
    print()
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()
