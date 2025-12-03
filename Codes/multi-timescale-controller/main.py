from assoc import *
from datatype import *
from metrics import *
from sim import *
from utils import *
from generate_training_data import create_grid_topology, create_random_topology

def main():
    env = Environment(x_min=0, x_max=50, y_min=0, y_max=50)

    base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=5.0)
    fading_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
    
    sim = WirelessSimulation(env, fading_model, interference_threshold_dbm=-75.0, enable_logging=True)

    N_ap = 6
    ap_positions = create_random_topology(N_ap, env)
    for i, (x, y) in enumerate(ap_positions):
        channel = random.choice([1, 6, 11])
        tx_power = random.uniform(20, 30)
        sim.add_access_point(AccessPoint(id=i, x=x, y=y, tx_power=tx_power, channel=channel, 
                                         bandwidth=20, max_throughput=150.0))

    N_client = 25
    client_positions = create_random_topology(N_client, env)
    for i, (x, y) in enumerate(client_positions):
        demand_mbps = random.uniform(5, 30)
        velocity = random.uniform(0.5, 2.0)
        sim.add_client(Client(id=i, x=x, y=y, demand_mbps=demand_mbps, velocity=velocity))


    sim.initialize()

    USE_VISUALIZATION = True

    if USE_VISUALIZATION and PYGAME_AVAILABLE:
        print("Starting visualization... (Press ESC to exit)")
        print("\nVisualization Controls:")
        print("  SPACE: Pause/Resume")
        print("  I: Toggle interference display")
        print("  A: Toggle association lines")
        print("  C: Toggle coverage areas")
        print("  G: Toggle graph view")
        print("  P: Toggle predicted graph (if GNN available)")
        print("  Left Click: Select AP/Client")
        print("  Drag AP: Move AP")
        print("  ESC: Exit")
        print("\nClient Color Code:")
        print("  Green: SINR > 20 dB (Excellent)")
        print("  Yellow: SINR 10-20 dB (Good)")
        print("  Red: SINR < 10 dB (Poor)")
        print("\nClient Info: [SINR in dB] [Allocated/Demand Mbps] [Airtime %]\n")

        sim.enable_visualization(width=1920, height=1080)
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


if __name__ == "__main__":
    main()
