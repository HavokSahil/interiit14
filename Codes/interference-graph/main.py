from assoc import *
from datatype import *
from metrics import *
from sim import *
from utils import *

def main():
    env = Environment(x_min=0, x_max=50, y_min=0, y_max=50)

    base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=3.0)
    fading_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
    
    sim = WirelessSimulation(env, fading_model, interference_threshold_dbm=-75.0, enable_logging=True)

    sim.add_access_point(AccessPoint(id=0, x=10, y=10, tx_power=25, channel=1, 
                                     bandwidth=20, max_throughput=150.0))
    sim.add_access_point(AccessPoint(id=1, x=40, y=10, tx_power=25, channel=11, 
                                     bandwidth=20, max_throughput=150.0))
    sim.add_access_point(AccessPoint(id=2, x=25, y=25, tx_power=25, channel=1, 
                                     bandwidth=20, max_throughput=150.0))                                     
    sim.add_access_point(AccessPoint(id=3, x=10, y=40, tx_power=25, channel=1, 
                                     bandwidth=20, max_throughput=150.0))
    sim.add_access_point(AccessPoint(id=4, x=40, y=40, tx_power=25, channel=1, 
                                     bandwidth=20, max_throughput=150.0))

    for i in range(25):
        x = random.uniform(5, 45)
        y = random.uniform(5, 45)
        demand_mbps = random.uniform(5, 30)
        sim.add_client(Client(id=i, x=x, y=y, demand_mbps=demand_mbps))


    sim.initialize()

    USE_VISUALIZATION = True

    if USE_VISUALIZATION and PYGAME_AVAILABLE:
        print("Starting visualization... (Press ESC to exit)")
        print("\nVisualization Controls:")
        print("  SPACE: Pause/Resume")
        print("  I: Toggle interference display")
        print("  A: Toggle association lines")
        print("  C: Toggle coverage areas")
        print("  ESC: Exit")
        print("\nClient Color Code:")
        print("  Green: SINR > 20 dB (Excellent)")
        print("  Yellow: SINR 10-20 dB (Good)")
        print("  Red: SINR < 10 dB (Poor)")
        print("\nClient Info: [SINR in dB] [Allocated/Demand Mbps] [Airtime %]\n")

        sim.enable_visualization(width=1000, height=1040)
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