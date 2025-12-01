from assoc import *
from datatype import *
from metrics import *
from sim import *
from utils import *

def test_logger():
    """Test the simulation logger functionality."""
    env = Environment(x_min=0, x_max=50, y_min=0, y_max=50)

    base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=3.0)
    fading_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
    
    # Create simulation with logging enabled
    sim = WirelessSimulation(env, fading_model, interference_threshold_dbm=-75.0, enable_logging=True)

    # Add APs
    sim.add_access_point(AccessPoint(id=0, x=10, y=10, tx_power=25, channel=1, 
                                     bandwidth=20, max_throughput=150.0))
    sim.add_access_point(AccessPoint(id=1, x=40, y=10, tx_power=25, channel=1, 
                                     bandwidth=20, max_throughput=150.0))

    # Add clients
    for i in range(5):
        x = random.uniform(5, 45)
        y = random.uniform(5, 45)
        demand_mbps = random.uniform(5, 30)
        sim.add_client(Client(id=i, x=x, y=y, demand_mbps=demand_mbps))

    # Initialize and run a few steps
    sim.initialize()
    
    print("Running 10 simulation steps...")
    for _ in range(10):
        sim.step()
    
    # Print logger summary
    if sim.logger:
        sim.logger.print_summary()
    
    print("\n✓ Logger test completed!")

if __name__ == "__main__":
    test_logger()
