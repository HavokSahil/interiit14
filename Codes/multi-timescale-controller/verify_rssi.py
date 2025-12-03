from assoc import *
from datatype import *
from metrics import *
from sim import *
from utils import *
from generate_training_data import create_grid_topology, create_random_topology
import os

def verify():
    print("Starting verification simulation...")
    env = Environment(x_min=0, x_max=50, y_min=0, y_max=50)

    base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=6.0)
    fading_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
    
    # Enable logging
    sim = WirelessSimulation(env, fading_model, interference_threshold_dbm=-75.0, enable_logging=True, log_dir="verify_logs_rssi")

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
    
    print("Running for 10 steps...")
    for i in range(10):
        sim.step()
            
    print("Simulation complete.")
    
    # Verify Client logs
    client_log = sim.logger.client_log_path
    print(f"Checking Client log: {client_log}")
    with open(client_log, 'r') as f:
        header = f.readline().strip().split(',')
        print(f"Client Log Header: {header}")
        # Check for new columns
        if 'rssi_dbm' not in header:
            print("FAILED: Missing rssi_dbm in Client log")
        else:
            print("SUCCESS: rssi_dbm column found.")
            
        # Check last line values
        last_line = f.readlines()[-1].strip().split(',')
        print(f"Sample Client data: {last_line}")
        
        # Check if rssi value is reasonable (should be negative)
        rssi_idx = header.index('rssi_dbm')
        rssi_val = float(last_line[rssi_idx])
        if rssi_val > 0 and rssi_val != 0.0: # 0.0 is default if not associated
             print(f"WARNING: RSSI value {rssi_val} seems positive, expected negative dBm")
        else:
             print(f"RSSI value {rssi_val} looks reasonable")

if __name__ == "__main__":
    verify()
