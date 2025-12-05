from assoc import *
from datatype import *
from metrics import *
from sim import *
from utils import *
from generate_training_data import create_random_topology
import random
import os
import shutil

def main():
    # Clear old logs
    if os.path.exists("logs"):
        shutil.rmtree("logs")
    os.makedirs("logs")

    NUM_EPISODES = 20
    STEPS_PER_EPISODE = 100
    
    print(f"Starting batch data generation: {NUM_EPISODES} episodes, {STEPS_PER_EPISODE} steps each.")

    for episode in range(NUM_EPISODES):
        print(f"\n--- Episode {episode+1}/{NUM_EPISODES} ---")
        
        env = Environment(x_min=0, x_max=50, y_min=0, y_max=50)
        base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=3.0)
        fading_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
        
        sim = WirelessSimulation(env, fading_model, interference_threshold_dbm=-75.0, enable_logging=True)

        # Randomize AP count and parameters
        N_ap = random.randint(4, 9)
        print(f"  APs: {N_ap}")
        ap_positions = create_random_topology(N_ap, env)
        
        for i, (x, y) in enumerate(ap_positions):
            channel = random.choice([1, 6, 11])
            # Randomize Tx power for diversity
            tx_power = random.uniform(10, 30) 
            sim.add_access_point(AccessPoint(id=i, x=x, y=y, tx_power=tx_power, channel=channel, 
                                             bandwidth=20, max_throughput=150.0))

        # Randomize Client count and parameters
        N_client = random.randint(15, 40)
        print(f"  Clients: {N_client}")
        client_positions = create_random_topology(N_client, env)
        
        for i, (x, y) in enumerate(client_positions):
            demand_mbps = random.uniform(2, 40)
            velocity = random.uniform(0.5, 3.0)
            sim.add_client(Client(id=i, x=x, y=y, demand_mbps=demand_mbps, velocity=velocity))

        sim.initialize()

        # Run simulation steps
        for step in range(STEPS_PER_EPISODE):
            sim.step()
            if (step + 1) % 20 == 0:
                print(f"    Step {step+1}/{STEPS_PER_EPISODE}")

        # Get interference graph stats
        graph = sim.get_interference_graph()
        print(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    print("\nBatch data generation complete.")

if __name__ == "__main__":
    main()