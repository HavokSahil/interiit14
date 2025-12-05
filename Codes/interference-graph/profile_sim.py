import time
import psutil
import os
import torch
import numpy as np
import networkx as nx
from sim import WirelessSimulation, APMetricsManager
from model import PathLossModel, MultipathFadingModel
from datatype import Environment, AccessPoint, Client
from generate_training_data import create_random_topology
from gnn_model import InterferenceGNN
from torch_geometric.data import Data
import random

class HeadlessGNN:
    def __init__(self, simulation):
        self.sim = simulation
        self.gnn_device = 'cpu'
        self.gnn_model = None
        self._load_gnn_model()
        
    def _load_gnn_model(self):
        """Load trained GNN model if it exists."""
        model_path = './models/best_model.pt'
        if not os.path.exists(model_path):
            print(f"GNN model not found at {model_path}. Train a model first.")
            return
        
        try:
            # Load model
            checkpoint = torch.load(model_path, map_location=self.gnn_device, weights_only=False)
            self.gnn_model = InterferenceGNN(
                in_channels=11,  # 11 node features
                hidden_channels=32,
                num_layers=3,
                dropout=0.2
            )
            self.gnn_model.load_state_dict(checkpoint['model_state_dict'])
            self.gnn_model.to(self.gnn_device)
            self.gnn_model.eval()
            print(f"Loaded GNN model from {model_path}")
        except Exception as e:
            print(f"Error loading GNN model: {e}")
            self.gnn_model = None

    def _create_graph_snapshot(self):
        """Create PyG Data object from current simulation state."""
        if self.gnn_model is None:
            return None
        
        # Create client map for fast lookup
        client_map = {c.id: c for c in self.sim.clients}
        
        # Extract features for each AP
        ap_features = []
        for ap in self.sim.access_points:
            # 1-3. Incoming energy (raw) for each channel
            inc_energy_ch1 = ap.inc_energy_ch1 if ap.inc_energy_ch1 != float('-inf') else -100.0
            inc_energy_ch6 = ap.inc_energy_ch6 if ap.inc_energy_ch6 != float('-inf') else -100.0
            inc_energy_ch11 = ap.inc_energy_ch11 if ap.inc_energy_ch11 != float('-inf') else -100.0
            
            # 2. Throughput
            total_throughput = 0.0
            for client_id in ap.connected_clients:
                if client_id in client_map:
                    total_throughput += client_map[client_id].throughput_mbps
            
            # 3. Number of clients
            num_clients = len(ap.connected_clients)
            
            # 4. Duty cycle
            duty_cycle = APMetricsManager.ap_duty(ap)
            
            # 5. Roaming events
            roam_in = self.sim.roam_events.get(ap.id, {}).get('in', 0) if hasattr(self.sim, 'roam_events') else 0
            roam_out = self.sim.roam_events.get(ap.id, {}).get('out', 0) if hasattr(self.sim, 'roam_events') else 0
            
            # 6. Channel
            channel = float(ap.channel)
            
            # 7. Bandwidth
            bandwidth = float(ap.bandwidth)
            
            # 8. Tx Power
            tx_power = float(ap.tx_power)
            
            ap_features.append([
                inc_energy_ch1,
                inc_energy_ch6,
                inc_energy_ch11,
                total_throughput,
                float(num_clients),
                duty_cycle,
                float(roam_in),
                float(roam_out),
                channel,
                bandwidth,
                tx_power
            ])
        x = torch.tensor(ap_features, dtype=torch.float)
        
        # Apply z-score normalization
        stats_path = 'models/norm_stats.pt'
        if os.path.exists(stats_path):
            try:
                stats = torch.load(stats_path, weights_only=False)
                mean = torch.tensor(stats['mean'], dtype=torch.float)
                std = torch.tensor(stats['std'], dtype=torch.float)
                x = (x - mean) / std
            except Exception as e:
                print(f"Error applying normalization: {e}")
                return None
        
        # Create fully connected edge index
        num_nodes = len(self.sim.access_points)
        src_list = []
        dst_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    src_list.append(i)
                    dst_list.append(j)
                    
        if len(src_list) > 0:
            fc_edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            fc_edge_index = torch.tensor([[], []], dtype=torch.long)
        
        data = Data(x=x, edge_index=fc_edge_index, num_nodes=num_nodes)
        return data

    def predict(self):
        data = self._create_graph_snapshot()
        if data is None:
            return None
        
        data = data.to(self.gnn_device)
        with torch.no_grad():
            self.gnn_model.predict_all_edges(data)
        return True

def profile_simulation():
    print("Starting headless profiling...")
    
    # Setup Environment
    env = Environment(x_min=0, x_max=50, y_min=0, y_max=50)
    base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=3.0)
    fading_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
    sim = WirelessSimulation(env, fading_model, interference_threshold_dbm=-75.0, enable_logging=False)

    # Setup Topology
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
    
    # Initialize Headless GNN
    gnn = HeadlessGNN(sim)
    if gnn.gnn_model is None:
        print("GNN model not loaded. Cannot profile inference.")
        return

    # Warmup
    print("Warming up...")
    for _ in range(10):
        sim.step()
        gnn.predict()

    # Metrics storage
    sim_step_times = []
    data_prep_times = []
    inference_times = []
    total_inference_times = []
    memory_usage = []
    cpu_usage = []

    process = psutil.Process(os.getpid())

    print("Profiling loop...")
    steps = 100
    for i in range(steps):
        # 1. Measure Simulation Physics Step
        t0 = time.perf_counter()
        sim.step()
        t1 = time.perf_counter()
        sim_step_times.append((t1 - t0) * 1000) # ms

        # 2. Measure GNN Pipeline
        t2 = time.perf_counter()
        data = gnn._create_graph_snapshot()
        t3 = time.perf_counter()
        
        if data is not None:
            data = data.to(gnn.gnn_device)
            with torch.no_grad():
                gnn.gnn_model.predict_all_edges(data)
            t4 = time.perf_counter()
            
            data_prep_times.append((t3 - t2) * 1000) # ms
            inference_times.append((t4 - t3) * 1000) # ms
            total_inference_times.append((t4 - t2) * 1000) # ms

        # 3. Resource Usage
        memory_usage.append(process.memory_info().rss / 1024 / 1024) # MB
        cpu_usage.append(process.cpu_percent())

    # Report
    print("\n=== Performance Profile (Headless) ===")
    print(f"Simulation Steps: {steps}")
    print(f"APs: {N_ap}, Clients: {N_client}")
    print("\nTiming (ms):")
    print(f"  Physics Simulation Step: Mean={np.mean(sim_step_times):.2f}, Std={np.std(sim_step_times):.2f}")
    print(f"  GNN Data Prep:           Mean={np.mean(data_prep_times):.2f}, Std={np.std(data_prep_times):.2f}")
    print(f"  GNN Inference (Model):   Mean={np.mean(inference_times):.2f}, Std={np.std(inference_times):.2f}")
    print(f"  Total Inference Pipeline:Mean={np.mean(total_inference_times):.2f}, Std={np.std(total_inference_times):.2f}")
    
    print("\nResource Usage:")
    print(f"  Memory (RSS):            Mean={np.mean(memory_usage):.2f} MB, Max={np.max(memory_usage):.2f} MB")
    print(f"  CPU Usage:               Mean={np.mean(cpu_usage):.1f}%")

if __name__ == "__main__":
    profile_simulation()
