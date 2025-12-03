# Wireless Interference Graph Prediction using GNN

This project implements a Graph Neural Network (GNN) to predict interference edge weights between Access Points (APs) in a wireless network. It uses an **EdgeConv** architecture to learn from spatial and RF parameters, enabling real-time interference estimation without expensive active measurements.

## Quick Start

### 1. Run the Live Simulation
To see the simulation with the pre-trained model:
```bash
python3 main.py
```
**Controls:**
- `P`: Toggle GNN-predicted interference graph
- `I`: Toggle ground-truth interference graph
- `C`: Toggle coverage areas
- `SPACE`: Pause/Resume

### 2. Train the Model
If you want to retrain the model from scratch:
```bash
# 1. Generate training data (runs simulation in headless mode)
python3 generate_training_data.py --steps 5000

# 2. Train the GNN
python3 train_gnn.py
```

---

## Project Structure

### Core Simulation
- **`main.py`**: Entry point. Sets up the environment, APs, clients, and starts the visualization loop.
- **`sim.py`**: The heart of the simulation. Handles:
    - Physics updates (mobility, signal propagation).
    - Visualization (PyGame).
    - **GNN Inference**: Extracts features and runs the trained model in real-time.
- **`model.py`**: RF propagation models (Path Loss, Multipath Fading) and Client Mobility logic.
- **`assoc.py`**: Client association algorithms (Voronoi, Signal Strength).
- **`datatype.py`**: Data classes (`AccessPoint`, `Client`, `Environment`).
- **`metrics.py`**: Calculation of SINR, throughput, and interference metrics.
- **`logger.py`**: Logs simulation state to CSV files in `logs/` for training.

### GNN & Machine Learning
- **`gnn_model.py`**: Defines the **EdgeConv** GNN architecture.
    - Input: 9 features per AP (Energy, Throughput, Clients, Duty Cycle, Roaming, Channel, BW, TxPower).
    - Output: Predicted edge weight (interference factor) for every AP pair.
- **`gnn_data.py`**: Data pipeline.
    - Loads CSV logs.
    - Preprocesses features (normalization).
    - Creates PyTorch Geometric `Data` objects.
- **`train_gnn.py`**: Training script.
    - Loads data, initializes model, runs training loop.
    - Saves best model to `models/best_model.pt` and normalization stats to `models/norm_stats.pt`.
- **`evaluate_gnn.py`**: Standalone script to evaluate a trained model on test data and print detailed metrics.
- **`generate_training_data.py`**: Helper script to run the simulation rapidly without visualization to generate large datasets for training.

---

## Workflow Details

### 1. Data Generation
The GNN needs historical data to learn. The simulation logs AP states, client states, and ground-truth interference graphs to `logs/`.

**Command:**
```bash
python3 generate_training_data.py --steps 2000 --scenarios 5
```
This runs 5 different random scenarios for 2000 steps each.

### 2. Training
The training script reads from `logs/`, creates a graph dataset, and trains the EdgeConv model.

**Command:**
```bash
python3 train_gnn.py
```
**Key Configs (in `train_gnn.py`):**
- `in_channels`: 9 (Input features)
- `hidden_channels`: 32
- `num_layers`: 3
- `learning_rate`: 0.001

### 3. Evaluation
To check model performance on unseen data:

**Command:**
```bash
python3 evaluate_gnn.py --model models/best_model.pt
```

### 4. Real-time Inference
The `WirelessSimulation` class in `sim.py` loads `models/best_model.pt` and `models/norm_stats.pt`.
During the simulation loop, it:
1. Extracts current AP features.
2. Normalizes them using the saved stats.
3. Feeds them to the GNN.
4. Updates the "Predicted Graph" visualization.

---

## Model Architecture

**Type:** EdgeConv (Edge-Aware Message Passing)
**Input Features (9):**
1.  **Incoming Energy** (dBm)
2.  **Total Throughput** (Mbps)
3.  **Number of Clients**
4.  **Duty Cycle** (%)
5.  **Roam In Events**
6.  **Roam Out Events**
7.  **Channel** (e.g., 1, 6, 11)
8.  **Bandwidth** (e.g., 20, 40 MHz)
9.  **Tx Power** (dBm)

**Output:**
- **Edge Weight**: A value between 0 and 1 representing the interference coupling between two APs.
