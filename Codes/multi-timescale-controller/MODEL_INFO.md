# GNN Model Documentation

## Model Architecture: EdgeConv

The interference prediction model is based on **EdgeConv** (Edge-Aware Message Passing), a Graph Neural Network architecture designed to learn from the spatial and feature relationships between nodes.

Unlike standard GCNs or GATs which primarily aggregate node features, **EdgeConv** dynamically constructs edge features by looking at the relationship between a node and its neighbors. This makes it ideal for predicting **edge weights** (interference coupling) in our fully connected interference graph.

### Architecture Details
- **Layer Type**: `EdgeConv` (PyTorch Geometric)
- **Message Passing**: $h_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \text{MLP}(h_i^{(l)} || h_j^{(l)} - h_i^{(l)})$
- **Number of Layers**: 3
- **Hidden Dimension**: 32
- **Dropout**: 0.2
- **Activation**: ELU (Exponential Linear Unit)
- **Readout Head**: MLP taking concatenated source and destination embeddings to predict a scalar edge weight.

---

## Input Features (9 Dimensions)

Each Access Point (AP) is represented by a feature vector with **9 components**. These features capture the RF environment, traffic load, and spatial configuration.

| Index | Feature | Unit | Description |
| :--- | :--- | :--- | :--- |
| 0 | **Incoming Energy** | dBm | Total RF energy detected at the AP's location (from all sources). |
| 1 | **Total Throughput** | Mbps | Sum of allocated throughput for all connected clients. |
| 2 | **Client Count** | Count | Number of clients currently associated with the AP. |
| 3 | **Duty Cycle** | 0-1 | Fraction of time the AP is transmitting (Airtime utilization). |
| 4 | **Roam In Events** | Count | Number of clients that roamed *to* this AP in the last step. |
| 5 | **Roam Out Events** | Count | Number of clients that roamed *away* from this AP in the last step. |
| 6 | **Channel** | Int | WiFi Channel ID (e.g., 1, 6, 11). Crucial for interference overlap. |
| 7 | **Bandwidth** | MHz | Channel width (e.g., 20, 40, 80). Affects overlap probability. |
| 8 | **Tx Power** | dBm | Transmission power level. Directly scales interference range. |

> **Note**: All features are Z-score normalized (mean=0, std=1) before being fed into the network.

---

## Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `in_channels` | 9 | Dimension of input node features. |
| `hidden_channels` | 32 | Size of internal node embeddings. |
| `num_layers` | 3 | Depth of the GNN. |
| `dropout` | 0.2 | Probability of zeroing elements during training (regularization). |
| `learning_rate` | 0.001 | Step size for the Adam optimizer. |
| `batch_size` | 1 | We train on 1 full graph snapshot at a time (Stochastic Gradient Descent). |

---

## Performance

The model achieves near-perfect regression on the test set, demonstrating that the 9 input features contain sufficient information to determine the interference graph.

- **R² Score**: ~0.97
- **Correlation**: ~0.99
- **MSE (Mean Squared Error)**: ~0.001
- **MAE (Mean Absolute Error)**: ~0.02

This high accuracy allows the simulation to skip expensive $O(N^2)$ ray-casting or path-loss computations for every frame, instead relying on the GNN inference which scales linearly with the number of edges but is highly parallelizable on GPUs.
