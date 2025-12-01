# Live GNN Prediction Feature

Press **'P'** during simulation to toggle between actual and predicted interference graphs!

## How It Works

1. The visualization automatically loads `models/best_model.pt` on startup
2. Press **'P'** to switch to GNN predictions
3. The model predicts edges in real-time based on current AP state
4. Predicted edges are shown in **blue** vs actual edges in **red/yellow**

## Visual Indicators

- **Top-right corner**: "PREDICTED GRAPH" indicator when active
- **Edge colors**:
  - Red→Yellow gradient: Actual interference (RSSI-based)
  - Blue gradient: GNN predictions (confidence-based)
- **Graph View (G)**: Also shows predicted graph in blue when active
- **Controls updated**: Shows "P: Toggle Predicted Graph"

## Features Used for Prediction

The GNN uses current AP state:
1. Incoming energy (dBm)
2. Allocated throughput
3. Number of connected clients
4. Duty cycle
5. Roaming activity (set to 0 for real-time - not tracked live)

## Usage Example

```bash
# 1. Run simulation with visualization
python main.py

# 2. During visualization:
#    - Watch actual interference (default)
#    - Press 'P' to see GNN predictions
#    - Press 'P' again to toggle back
#    - Press 'I' to hide/show interference
```

## Notes

- If no model exists, pressing 'P' shows: "GNN model not available"
- Model predictions update every frame (real-time)
- Works with any number of APs (handles 3-7 dynamically)
- Threshold: 0.5 probability for edge display
