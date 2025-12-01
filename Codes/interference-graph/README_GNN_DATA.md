"""
Quick start guide for generating diverse GNN training data.

Example usage:

# Generate 20 scenarios with 100 steps each (2000 total timesteps)
python generate_training_data.py

# Generate more scenarios for larger dataset
python generate_training_data.py --scenarios 50 --steps 100

# Generate longer scenarios
python generate_training_data.py --scenarios 20 --steps 200

# Custom configuration
python generate_training_data.py --scenarios 30 --steps 150 --log-dir my_logs
"""

## Dataset Characteristics

The enhanced data generator creates diverse scenarios with:

### Network Scale Variation
- **APs**: 3-7 per scenario (randomly sampled)
- **Clients**: 10-50 per scenario (randomly sampled)

### Topology Types
1. **Grid**: Regular grid layout (structured)
2. **Random**: Uniformly distributed (unstructured)
3. **Clustered**: Hotspot-based grouping (realistic)
4. **Linear**: Corridor/hallway deployment (specialized)

### Parameter Variation
- **Channels**: Rotating through 1, 6, 11 (non-overlapping)
- **TX Power**: 20-30 dBm (randomized per AP)
- **Client Demand**: 5-35 Mbps (randomized per client)
- **Mobility**: Velocity 0.5-2.0 m/s (varied client speeds)

### Example Configurations

**Quick test** (400 timesteps):
```bash
python generate_training_data.py --scenarios 4 --steps 100
```

**Medium dataset** (2000 timesteps):
```bash
python generate_training_data.py --scenarios 20 --steps 100
```

**Large dataset** (5000 timesteps):
```bash
python generate_training_data.py --scenarios 50 --steps 100
```

**Very large dataset** (10000 timesteps):
```bash
python generate_training_data.py --scenarios 50 --steps 200
```

### Dataset Benefits for GNN

The diversity ensures the GNN learns to:
1. **Generalize across scales**: Handle 3-7 APs (not just fixed 5)
2. **Adapt to topologies**: Recognize patterns in different layouts
3. **Handle varied loads**: Deal with 10-50 clients per scenario
4. **Learn interference patterns**: Understand how channels/power affect graphs

### Training Considerations

- **Batching**: Graphs have different sizes - use PyG's dynamic batching
- **Normalization**: Features may need scaling across scenarios
- **Validation**: Ensure test scenarios include unseen AP/client counts
