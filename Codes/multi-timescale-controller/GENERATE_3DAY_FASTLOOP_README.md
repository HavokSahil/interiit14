# 3-Day Fast Loop RRM Simulation - README

## Overview

`generate_3day_fastloop_logs.py` generates comprehensive simulation logs for 3 days (72 hours) with a focus on tracking the **Refactored Fast Loop Controller**'s behavior, EWMA baseline evolution, adaptive tolerances, and automatic rollback mechanisms.

---

## Features

### 🕐 Realistic Network Behavior

**Day/Night Cycles** (same as Event Loop simulation)
- **Peak Hours** (9am-5pm): 2x client density
- **Night Hours** (10pm-6am): 0.3x client density
- **Weekend**: 50% reduction on Day 2

**Network Stress Injection**
- Traffic bursts during peak hours (every 30 min)
- CCA busy increases (10-30%)
- Client RSSI degradation (3-8 dB)
- Retry rate spikes (5-15%)

### ⚡ Fast Loop Tracking

**Actions Logged**:
- TX Power Refinement
- QoE Rapid Correction
- Action success/failure
- Rollback triggers

**Metrics Tracked**:
- Actions per hour
- Rollbacks per hour
- Actions per AP
- Rollback rate evolution

### 📊 EWMA Baseline Evolution

**Tracked per AP**:
- `ewma_mean_throughput_mean` - Throughput baseline
- `ewma_mean_median_rssi` - RSSI baseline
- `ewma_mean_retry_rate` - Retry rate baseline
- `ewma_mean_cca_busy_percentage` - CCA busy baseline
- `ewma_var_*` - Variance for adaptive tolerance

**Logged Every Hour**: 72 data points × 6 APs = 432 baseline snapshots

### 🔄 Rollback Monitoring

**Detailed Tracking**:
- Step when rollback occurred
- Hour of day
- AP ID
- Action type
- Rollback reason (throughput drop, etc.)
- Pre/post metrics

---

## Output Files

### 1. Fast Loop Metrics CSV

**Location**: `simulation_logs_fastloop/fastloop_metrics_YYYYMMDD_HHMMSS.csv`

**Columns**:
```csv
step,hour,day,ap_id,
actions_executed,actions_succeeded,actions_rolled_back,
rollback_rate,active_penalties,
ewma_throughput,ewma_rssi,ewma_retry,ewma_cca,
var_throughput,var_rssi
```

**Content**: Logged every hour for each AP
- Step: Simulation step number
- Stats: Cumulative Fast Loop statistics
- EWMA baselines: Current baseline values
- Variances: For adaptive tolerance computation

**Size**: ~50-100 KB (72 hours × 6 APs)

### 2. Simulation CSV Logs

**Location**: `simulation_logs_fastloop/sim_*.csv`

**Content**: Standard simulation logs (from `WirelessSimulation`)
- AP configurations
- Client metrics
- Roaming events

**Size**: ~30-50 MB

### 3. Audit Trail

**Location**: `fastloop_audit/audit_YYYYMMDD.jsonl`

**Content**: HMAC-signed audit records
- Event Loop actions
- Fast Loop actions (via integrated audit logger)
- Configuration changes
- Rollback events

**Size**: ~10-20 MB

### 4. Summary JSON

**Location**: `simulation_logs_fastloop/fastloop_simulation_summary.json`

**Content**:
```json
{
  "simulation_duration_sec": 850.2,
  "total_steps": 25920,
  "num_days": 3,
  "network": {
    "peak_clients": 45,
    "total_roams": 3240
  },
  "fast_loop": {
    "total_actions": 156,
    "successful_actions": 142,
    "rolled_back_actions": 14,
    "tx_power_actions": 98,
    "qoe_corrections": 58,
    "actions_by_hour": [...],
    "rollbacks_by_hour": [...],
    "actions_by_ap": {...}
  },
  "event_loop": {...},
  "rollback_details": [...]
}
```

**Size**: ~50-100 KB

---

## Usage

### Basic Run

```bash
python generate_3day_fastloop_logs.py
# Confirm with 'y'
```

### Expected Output

```
======================================================================
3-DAY FAST LOOP RRM SIMULATION
======================================================================
Total steps: 25,920
Steps per hour: 360
Duration: 3 days
Output: simulation_logs_fastloop/
Audit: fastloop_audit/
======================================================================

Creating network topology...
Initializing Enhanced RRM Engine with Refactored Fast Loop...
[RRM] Refactored Fast Loop Controller initialized

Starting 3-day simulation (25,920 steps)...

[Day 1, Hour 09:00] Step 3,240/25,920 (12.5%) | Clients: 38 | Fast Loop Actions: 12 | Rollbacks: 1 | ETA: 13.5min
[Day 1, Hour 10:00] Step 3,600/25,920 (13.9%) | Clients: 42 | Fast Loop Actions: 18 | Rollbacks: 2 | ETA: 12.8min
...
[Day 3, Hour 23:00] Step 25,920/25,920 (100.0%) | Clients: 5 | Fast Loop Actions: 156 | Rollbacks: 14 | ETA: 0.0min

Simulation complete!
```

---

## Detailed Output Statistics

### Example Summary

```
======================================================================
3-DAY FAST LOOP SIMULATION SUMMARY
======================================================================

Duration: 14.2 minutes (851.4 seconds)
Steps executed: 25,920
Steps per second: 30.4

Network Statistics:
  Access Points: 6
  Peak Clients: 45
  Total Client Roams: 3,240

Fast Loop Statistics:
  Total Actions: 156
  Successful Actions: 142
  Rolled Back Actions: 14
  Rollback Rate: 9.0%
  TX Power Actions: 98
  QoE Corrections: 58

Actions by AP:
  AP 0: 28 actions
  AP 1: 32 actions
  AP 2: 24 actions
  AP 3: 26 actions
  AP 4: 22 actions
  AP 5: 24 actions

Fast Loop Actions by Hour:
  00:00 - 00:59  Actions:    2  Rollbacks:   0  
  01:00 - 01:59  Actions:    1  Rollbacks:   0  
  ...
  09:00 - 09:59  Actions:   18  Rollbacks:   2  ███
  10:00 - 10:59  Actions:   22  Rollbacks:   2  ████
  11:00 - 11:59  Actions:   20  Rollbacks:   1  ████
  12:00 - 12:59  Actions:   24  Rollbacks:   3  ████
  13:00 - 13:59  Actions:   21  Rollbacks:   2  ████
  14:00 - 14:59  Actions:   19  Rollbacks:   1  ███
  15:00 - 15:59  Actions:   16  Rollbacks:   1  ███
  16:00 - 16:59  Actions:   12  Rollbacks:   1  ██
  ...

Final Fast Loop Engine State:
  Total Actions Executed: 156
  Total Actions Succeeded: 142
  Total Rollbacks: 14
  Final Rollback Rate: 9.0%
  Active Penalties: 2
```

---

## Analysis Examples

### 1. EWMA Baseline Evolution

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load Fast Loop metrics
df = pd.read_csv('simulation_logs_fastloop/fastloop_metrics_*.csv')

# Plot EWMA throughput baseline evolution for AP 0
ap0 = df[df['ap_id'] == 0]

plt.figure(figsize=(12, 6))
plt.plot(ap0['step'], ap0['ewma_throughput'], label='Throughput Baseline')
plt.xlabel('Simulation Step')
plt.ylabel('EWMA Baseline (Mbps)')
plt.title('AP 0: Throughput Baseline Evolution (3 days)')
plt.grid(True)
plt.legend()
plt.show()
```

### 2. Rollback Rate Over Time

```python
# Calculate rollback rate over time
df['rollback_rate_pct'] = df['rollback_rate'] * 100

# Plot for all APs
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, ax in enumerate(axes.flat):
    if i < 6:
        ap_data = df[df['ap_id'] == i]
        ax.plot(ap_data['step'], ap_data['rollback_rate_pct'])
        ax.set_title(f'AP {i}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Rollback Rate (%)')
        ax.grid(True)
plt.tight_layout()
plt.show()
```

### 3. Actions by Time of Day

```python
import json

# Load summary
with open('simulation_logs_fastloop/fastloop_simulation_summary.json') as f:
    summary = json.load(f)

# Extract actions by hour
actions_by_hour = summary['fast_loop']['actions_by_hour']
rollbacks_by_hour = summary['fast_loop']['rollbacks_by_hour']

# Plot
hours = range(24)
plt.figure(figsize=(14, 6))
plt.bar(hours, actions_by_hour, label='Actions', alpha=0.7)
plt.bar(hours, rollbacks_by_hour, label='Rollbacks', alpha=0.7, color='red')
plt.xlabel('Hour of Day')
plt.ylabel('Count')
plt.title('Fast Loop Actions and Rollbacks by Hour (3-day aggregate)')
plt.legend()
plt.grid(True, axis='y')
plt.show()
```

### 4. Adaptive Tolerance Analysis

```python
# Load metrics
df = pd.read_csv('simulation_logs_fastloop/fastloop_metrics_*.csv')

# For AP 0, plot variance evolution
ap0 = df[df['ap_id'] == 0]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Throughput variance
ax1.plot(ap0['step'], ap0['var_throughput'])
ax1.set_title('Throughput Variance (for adaptive tolerance)')
ax1.set_ylabel('Variance')
ax1.grid(True)

# RSSI variance
ax2.plot(ap0['step'], ap0['var_rssi'])
ax2.set_title('RSSI Variance (for adaptive tolerance)')
ax2.set_xlabel('Step')
ax2.set_ylabel('Variance')
ax2.grid(True)

plt.tight_layout()
plt.show()
```

---

## Customization

### Adjust Simulation Duration

```python
# In ThreeDayFastLoopSimulation.__init__()
self.num_days = 7  # Change to 7 days
```

### Adjust Client Patterns

```python
# In get_client_count()
base_count = 30  # Change base client count

if self.is_peak_hour(hour):
    multiplier = 3.0  # Increase peak multiplier
```

### Adjust Network Stress

```python
# In inject_network_stress()
if self.is_peak_hour(hour) and random.random() < 0.2:  # More frequent
    for ap in sim.access_points:
        ap.cca_busy_percentage = min(0.98, ...)  # Higher stress
```

### Change Logging Frequency

```python
# In run()
if step % (self.steps_per_hour // 2) == 0:  # Every 30 min instead of hourly
    self.log_fast_loop_metrics(step, rrm, rrm_result)
```

---

## Performance

### Expected Runtime

- **Total Steps**: 25,920
- **Duration**: 12-18 minutes (depends on CPU)
- **Steps/second**: ~25-35

### Memory Usage

- **Peak**: ~300-400 MB
- **EWMA state**: ~10 KB per AP
- **Audit cache**: ~1 MB

### Disk Usage

- **Fast Loop Metrics CSV**: ~100 KB
- **Simulation logs**: ~50 MB
- **Audit trail**: ~20 MB
- **Summary**: ~100 KB
- **Total**: ~70-80 MB

---

## Troubleshooting

### High Rollback Rate (>20%)

**Causes**:
1. Too aggressive Fast Loop parameters
2. High network variance
3. Insufficient EWMA samples

**Solutions**:
```python
# Increase tolerances in fast_loop_refactored.py
pol = {
    "throughput_drop_tolerance": 0.20,  # Increase from 0.12
    "t_eval": 120  # Longer evaluation window
}

# Or increase EWMA alpha for faster adaptation
pol = {"ewma_alpha": 0.35}  # Increase from 0.25
```

### Too Few Actions

**Symptoms**: <50 actions in 3 days

**Causes**:
1. No triggers met
2. APs in penalty/cooldown
3. Metrics too stable

**Debug**:
```python
# Check if Fast Loop actually loaded
print(hasattr(rrm.fast_loop_engine, 'get_statistics'))

# Check penalty state
print(rrm.fast_loop_engine.state_store.get('penalties', {}))
```

### Memory Issues

**Solution**: Reduce logging frequency
```python
# Log every 2 hours instead of hourly
if step % (self.steps_per_hour * 2) == 0:
    self.log_fast_loop_metrics(...)
```

---

## Validation

### Check Output Files

```bash
# List generated files
ls -lh simulation_logs_fastloop/
ls -lh fastloop_audit/

# Count CSV rows
wc -l simulation_logs_fastloop/fastloop_metrics_*.csv

# Validate JSON
python -m json.tool simulation_logs_fastloop/fastloop_simulation_summary.json

# Check for rollback details
grep -c "rolled_back" simulation_logs_fastloop/fastloop_simulation_summary.json
```

---

## Comparison with Event Loop Simulation

| Feature | Event Loop Sim | Fast Loop Sim |
|---------|---------------|---------------|
| **Focus** | Critical events | Fine-grained optimization |
| **Actions** | DFS, interference | TX power, QoE |
| **Frequency** | Rare (events) | Frequent (every step) |
| **Metrics** | Event count | EWMA baselines, variances |
| **Rollbacks** | DFS rollback | Metric degradation |
| **Logs** | Event audit | EWMA evolution |

Both simulations can run simultaneously in the same RRM Engine!

---

## Use Cases

### 1. EWMA Tuning
Generate data to optimize EWMA alpha parameter

### 2. Tolerance Calibration
Analyze rollback rates to calibrate adaptive tolerances

### 3. Fast Loop Validation
Verify Fast Loop behavior over extended periods

### 4. Performance Benchmarking
Compare Fast Loop vs Manual configuration

### 5. ML Training Data
Use EWMA baselines as features for ML models

---

**Version**: 1.0  
**Date**: December 4, 2024  
**Status**: Production Ready  
**Related**: `generate_3day_logs.py` (Event Loop simulation)
