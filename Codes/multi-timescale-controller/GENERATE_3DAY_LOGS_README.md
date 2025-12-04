# 3-Day Simulation Log Generator - README

## Overview

`generate_3day_logs.py` generates comprehensive simulation logs spanning 3 days (72 hours) with realistic network behavior patterns, event injection, and complete audit trails.

---

## Features

### 🕐 Realistic Time-Based Patterns

**Day/Night Cycles**
- **Peak Hours** (9am-5pm): 2x client density
- **Night Hours** (10pm-6am): 0.3x client density  
- **Normal Hours**: 1x client density
- **Weekend Effect**: 50% reduction on Day 2 (Saturday)

**Dynamic Client Count**
- Base: 20 clients
- Peak: ~40 clients
- Night: ~6 clients
- Clients added/removed every 10 minutes

### ⚡ Event Injection Schedule

| Event Type | Probability | Peak Hour Boost | Typical Count (3 days) |
|------------|-------------|-----------------|------------------------|
| DFS Radar | 0.1% per hour | No | ~2-5 events |
| Interference Burst | 0.5% per hour | 4x during peak | ~100-200 events |
| Spectrum Saturation | 1% per hour | Peak hours only | ~20-40 events |

### 📊 Output Data

**Simulation Logs** (`simulation_logs_3day/`)
- CSV files with step-by-step network state
- AP configurations, client metrics, roaming events
- Interference graph snapshots

**Audit Logs** (`audit_logs_3day/`)
- HMAC-signed audit trail
- All event actions and rollbacks
- Configuration change history

**Summary Report** (`simulation_summary.json`)
```json
{
  "simulation_duration_sec": 850.5,
  "total_steps": 25920,
  "num_days": 3,
  "network": {
    "access_points": 6,
    "peak_clients": 42,
    "total_roams": 3450
  },
  "events": {
    "total": 245,
    "by_type": {
      "dfs_radar": 3,
      "interference": 182,
      "spectrum_sat": 60
    }
  },
  "rrm": {
    "actions_executed": 198,
    "rollbacks_triggered": 12
  }
}
```

---

## Usage

### Basic Run

```bash
python generate_3day_logs.py
```

You'll be prompted to confirm:
```
Proceed? (y/n): y
```

### Expected Runtime

- **Duration**: 10-15 minutes (depending on CPU)
- **Steps**: 25,920 (3 days × 24 hours × 360 steps/hour)
- **Disk Space**: 50-100 MB

### Progress Output

```
[Day 1, Hour 09:00] Step 3,240/25,920 (12.5%) | Clients: 38 | Events: 25 | Actions: 18 | ETA: 11.2min
[Day 1, Hour 10:00] Step 3,600/25,920 (13.9%) | Clients: 42 | Events: 32 | Actions: 22 | ETA: 10.8min
...
```

---

## Simulation Parameters

### Network Topology

```python
# 6 Access Points (grid layout)
AP Channels: [52, 6, 36, 11, 40, 1]
  - 3 on 5 GHz (52, 36, 40)
  - 3 on 2.4 GHz (6, 11, 1)

TX Power: 20-23 dBm (random per AP)
Bandwidth: 80 MHz (5G), 20 MHz (2.4G)

# 2 Interferers
Types: Microwave, Bluetooth
Duty Cycle: 50-80%
Channels: 6, 11
```

### Client Dynamics

```python
Initial: 20 clients
Peak: ~40 clients (9am-5pm)
Night: ~6 clients (10pm-6am)

Demand: 5-30 Mbps per client
Velocity: 0.5-2.0 m/s (random walk)
Roaming: Signal-strength based
```

### Time Scale

```python
1 step = 10 seconds
360 steps = 1 hour
8,640 steps = 1 day
25,920 steps = 3 days

Total simulation time: ~15 minutes
Simulated time: 72 hours
```

---

## Output Files

### 1. Simulation Logs

**Location**: `simulation_logs_3day/3day_sim_YYYYMMDD_HHMMSS.csv`

**Format**: CSV with columns:
```csv
step,ap_id,channel,tx_power,num_clients,roam_in,roam_out,p95_throughput,p95_retry_rate,...
```

**Size**: ~30-50 MB

### 2. Audit Logs

**Location**: `audit_logs_3day/audit_YYYYMMDD.jsonl`

**Format**: JSON Lines (one record per line)
```json
{"audit_id": "...", "event_type": "DFS_RADAR", "action_type": "CHANNEL_CHANGE", ...}
{"audit_id": "...", "event_type": "NON_WIFI_BURST", "action_type": "CHANNEL_CHANGE", ...}
```

**Size**: ~10-20 MB

### 3. Summary Report

**Location**: `simulation_logs_3day/simulation_summary.json`

**Contents**:
- Execution statistics
- Network metrics (roams, peak clients)
- Event distribution by type and hour
- RRM action summary

**Size**: ~5 KB

### 4. Audit Export

**Location**: `audit_logs_3day/export_YYYYMMDD_HHMMSS.jsonl`

**Contents**: Filtered audit trail for analysis

---

## Event Distribution Example

### Events by Hour of Day

```
00:00 - 00:59    5  █
01:00 - 01:59    3  
02:00 - 02:59    4  
03:00 - 03:59    2  
04:00 - 04:59    3  
05:00 - 05:59    6  █
06:00 - 06:59   12  ██
07:00 - 07:59   18  ███
08:00 - 08:59   24  ████
09:00 - 09:59   45  █████████    ← Peak start
10:00 - 10:59   52  ██████████
11:00 - 11:59   48  █████████
12:00 - 12:59   50  ██████████
13:00 - 13:59   49  █████████
14:00 - 14:59   46  █████████
15:00 - 15:59   43  ████████
16:00 - 16:59   38  ███████      ← Peak end
17:00 - 17:59   22  ████
18:00 - 18:59   15  ███
19:00 - 19:59   10  ██
20:00 - 20:59    8  █
21:00 - 21:59    6  █
22:00 - 22:59    4  
23:00 - 23:59    3  
```

---

## Customization

### Adjust Simulation Duration

```python
# In ThreeDaySimulation.__init__()
self.num_days = 7  # Change to 7 days
```

### Adjust Client Patterns

```python
# In get_client_count()
base_count = 50  # Change base client count

# Peak hours
if self.is_peak_hour(hour):
    multiplier = 3.0  # Increase peak multiplier
```

### Adjust Event Probabilities

```python
# In inject_random_events()

# DFS events (increase probability)
if random.random() < 0.005:  # was 0.001
    rrm.inject_dfs_event(...)

# Interference events
interference_prob = 0.05 if peak else 0.01  # was 0.02/0.005
```

### Add More APs or Interferers

```python
# In run()
N_ap = 10  # Change from 6 to 10 APs

# Add more interferers
for i in range(5):  # Change from 2 to 5
    sim.add_interferer(...)
```

---

## Analysis Examples

### 1. Load Simulation Logs

```python
import pandas as pd

df = pd.read_csv('simulation_logs_3day/3day_sim_*.csv')

# Analyze roaming patterns
roam_stats = df.groupby('hour')[['roam_in', 'roam_out']].sum()

# Plot client count over time
import matplotlib.pyplot as plt
client_count = df.groupby('step')['num_clients'].sum()
plt.plot(client_count)
plt.xlabel('Step')
plt.ylabel('Total Clients')
plt.show()
```

### 2. Analyze Audit Trail

```python
import json

audit_records = []
with open('audit_logs_3day/audit_20241204.jsonl', 'r') as f:
    for line in f:
        audit_records.append(json.loads(line))

# Count events by type
from collections import Counter
event_types = [r['event']['event_type'] for r in audit_records if r['event']]
print(Counter(event_types))

# Rollback rate
total_actions = len(audit_records)
rolled_back = sum(1 for r in audit_records if r['execution_status'] == 'ROLLED_BACK')
print(f"Rollback rate: {rolled_back/total_actions:.1%}")
```

### 3. Peak Hour Analysis

```python
# Extract peak hour events
peak_events = [r for r in audit_records 
               if 9 <= extract_hour(r['timestamp_utc']) < 17]

print(f"Peak hour events: {len(peak_events)}")
print(f"Off-peak events: {len(audit_records) - len(peak_events)}")
```

---

## Performance Benchmarks

### Typical Performance (Intel i7, 16GB RAM)

| Metric | Value |
|--------|-------|
| Steps/second | ~30-50 |
| Memory usage | ~200 MB |
| CPU usage | 80-100% (single core) |
| Disk I/O | ~5 MB/min |
| Total runtime | 10-15 minutes |

### Optimization Tips

**1. Reduce Logging Frequency**
```python
# In WirelessSimulation
if step % 10 == 0:  # Log every 10 steps instead of every step
    self.logger.log_step(...)
```

**2. Disable Visualization**
```python
sim = WirelessSimulation(..., enable_logging=True)
# Don't call enable_visualization()
```

**3. Run on Multiple Cores**
```python
# Split into multiple simulations
# Day 1, Day 2, Day 3 in parallel
```

---

## Troubleshooting

### Issue: Simulation Too Slow

**Solution**: Reduce steps per hour
```python
self.steps_per_hour = 180  # 20 sec/step instead of 10
```

### Issue: Out of Memory

**Solution**: Reduce client count or log frequency
```python
base_count = 10  # Reduce from 20
# Or reduce logging frequency
```

### Issue: Too Few Events

**Solution**: Increase event probabilities
```python
# DFS
if random.random() < 0.01:  # 10x increase

# Interference
interference_prob = 0.1 if peak else 0.05  # 10x increase
```

### Issue: Disk Space Full

**Solution**: Reduce simulation duration or increase log interval
```python
self.num_days = 1  # Just 1 day
# Or log less frequently
```

---

## Validation

### Check Output Files

```bash
# List generated files
ls -lh simulation_logs_3day/
ls -lh audit_logs_3day/

# Count log entries
wc -l simulation_logs_3day/*.csv
wc -l audit_logs_3day/*.jsonl

# Validate JSON
python -m json.tool simulation_logs_3day/simulation_summary.json
```

### Verify Event Counts

```bash
# Count events in audit log
grep -c '"event_type"' audit_logs_3day/audit_*.jsonl

# Count rollbacks
grep -c 'ROLLED_BACK' audit_logs_3day/audit_*.jsonl
```

---

## Example Run Output

```
======================================================================
3-DAY SIMULATION LOG GENERATOR
======================================================================
Total steps: 25,920
Steps per hour: 360
Duration: 3 days
Output: simulation_logs_3day/
Audit: audit_logs_3day/
======================================================================

Proceed? (y/n): y

======================================================================
3-DAY SIMULATION LOG GENERATOR
======================================================================

Creating network topology...
Starting 3-day simulation (25,920 steps)...
This will take several minutes...

[Day 1, Hour 00:00] Step 360/25,920 (1.4%) | Clients: 8 | Events: 2 | Actions: 1 | ETA: 14.5min
[Day 1, Hour 01:00] Step 720/25,920 (2.8%) | Clients: 6 | Events: 3 | Actions: 2 | ETA: 14.1min
...
[Day 3, Hour 23:00] Step 25,920/25,920 (100.0%) | Clients: 5 | Events: 245 | Actions: 198 | ETA: 0.0min

Simulation complete!

======================================================================
3-DAY SIMULATION SUMMARY
======================================================================

Duration: 12.4 minutes (745.2 seconds)
Steps executed: 25,920
Steps per second: 34.8

Network Statistics:
  Access Points: 6
  Peak Clients: 42
  Total Client Roams: 3,450

Event Statistics:
  Total Events: 245
    dfs_radar: 3
    interference: 182
    spectrum_sat: 60

RRM Statistics:
  Actions Executed: 198
  Rollbacks Triggered: 12

Events by Hour of Day:
  [Distribution chart shown above]

======================================================================
Logs saved to: simulation_logs_3day/
Audit trail: audit_logs_3day/
Summary: simulation_logs_3day/simulation_summary.json
======================================================================

✓ 3-day simulation log generation complete!
```

---

## Use Cases

### 1. Algorithm Training Data
Generate training data for machine learning models (GNN, RL)

### 2. System Validation
Validate RRM Engine behavior over extended periods

### 3. Performance Testing
Stress test event loop and rollback mechanisms

### 4. Analytics & Visualization
Generate data for network analytics dashboards

### 5. Documentation
Create realistic examples for presentations/papers

---

**Version**: 1.0  
**Date**: December 4, 2024  
**Status**: Production Ready
