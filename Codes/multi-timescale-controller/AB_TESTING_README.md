# A/B Testing & QoE Monitoring - Quick Start Guide

## Overview
This guide shows how to use the A/B testing and QoE monitoring features to compare RRM strategies.

## Features
- **A/B Testing**: Compare different RRM configurations (No RRM, Fast Loop, Event Loop, Full RRM)
- **QoE Monitoring**: Track Quality of Experience in real-time with role-based weights
- **Visualization**: Interactive dashboards showing QoE improvements
- **Statistical Analysis**: T-tests for significance, improvement percentages

---

## Quick Example

```python
from ab_testing import ABTestFramework, ABTestConfig
from qoe_monitor import QoEMonitor
from qoe_dashboard import QoEDashboard

# Define variants
variant_a = ABTestConfig(
    variant_name="No RRM (Baseline)",
    rrm_mode="none"  # No optimization
)

variant_b = ABTestConfig(
    variant_name="Fast Loop",
    rrm_mode="fast_loop"  # Fast Loop enabled
)

# Create framework
ab_test = ABTestFramework(variant_a, variant_b)

# Run test (see example_ab_test.py for full code)
# ... simulation code ...

# Get results
comparison = ab_test.compute_comparison()
comparison.print_report()
```

---

## Running the Example

```bash
# Run example A/B test
python example_ab_test.py
```

This will:
1. Run 5000 steps with no RRM (baseline)
2. Run 5000 steps with Fast Loop enabled
3. Compare QoE, throughput, retry rate
4. Generate visualizations
5. Export results to JSON/CSV

---

## RRM Modes

| Mode | Fast Loop | Event Loop | Slow Loop | Description |
|------|-----------|------------|-----------|-------------|
| `none` | ❌ | ❌ | ❌ | No optimization (baseline) |
| `fast_loop` | ✅ | ❌ | ❌ | Only Fast Loop active |
| `event_loop` | ❌ | ✅ | ❌ | Only Event Loop active |
| `slow_loop` | ❌ | ❌ | ✅ | Only Slow Loop active |
| `full` | ✅ | ✅ | ✅ | All loops enabled |

---

## QoE Computation

QoE is computed using role-based weights from the Policy Engine:

```
QoE = ws×Signal + wt×Throughput + wr×Reliability + wl×Latency + wa×Activity
```

Where weights are defined in `slo_catalog.yml` for each role (VO, BE, etc.)

---

## Output Files

After running an A/B test, you'll get:

```
ab_test_results_comparison.png      # Bar charts comparing variants
ab_test_results_distribution.png    # QoE distribution histograms
ab_test_results.json                 # Full results in JSON
ab_test_results.csv                  # Summary in CSV
```

---

## API Reference

### QoEMonitor

```python
qoe_monitor = QoEMonitor(slo_catalog, policy_engine)

# Update with current network state
qoe_monitor.update(step, clients, access_points)

# Get statistics
stats = qoe_monitor.get_network_qoe_stats()
print(f"Mean QoE: {stats.mean:.3f}")
print(f"P95 QoE: {stats.p95:.3f}")

# Get time series
history = qoe_monitor.get_time_series(window=100)

# Export to CSV
qoe_monitor.export_metrics("qoe_data.csv")
```

### ABTestFramework

```python
ab_test = ABTestFramework(variant_a, variant_b)

# Collect metrics during simulation
ab_test.collect_metrics("A", qoe=0.85, throughput=50.0, retry_rate=5.0, interference=0.3)

# Compute comparison
report = ab_test.compute_comparison()

# Print report
report.print_report()

# Export results
ab_test.export_results("results.json", format="json")
ab_test.export_results("results.csv", format="csv")
```

### QoEDashboard

```python
dashboard = QoEDashboard()

# Show real-time plot
dashboard.create_realtime_plot(qoe_monitor)

# Show A/B comparison
dashboard.show_dashboard(qoe_monitor, ab_framework)

# Save plots
dashboard.save_report("output", qoe_monitor, ab_framework)
```

---

## Example Results

```
==========================================================================
A/B TEST COMPARISON REPORT
==========================================================================
Variant A: No RRM (Baseline)
Variant B: Fast Loop Enabled

QoE Metrics:
  Variant A Mean: 0.6234
  Variant B Mean: 0.7891
  Improvement:    +26.58%
  P-value:        0.0012
  Significant:    ✓ YES

Throughput:
  Variant A: 245.32 Mbps
  Variant B: 312.45 Mbps
  Improvement: +27.36%

Retry Rate:
  Variant A: 12.45%
  Variant B: 7.23%
  Improvement: +41.93%

Actions Taken:
  Variant A: 0
  Variant B: 247
==========================================================================
```

---

## Tips

1. **Warmup Period**: Consider excluding the first ~100 steps from comparison to allow network stabilization
2. **Statistical Significance**: P-value < 0.05 indicates significant difference
3. **Multiple Runs**: Run tests multiple times with different seeds for robustness
4. **Role Distribution**: Ensure both variants have same client role distribution

---

## Troubleshooting

**Q: QoE values are all 0**
A: Ensure `slo_catalog.yml` exists and Policy Engine is initialized

**Q: No visualization showing**
A: Install matplotlib: `pip install matplotlib`

**Q: "scipy not found" warning**
A: Install scipy for statistical tests: `pip install scipy`

**Q: EnhancedRRMEngine errors**
A: Make sure all loop controllers are properly initialized

---

## Next Steps

- Try comparing different RRM modes
- Test with different network topologies
- Analyze per-role QoE improvements
- Export results for reporting
