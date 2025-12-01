# Spectrum Sensing Pipeline Logs

This repository contains logs and analysis from a spectrum sensing pipeline designed to monitor and classify wireless activity in the 2.4 GHz ISM band.

## Overview

The pipeline performs radio frequency (RF) spectrum scanning across multiple channels, detecting and classifying non-WiFi wireless protocols (Zigbee and BLE) using machine learning classifiers. The data captures real-world wireless activity over approximately 13.8 hours of wall-clock time.

## Dataset Summary

- **Total Epochs**: 3,647
- **Total Scans**: 21,882
- **Monitoring Duration**: ~13.8 hours (49,815 seconds)
- **Total Dwell Time**: ~2,152 hours (7,749,875 seconds cumulative)
- **Frequencies Monitored**: 3 channels
  - 2412 MHz (Channel 1): 7,294 scans
  - 2437 MHz (Channel 6): 7,294 scans
  - 2462 MHz (Channel 11): 7,294 scans

## Detected Protocols

### Zigbee Activity
- **Total Detections**: 32,146
- **Average Duty Cycle**: 1.21%
- **Duty Cycle Range**: 0.03% - 72.8%
- **Average Confidence**: 99.3%
- **Average Bandwidth**: 13.99 MHz
- **Bandwidth Range**: 312.5 kHz - 17.5 MHz

### Bluetooth Low Energy (BLE)
- **Total Detections**: 138,118
- **Average Duty Cycle**: 0.46%
- **Duty Cycle Range**: 0.03% - 100%
- **Average Confidence**: 86.7%
- **Confidence Range**: 53.3% - 100%
- **Average Bandwidth**: 9.30 MHz
- **Bandwidth Range**: 312.5 kHz - 17.5 MHz

## Signal Quality Metrics

- **SNR (Signal-to-Noise Ratio)**
  - Average: 78.5 dB
  - Range: 69.9 dB - 80.7 dB
  
- **Noise Floor**
  - Average: -51.8 dBm
  - Range: -52.2 dBm to -44.2 dBm

## Reward Statistics

The pipeline includes a reinforcement learning component that evaluates scanning decisions:

- **Total Rewards Recorded**: 10,941
- **Average Reward**: -0.67
- **Reward Range**: -1.6 to -0.1

Negative rewards suggest the system is optimizing for minimizing interference or maximizing efficiency in spectrum usage.

## File Structure

```
.
├── README.md                      # This file
├── schema.json                    # JSON schema for log data structure
├── scripts/
│   ├── gen.py                    # Log generation script
│   ├── genschema.py              # Schema generation script
│   └── genstats.py               # Statistics computation script
├── sensing-pipeline-logs.json    # Raw sensing data logs
└── stats.json                    # Aggregated statistics
```

## Data Schema

Each epoch in `sensing-pipeline-logs.json` contains:

- **scans**: Array of individual spectrum scans with:
  - Timestamp
  - Center frequency
  - Dwell time
  - Channel clear assessment (CCA) busy percentage
  - Mean energy
  - WiFi airtime percentage
  - Noise floor
  - SNR
  - Non-WiFi classifier predictions (Zigbee/BLE detections)

- **epoch_end_timestamp**: End time of the epoch
- **epoch_t**: Epoch number
- **actual_rewards**: Reward values for each frequency
- **reward_this_epoch**: Total reward for this epoch
- **equal_rewards**: Baseline reward with equal frequency distribution
- **random_rewards**: Baseline reward with random selection
- **optimal**: Indicator for optimal channel selection

See `schema.json` for complete data structure details.

## Usage

### Analyzing the Data

```bash
# Generate statistics from logs
python scripts/genstats.py

# View statistics
cat stats.json | jq .
```

### Key Insights

1. **BLE Dominance**: BLE activity is ~4.3x more frequent than Zigbee (138K vs 32K detections)
2. **High Confidence**: Zigbee detections have exceptional classifier confidence (99.3%)
3. **Variable Duty Cycles**: Both protocols show highly variable channel utilization
4. **Excellent SNR**: Average SNR of 78.5 dB indicates strong signal reception
5. **Balanced Scanning**: Equal distribution across the three monitored channels

## Applications

This dataset is suitable for:

- Wireless protocol classification research
- Spectrum occupancy analysis
- Coexistence studies in the 2.4 GHz band
- Reinforcement learning for dynamic spectrum access
- IoT device activity profiling
- Network planning and optimization

## License
The MIT License (MIT)

Copyright © 2025 Team15@InterIIT14.0

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE

## Contact
See you after the competition.
