# Transition Management System — Repository Overview

This repository contains the full implementation, design artifacts and visualization for Policy Engine. The full implementations, input data, output results and saved model from the Bayesian Optimization module.

## Repository Structure

```
 .
├── Codes/
│   ├── policy_engine.py
│   ├── bo.py
│   ├── warmup.py
│   └── usage/
│       ├── bo_api_usage.py
│       └── warmup_api_usage.py
├── Visualization/
│   ├── Prometheus_Config.yml
│   └── Grafana_Dashboard.json
├── data/
│   └── input_snapshots_50.json
├── models/
│   └── bo_state_siteA.pkl
├── results/
│   ├── suggestions_50.jsonl
│   └── suggestions_50.csv
└── README.md    
            
```

## System Summary

The Policy Engine implements:

- Time of Day calculation 
- Hidden Node Detection and alert
- Threshold based configuration change request  

The Bayesian Optimization Module implements: 

- An API to send measured metrics and obtain configuration suggestions.
- Mechanism to augment data to the safe set and saves it to prevent losing progress.
- Mechanism to save the updated model as a pickle file regularly. 


## Running

```
cd Codes
python3 policy_engine.py
sudo systemctl start grafana-server
sudo systemctl restart prometheus
```
NOTE: Please ensure Prometheus and Grafana are correctly installed. Replace prometheus.yml in local device with Prometheus_Config.yml from repository.

## Status

Introducing specific change requests like Channel_Change, Steer_Clients etc. based on Root Cause Analysis. Exploration parameter integration in Bayesian Optimizer for appropriate configuration selection. 

## License
The MIT License (MIT)

Copyright © 2025 Team15@InterIIT14.0

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE

## Contact
See you after competition
