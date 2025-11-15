project-root/
├── main.py
│   └── Uses:
│       ├── rf_scanner_controller.py  → Extracts features
│       ├── ml_online.py              → Dynamic Multi Arm Bandit (RL)
│       ├── change_detection.py       → Generates alerts from extracted features
│       ├── infer_real_capture_airshark.py → Non-WiFi classifier
│       └── Serves API at http://localhost:5000/
│           └── /sensing-data endpoint for sensing output
│
├── rf_scanner_controller.py
│   └── Initial feature extraction logic from IQ samples
│
├── change_detection.py
│   └── Change detection on feature stream
│
├── ml_online.py
│   └── Dynamic Multi Arm Bandit solver (Reinforcement Learning)
│
├── infer_real_capture_airshark.py
│   └── Loads and uses:
│       └── airshark_c45_model.pkl    → Decision tree model for Non-WiFi classification
│
├── airshark_c45_model.pkl
│   └── Decision Tree model used by Non-WiFi classifier
│
├── pre_processed_online.csv
│   └── Dataset for the Multi Arm Bandit solver
│
└── offline_model_artifacts/
    └── Model artifacts for Dynamic Multi Arm Bandit
