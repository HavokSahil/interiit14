# `pilot-site-logs` — Log Directory Reference

This directory contains **raw telemetry, measurement batches, and steering-related logs** collected from the pilot deployment. The structure is fully time-scoped and organized for efficient ingestion, replay, offline QoE analysis, and steering-event forensics.

```
pilot-site-logs/
├── 2025/
│   └── 11/
│       └── 15/
│           ├── 00/
│           │   ├── _batch_XXXX_metadata.json
│           │   ├── BM_YYYYMMDD_HHMMSS_XXXX.json
│           │   ├── BSSTM_YYYYMMDD_HHMMSS_XXXX.json
│           │   └── ...
│           ├── bssTmStat_20251115_022013
│           ├── bssTmStat_20251115_075750
│           ├── ...
└── _metadata/
```

## 1. Time-Scoped Log Root

Each day of logs is stored under:

```
pilot-site-logs/YYYY/MM/DD/HH/
```

Example:

```
2025/11/15/00/
```

This ensures deterministic replay ordering and compatibility with time-sliced queries.

## 2. Batch Metadata Files

Files like:

```
_batch_0000_metadata.json
_batch_0001_metadata.json
...
```

These describe batch windows containing measurement and steering-related events.

## 3. Beam Measurement Logs (BM\_\*.json)

```
BM_YYYYMMDD_HHMMSS_XXXX.json
```

Beacon Measurement Reports used for QoE reconstruction and neighbor ranking.

## 4. BSS Transition Management Logs (BSSTM\_\*.json)

```
BSSTM_YYYYMMDD_HHMMSS_XXXX.json
```

Captures 802.11v steering frames, client responses, and ranking decisions.

## 5. `bssTmStat_*` Daily Summary Files

Periodic snapshots summarizing steering attempts, outcomes, and ranking states.

## 6. `_metadata/` Repository-Level Metadata

Contains schema versions, retention policy markers, and site-level configurations.

