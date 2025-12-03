# Simulation Environment & Assumptions

This document details the physical models and assumptions used in the wireless network simulation.

## 1. Signal Propagation
The simulation uses a **Log-Distance Path Loss Model** combined with **Multipath Fading**.

- **Path Loss Formula**:
  $$PL(d) = 20 \log_{10}(d) + 20 \log_{10}(f) - 27.55 + 10(n-2)\log_{10}(d)$$
  - $f$: Frequency (2400 MHz)
  - $n$: Path loss exponent (default: 3.0, representing indoor obstructed environment)
  - $d$: Distance in meters

- **Fading**:
  - A Rayleigh fading margin (default: 8.0 dB) is subtracted from the received power to account for signal fluctuations.
  - $P_{rx} = P_{tx} - PL(d) - \text{FadingMargin}$

## 2. Client Mobility
Clients move according to a **Correlated Random Walk** model.

- **Velocity**: Varies smoothly between `min_velocity` (0.5 m/s) and `max_velocity` (2.0 m/s).
- **Direction**: Changes gradually with a random perturbation ($\pm \pi/8$ radians per step).
- **Boundaries**: Clients reflect off the simulation boundaries (elastic collision).

## 3. Client Association
Clients associate with Access Points (APs) based on **Received Signal Strength Indicator (RSSI)**.

- **Logic**: At every step, each client scans all APs and connects to the one providing the highest $P_{rx}$.
- **Roaming**:
  - A "Roam Event" is triggered if the best AP changes from the previous step.
  - Roaming events are tracked per AP (`roam_in`, `roam_out`) and used as features for the GNN.

## 4. Traffic & Throughput
- **Demand**: Each client has a fixed demand (e.g., 5-30 Mbps).
- **Capacity**: Each AP has a maximum throughput capacity (e.g., 150 Mbps).
- **Allocation**:
  - Bandwidth is shared among connected clients.
  - If total demand < capacity, all clients get their full demand.
  - If total demand > capacity, throughput is allocated proportionally or equally (depending on the specific scheduler implementation in `metrics.py`).

## 5. Interference Modeling
Interference is calculated based on **Channel Overlap** and **Signal Strength**.

- **Co-Channel Interference**: 100% overlap if APs are on the same channel.
- **Adjacent Channel Interference**: Modeled by an overlap factor (spectral mask) if channels are close (e.g., Ch 1 and Ch 2).
- **Interference Graph Edge Weight**:
  - Represents the "coupling" or impact of AP $u$ on AP $v$.
  - Calculated as a function of the power received from $u$ at $v$'s location relative to the noise floor and $v$'s own signal strength.
  - **Ground Truth**: Computed using the physical path loss model.
  - **Prediction**: The GNN attempts to predict this weight using only node features.

## 6. General Assumptions
- **2D Plane**: The simulation operates in a 2D coordinate system ($z=0$).
- **Isotropic Antennas**: APs and clients have omnidirectional antennas with 0 dBi gain.
- **Static Environment**: No dynamic obstacles (walls/furniture) are modeled; path loss exponent is constant throughout the area.
