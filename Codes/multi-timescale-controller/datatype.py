from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from abc import ABC, abstractclassmethod
from collections import deque

@dataclass
class AccessPoint:
    id: int
    x: float
    y: float
    tx_power: float
    channel: int
    bandwidth: float = 20.0
    noise_floor: float = -95.0 # in dBm
    max_throughput: float = 100.0
    inc_energy_ch1: float = float('-inf') # Incident energy on channel 1 in dBm
    inc_energy_ch6: float = float('-inf') # Incident energy on channel 6 in dBm
    inc_energy_ch11: float = float('-inf') # Incident energy on channel 11 in dBm
    obss_pd_threshold: float = -82.0 # OBSS PD threshold in dBm
    cca_busy_percentage: float = 0.0 # Fraction of time energy on operating channel > threshold
    obss_pd_violation_history: deque = field(default_factory=lambda: deque(maxlen=100)) # History of violations
    roam_in_rate: float = 0.0 # Rate of clients roaming into this AP (clients/step)
    roam_in_history: deque = field(default_factory=lambda: deque(maxlen=100)) # Roam-in history
    roam_out_rate: float = 0.0 # Rate of clients roaming out of this AP (clients/step)
    roam_out_history: deque = field(default_factory=lambda: deque(maxlen=100)) # Roam-out history
    p95_throughput: float = 0.0 # 95% of clients get at least this throughput (Mbps)
    p95_retry_rate: float = 0.0 # 95th percentile of client retry rates (0-100%)
    total_allocated_throughput: float = 0.0
    connected_clients: List[int] = field(default_factory=list)
    roam_in: int = 0 # number of clients currently roaming into the AP
    roam_out: int = 0 # number of clients currently roaming out of the AP

@dataclass
class Client:
    id: int
    x: float
    y: float
    demand_mbps: float # client demand in mbps
    associated_ap: Optional[int] = None
    last_assoc_ap: Optional[int] = None
    velocity: float = 1.0
    direction: float = 0.0
    sinr_db: float = 0.0
    max_rate_mbps: float = 0.0 # maximum achievable rate based on the SINR
    throughput_mbps: float = 0.0 # actual allocated througput
    airtime_fraction: float = 0.0
    rssi_dbm: float = 0.0 # Received Signal Strength Indicator in dBm
    retry_rate: float = 0.0 # Packet retry rate as percentage (0-100)

@dataclass
class Interferer:
    id: int
    x: float
    y: float
    tx_power: float
    channel: int
    type: str = "Bluetooth"  # Type of interferer: BLE, Bluetooth, or Microwave
    bandwidth: float = 20.0
    duty_cycle: float = 1.0 # Fraction of time it is transmitting (0.0 to 1.0)
    hopping_enabled: bool = False
    hopping_channels: List[int] = field(default_factory=list)  # List of channels to hop through
    hopping_index: int = 0  # Current index in hopping_channels list

@dataclass
class Environment:
    x_min: float = 0.0
    x_max: float = 100.0
    y_min: float = 0.0
    y_max: float = 100.0
