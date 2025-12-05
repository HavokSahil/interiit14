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

@dataclass
class Environment:
    x_min: float = 0.0
    x_max: float = 100.0
    y_min: float = 0.0
    y_max: float = 100.0
