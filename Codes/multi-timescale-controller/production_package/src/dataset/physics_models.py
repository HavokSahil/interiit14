"""
Improved Physics Models for Wi-Fi Simulation

Implements realistic Wi-Fi physics:
- SINR calculation (Signal-to-Interference-plus-Noise Ratio)
- MCS (Modulation Coding Scheme) selection
- PER (Packet Error Rate) curves
- Throughput calculation based on MCS
- Log-distance path loss model
"""

import numpy as np
from typing import Dict, Tuple, Optional
from numpy.random import Generator


# 802.11n/ac MCS Tables (simplified)
# Format: (MCS_index, data_rate_Mbps_20MHz, data_rate_Mbps_40MHz, data_rate_Mbps_80MHz, min_SINR_dB)
MCS_TABLE_80211N = [
    (0, 6.5, 13.5, 27.0, 5),   # BPSK 1/2
    (1, 13.0, 27.0, 54.0, 8),   # QPSK 1/2
    (2, 19.5, 40.5, 81.0, 11), # QPSK 3/4
    (3, 26.0, 54.0, 108.0, 14), # 16-QAM 1/2
    (4, 39.0, 81.0, 162.0, 17), # 16-QAM 3/4
    (5, 52.0, 108.0, 216.0, 20), # 64-QAM 2/3
    (6, 58.5, 121.5, 243.0, 23), # 64-QAM 3/4
    (7, 65.0, 135.0, 270.0, 26), # 64-QAM 5/6
]

# PER curves (simplified exponential model)
# PER(SINR) = a * exp(-b * (SINR - threshold))
PER_CURVES = {
    'mcs0': {'a': 0.5, 'b': 0.15, 'threshold': 5},
    'mcs1': {'a': 0.4, 'b': 0.12, 'threshold': 8},
    'mcs2': {'a': 0.35, 'b': 0.10, 'threshold': 11},
    'mcs3': {'a': 0.3, 'b': 0.09, 'threshold': 14},
    'mcs4': {'a': 0.25, 'b': 0.08, 'threshold': 17},
    'mcs5': {'a': 0.2, 'b': 0.07, 'threshold': 20},
    'mcs6': {'a': 0.15, 'b': 0.06, 'threshold': 23},
    'mcs7': {'a': 0.1, 'b': 0.05, 'threshold': 26},
}


def calculate_sinr(
    rssi: float,
    noise_floor: float,
    neighbor_rssi: float = None,
    interference_factor: float = 0.5
) -> float:
    """
    Calculate Signal-to-Interference-plus-Noise Ratio (SINR).
    
    Args:
        rssi: Signal strength (dBm)
        noise_floor: Noise floor (dBm)
        neighbor_rssi: Neighbor AP RSSI (dBm), None if no interference
        interference_factor: Factor for interference contribution (0-1)
        
    Returns:
        SINR in dB
    """
    # Convert to linear scale for calculation
    signal_linear = 10 ** (rssi / 10)  # mW
    noise_linear = 10 ** (noise_floor / 10)  # mW
    
    # Interference from neighbor APs
    if neighbor_rssi is not None and neighbor_rssi > -90:
        interference_linear = 10 ** (neighbor_rssi / 10) * interference_factor
    else:
        interference_linear = 0
    
    # Total interference + noise
    total_interference = noise_linear + interference_linear
    
    # SINR in linear scale
    if total_interference > 0:
        sinr_linear = signal_linear / total_interference
        sinr_db = 10 * np.log10(sinr_linear)
    else:
        sinr_db = 50  # Very high if no interference
    
    return np.clip(sinr_db, -10, 50)  # Reasonable range


def select_mcs(sinr: float, channel_width: float) -> int:
    """
    Select Modulation Coding Scheme based on SINR.
    
    Args:
        sinr: SINR in dB
        channel_width: Channel width in MHz (20, 40, or 80)
        
    Returns:
        MCS index (0-7)
    """
    # Find highest MCS that can be supported
    for mcs_idx in range(len(MCS_TABLE_80211N) - 1, -1, -1):
        _, _, _, _, min_sinr = MCS_TABLE_80211N[mcs_idx]
        if sinr >= min_sinr:
            return mcs_idx
    
    return 0  # Fallback to MCS 0


def calculate_per(sinr: float, mcs: int) -> float:
    """
    Calculate Packet Error Rate using PER curves.
    
    Args:
        sinr: SINR in dB
        mcs: MCS index (0-7)
        
    Returns:
        PER (0-1)
    """
    if mcs < 0 or mcs >= len(PER_CURVES):
        mcs = 0
    
    curve = PER_CURVES[f'mcs{mcs}']
    a = curve['a']
    b = curve['b']
    threshold = curve['threshold']
    
    # Exponential PER model
    if sinr < threshold:
        per = a
    else:
        per = a * np.exp(-b * (sinr - threshold))
    
    return np.clip(per, 0.001, 0.5)  # Reasonable PER range


def calculate_throughput(
    sinr: float,
    channel_width: float,
    mcs: Optional[int] = None,
    per: Optional[float] = None,
    channel_util: float = 0.5,
    frame_aggregation: bool = True
) -> float:
    """
    Calculate throughput based on MCS, PER, and channel conditions.
    
    Args:
        sinr: SINR in dB
        channel_width: Channel width in MHz (20, 40, or 80)
        mcs: MCS index (None = auto-select)
        per: Packet Error Rate (None = calculate from SINR)
        channel_util: Channel utilization (0-1)
        frame_aggregation: Whether A-MPDU aggregation is used
        
    Returns:
        Throughput in Mbps
    """
    # Select MCS if not provided
    if mcs is None:
        mcs = select_mcs(sinr, channel_width)
    
    # Get MCS data rate
    mcs_entry = MCS_TABLE_80211N[mcs]
    
    # Select data rate based on channel width
    if channel_width <= 20:
        data_rate = mcs_entry[1]  # 20 MHz
    elif channel_width <= 40:
        data_rate = mcs_entry[2]  # 40 MHz
    else:
        data_rate = mcs_entry[3]  # 80 MHz
    
    # Calculate PER if not provided
    if per is None:
        per = calculate_per(sinr, mcs)
    
    # Throughput = Data Rate * (1 - PER) * efficiency factors
    # Efficiency factors:
    # - Frame aggregation: 0.85-0.95 (A-MPDU overhead)
    # - Channel utilization: (1 - util) penalty
    # - Protocol overhead: ~0.9 (802.11 overhead)
    
    aggregation_factor = 0.9 if frame_aggregation else 0.7
    util_factor = 1.0 - (channel_util * 0.3)  # Up to 30% penalty
    protocol_overhead = 0.9
    
    throughput = data_rate * (1 - per) * aggregation_factor * util_factor * protocol_overhead
    
    return np.clip(throughput, 1.0, 500.0)


def calculate_edge_throughput(
    avg_throughput: float,
    sinr: float,
    edge_sinr_penalty: float = 5.0
) -> float:
    """
    Calculate edge client throughput (P10 percentile).
    
    Edge clients typically have:
    - Lower SINR (5-10 dB worse)
    - Lower MCS (fallback)
    - Higher PER
    
    Args:
        avg_throughput: Average throughput
        sinr: Average SINR
        edge_sinr_penalty: SINR penalty for edge clients (dB)
        
    Returns:
        Edge throughput in Mbps
    """
    # Edge clients have lower SINR
    edge_sinr = sinr - edge_sinr_penalty
    
    # Select lower MCS for edge clients
    edge_mcs = select_mcs(edge_sinr, 20)  # Assume 20 MHz for edge
    avg_mcs = select_mcs(sinr, 20)
    
    # MCS penalty
    mcs_penalty = (edge_mcs + 1) / (avg_mcs + 1)
    
    # PER penalty (edge clients have higher PER)
    edge_per = calculate_per(edge_sinr, edge_mcs)
    avg_per = calculate_per(sinr, avg_mcs)
    per_penalty = (1 - edge_per) / (1 - avg_per + 1e-6)
    
    # Combined penalty
    edge_throughput = avg_throughput * mcs_penalty * per_penalty * 0.7  # Additional 30% penalty
    
    return np.clip(edge_throughput, 0.5, avg_throughput * 0.8)


def log_distance_path_loss(
    distance: float,
    reference_distance: float = 1.0,
    path_loss_exponent: float = 3.0,
    reference_loss: float = 40.0,
    shadowing_std: float = 6.0,
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Calculate path loss using log-distance model with shadowing.
    
    PL(d) = PL(d₀) + 10n*log₁₀(d/d₀) + Xσ
    
    Args:
        distance: Distance in meters
        reference_distance: Reference distance (1m)
        path_loss_exponent: Path loss exponent (2=free space, 3-4=indoor)
        reference_loss: Path loss at reference distance (dB)
        shadowing_std: Shadowing standard deviation (dB)
        rng: Random number generator (None = use numpy default)
        
    Returns:
        Path loss in dB
    """
    if distance <= 0:
        distance = 0.1  # Minimum distance
    
    # Log-distance path loss
    path_loss = reference_loss + 10 * path_loss_exponent * np.log10(distance / reference_distance)
    
    # Add shadowing (log-normal)
    if rng is not None:
        shadowing = rng.normal(0, shadowing_std)
    else:
        shadowing = np.random.normal(0, shadowing_std)
    
    return path_loss + shadowing


def calculate_rssi_from_tx_power(
    tx_power: float,
    distance: float,
    path_loss_exponent: float = 3.0
) -> float:
    """
    Calculate RSSI from Tx power using path loss model.
    
    Args:
        tx_power: Transmit power (dBm)
        distance: Distance in meters
        path_loss_exponent: Path loss exponent
        
    Returns:
        RSSI in dBm
    """
    path_loss = log_distance_path_loss(distance, path_loss_exponent=path_loss_exponent)
    rssi = tx_power - path_loss
    
    return np.clip(rssi, -90, -30)


def calculate_retry_rate_from_per(per: float, max_retries: int = 7) -> float:
    """
    Calculate retry rate from PER.
    
    Retry rate ≈ PER * (1 + retry_limit) / 2 (simplified)
    
    Args:
        per: Packet Error Rate
        max_retries: Maximum retries (802.11 short retry = 7)
        
    Returns:
        Retry rate (0-1)
    """
    # Simplified model: retry rate increases with PER
    retry_rate = per * (1 + max_retries / 2) / 2
    
    return np.clip(retry_rate, 0.01, 0.30)

