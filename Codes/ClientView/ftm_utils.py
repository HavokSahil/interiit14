from typing import Tuple, List, Optional
import math
import numpy as np

# check if a point is within bounds
def is_within_bounds(env: 'Environment', x: float, y: float) -> bool:
    return env.x_min <= x <= env.x_max and env.y_min <= y <= env.y_max

# clip a point to bounds
def clip_to_bounds(env: 'Environment', x: float, y: float) -> Tuple[float, float]:
    x = max(env.x_min, min(x, env.x_max))
    y = max(env.y_min, min(y, env.y_max))
    return x, y

def compute_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance."""
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def dbm_to_mw(dbm: float) -> float:
    """Convert dBm to milliwatts."""
    return 10 ** (dbm / 10.0)

def compute_channel_overlap(ch1: int, ch2: int, bw1: float, bw2: float) -> float:
        """
        Compute channel overlap factor (0-1) for 2.4 GHz WiFi.
        Adjacent channels in 2.4 GHz overlap significantly.
        """
        # Center frequency for each channel: 2407 + 5*channel MHz
        freq1 = 2407 + 5 * ch1
        freq2 = 2407 + 5 * ch2
        freq_separation = abs(freq1 - freq2)
        
        # Channels need ~25 MHz separation for no overlap (5 channels apart)
        # Gaussian-like overlap model
        bandwidth_sum = (bw1 + bw2) / 2.0
        overlap = math.exp(-2.0 * (freq_separation / bandwidth_sum) ** 2)
        
        return overlap

def trilaterate_position(ap_positions: List[Tuple[float, float]], distances: List[float]) -> Optional[Tuple[float, float]]:
    """
    Estimate position (x, y) given a list of AP positions and distances.
    Uses a least-squares approach to solve the intersection of circles.
    
    Args:
        ap_positions: List of (x, y) tuples for APs.
        distances: List of measured distances to each AP.
        
    Returns:
        (x, y) estimated position, or None if not enough data (need >= 3 APs).
    """
    if len(ap_positions) < 3 or len(distances) < 3:
        return None
    
    # We need to solve a system of equations:
    # (x - xi)^2 + (y - yi)^2 = di^2
    # This is non-linear. We can linearize it by subtracting the last equation from the others.
    # A * [x, y]^T = b
    
    n = len(ap_positions)
    A = []
    b = []
    
    x_n, y_n = ap_positions[-1]
    d_n = distances[-1]
    
    for i in range(n - 1):
        x_i, y_i = ap_positions[i]
        d_i = distances[i]
        
        # Linearized form:
        # 2(x_n - x_i)x + 2(y_n - y_i)y = (d_i^2 - d_n^2) - (x_i^2 - x_n^2) - (y_i^2 - y_n^2)
        
        A.append([2 * (x_n - x_i), 2 * (y_n - y_i)])
        b.append((d_i**2 - d_n**2) - (x_i**2 - x_n**2) - (y_i**2 - y_n**2))
        
    try:
        # Solve using NumPy's least squares
        result = np.linalg.lstsq(A, b, rcond=None)
        estimated_pos = result[0]
        return float(estimated_pos[0]), float(estimated_pos[1])
    except np.linalg.LinAlgError:
        return None