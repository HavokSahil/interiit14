from typing import Tuple
import math

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