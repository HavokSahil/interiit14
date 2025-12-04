"""
Emergency Channel Selector for Event Loop.

Implements intelligent channel selection for DFS radar events
and interference bursts with multi-criteria scoring.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

from datatype import AccessPoint, Interferer


@dataclass
class ChannelScore:
    """Score for a candidate channel"""
    channel: int
    interference_score: float  # 0-100, lower is better
    neighbor_overlap_score: float  # 0-100, lower is better
    client_compat_score: float  # 0-100, lower is better
    dfs_penalty: float  # 0 or 50
    total_score: float = 0.0
    
    def compute_total(self, weights: Optional[Dict[str, float]] = None):
        """Compute weighted total score"""
        if weights is None:
            weights = {
                'interference': 0.4,
                'neighbor_overlap': 0.3,
                'client_compat': 0.2,
                'dfs': 0.1
            }
        
        self.total_score = (
            weights['interference'] * self.interference_score +
            weights['neighbor_overlap'] * self.neighbor_overlap_score +
            weights['client_compat'] * self.client_compat_score +
            weights['dfs'] * self.dfs_penalty
        )


class EmergencyChannelSelector:
    """
    Select best emergency channel based on multiple criteria.
    """
    
    # WiFi 2.4 GHz channels
    CHANNELS_2G = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    NON_OVERLAP_2G = [1, 6, 11]
    
    # WiFi 5 GHz channels (subset)
    CHANNELS_5G = [36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112,
                   116, 120, 124, 128, 132, 136, 140, 149, 153, 157, 161, 165]
    
    # DFS channels (5 GHz)
    DFS_CHANNELS_5G = [52, 56, 60, 64, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140]
    
    def __init__(self, prefer_non_dfs: bool = True):
        """
        Initialize Emergency Channel Selector.
        
        Args:
            prefer_non_dfs: Prefer non-DFS channels when available
        """
        self.prefer_non_dfs = prefer_non_dfs
        
        # DFS channel clearance status (for simulation)
        self.dfs_cleared: Dict[int, bool] = {}
    
    def select_channel(self,
                      current_channel: int,
                      radio: str,
                      access_points: List[AccessPoint],
                      interferers: Optional[List[Interferer]] = None,
                      excluded_channels: Optional[List[int]] = None,
                      interference_data: Optional[Dict[int, float]] = None) -> int:
        """
        Select best emergency channel.
        
        Args:
            current_channel: Current channel to avoid
            radio: "2g" or "5g"
            access_points: List of all APs (for neighbor overlap)
            interferers: List of interferers
            excluded_channels: Channels to exclude (e.g., DFS detected)
            interference_data: Per-channel interference measurements
            
        Returns:
            Best channel number
        """
        # Get candidate channels
        if radio == "2g":
            candidates = self.CHANNELS_2G.copy()
            fallback = 1
        else:  # 5g
            candidates = self.CHANNELS_5G.copy()
            fallback = 36
        
        # Remove excluded channels
        if excluded_channels:
            candidates = [c for c in candidates if c not in excluded_channels]
        
        # Remove current channel
        candidates = [c for c in candidates if c != current_channel]
        
        # If no candidates, use fallback
        if not candidates:
            return fallback
        
        # Score each candidate
        scores = []
        for channel in candidates:
            score = self._score_channel(
                channel,
                radio,
                access_points,
                interferers,
                interference_data
            )
            scores.append(score)
        
        # Sort by total score (lower is better)
        scores.sort(key=lambda s: s.total_score)
        
        # Return best channel
        return scores[0].channel
    
    def _score_channel(self,
                      channel: int,
                      radio: str,
                      access_points: List[AccessPoint],
                      interferers: Optional[List[Interferer]],
                      interference_data: Optional[Dict[int, float]]) -> ChannelScore:
        """Score a candidate channel"""
        
        # 1. Interference score
        interference_score = self._compute_interference_score(
            channel, interferers, interference_data
        )
        
        # 2. Neighbor overlap score
        neighbor_overlap_score = self._compute_neighbor_overlap_score(
            channel, access_points
        )
        
        # 3. Client compatibility score (assume all clients support all channels)
        client_compat_score = 0.0  # Perfect compatibility
        
        # 4. DFS penalty
        dfs_penalty = 0.0
        if radio == "5g" and channel in self.DFS_CHANNELS_5G:
            if self.prefer_non_dfs:
                dfs_penalty = 50.0  # Heavy penalty
            else:
                dfs_penalty = 10.0  # Light penalty
        
        # Create score object
        score = ChannelScore(
            channel=channel,
            interference_score=interference_score,
            neighbor_overlap_score=neighbor_overlap_score,
            client_compat_score=client_compat_score,
            dfs_penalty=dfs_penalty
        )
        
        score.compute_total()
        return score
    
    def _compute_interference_score(self,
                                   channel: int,
                                   interferers: Optional[List[Interferer]],
                                   interference_data: Optional[Dict[int, float]]) -> float:
        """
        Compute interference score for a channel.
        
        Returns:
            Score 0-100 (lower is better)
        """
        # Use measured data if available
        if interference_data and channel in interference_data:
            # Assume interference_data is CCA busy % (0-100)
            return interference_data[channel]
        
        # Otherwise, estimate from interferers
        if interferers:
            interference = 0.0
            for interferer in interferers:
                overlap = self._channel_overlap(channel, interferer.channel)
                interference += overlap * interferer.duty_cycle
            
            # Normalize to 0-100
            return min(interference * 100, 100.0)
        
        # No data, assume clean
        return 0.0
    
    def _compute_neighbor_overlap_score(self,
                                       channel: int,
                                       access_points: List[AccessPoint]) -> float:
        """
        Compute co-channel and adjacent-channel interference from neighbor APs.
        
        Returns:
            Score 0-100 (lower is better)
        """
        overlap = 0.0
        
        for ap in access_points:
            if ap.channel == channel:
                # Co-channel: heavy penalty
                overlap += 100.0
            else:
                # Adjacent channel: based on overlap
                channel_overlap = self._channel_overlap(channel, ap.channel)
                overlap += channel_overlap * 50.0
        
        # Normalize
        if len(access_points) > 0:
            overlap /= len(access_points)
        
        return min(overlap, 100.0)
    
    def _channel_overlap(self, ch1: int, ch2: int) -> float:
        """
        Compute overlap factor between two channels.
        
        Returns:
            Overlap 0.0-1.0
        """
        # 2.4 GHz: 5 MHz spacing, 20 MHz bandwidth
        # Channels are 5 MHz apart, bandwidth is 20 MHz
        # So 4 channels apart = no overlap
        
        center1 = self._channel_to_freq(ch1)
        center2 = self._channel_to_freq(ch2)
        
        freq_diff = abs(center1 - center2)
        
        # Assume 20 MHz bandwidth
        bandwidth = 20.0
        
        # Overlap calculation
        if freq_diff >= bandwidth:
            return 0.0
        else:
            return 1.0 - (freq_diff / bandwidth)
    
    def _channel_to_freq(self, channel: int) -> float:
        """Convert channel number to center frequency in MHz"""
        if 1 <= channel <= 14:
            # 2.4 GHz band
            return 2412 + (channel - 1) * 5
        elif 36 <= channel <= 165:
            # 5 GHz band
            return 5000 + channel * 5
        else:
            return 2412  # Default
    
    def mark_dfs_cleared(self, channel: int):
        """Mark a DFS channel as cleared"""
        self.dfs_cleared[channel] = True
    
    def mark_dfs_detected(self, channel: int):
        """Mark a DFS channel as having radar detected"""
        self.dfs_cleared[channel] = False
    
    def is_dfs_channel(self, channel: int) -> bool:
        """Check if channel is a DFS channel"""
        return channel in self.DFS_CHANNELS_5G
    
    def get_safe_fallback(self, radio: str) -> int:
        """Get safe fallback channel"""
        if radio == "2g":
            return 1  # Channel 1
        else:
            return 36  # Channel 36 (5 GHz, non-DFS)


# Singleton instance for convenience
_channel_selector = EmergencyChannelSelector()


def select_emergency_channel(current_channel: int,
                             radio: str,
                             access_points: List[AccessPoint],
                             interferers: Optional[List[Interferer]] = None,
                             excluded_channels: Optional[List[int]] = None,
                             interference_data: Optional[Dict[int, float]] = None) -> int:
    """
    Convenience function for emergency channel selection.
    
    Args:
        current_channel: Current channel to avoid
        radio: "2g" or "5g"
        access_points: List of all APs
        interferers: List of interferers
        excluded_channels: Channels to exclude
        interference_data: Per-channel interference measurements
        
    Returns:
        Best emergency channel
    """
    return _channel_selector.select_channel(
        current_channel, radio, access_points,
        interferers, excluded_channels, interference_data
    )
