"""
Configuration Engine for RRMEngine.

This module provides configuration management for access points:
- Store current and proposed configurations
- Validate configuration changes
- Apply configurations to APs
- Track configuration history with rollback support
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from datatype import AccessPoint
import copy


@dataclass
class APConfig:
    """Configuration for a single access point."""
    ap_id: int
    channel: int
    tx_power: float
    bandwidth: float = 20.0
    obss_pd_threshold: float = -82.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'ap_id': self.ap_id,
            'channel': self.channel,
            'tx_power': self.tx_power,
            'bandwidth': self.bandwidth,
            'obss_pd_threshold': self.obss_pd_threshold
        }


@dataclass
class NetworkConfig:
    """Complete network configuration."""
    timestamp: float
    ap_configs: Dict[int, APConfig]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'ap_configs': {ap_id: cfg.to_dict() for ap_id, cfg in self.ap_configs.items()},
            'metadata': self.metadata
        }


class ConfigEngine:
    """
    Configuration Engine for managing AP configurations.
    
    Responsibilities:
    - Get current network configuration
    - Build and validate new configurations
    - Apply configurations to APs
    - Maintain configuration history
    - Support rollback
    """
    
    def __init__(self, access_points: List[AccessPoint]):
        """
        Initialize Configuration Engine.
        
        Args:
            access_points: List of access points to manage
        """
        self.aps: Dict[int, AccessPoint] = {ap.id: ap for ap in access_points}
        self.config_history: List[NetworkConfig] = []
        self.max_history_size = 10
        
        # Configuration constraints
        self.allowed_channels = [1, 6, 11]
        self.min_power = 10.0  # dBm
        self.max_power = 30.0  # dBm
        self.allowed_bandwidths = [20.0, 40.0, 80.0, 160.0]
        
        # Save initial configuration
        initial_config = self.get_current_config()
        self.config_history.append(initial_config)
    
    def get_current_config(self) -> NetworkConfig:
        """
        Get current network configuration from APs.
        
        Returns:
            NetworkConfig with current AP settings
        """
        import time
        
        ap_configs = {}
        for ap_id, ap in self.aps.items():
            ap_configs[ap_id] = APConfig(
                ap_id=ap_id,
                channel=ap.channel,
                tx_power=ap.tx_power,
                bandwidth=ap.bandwidth,
                obss_pd_threshold=ap.obss_pd_threshold
            )
        
        return NetworkConfig(
            timestamp=time.time(),
            ap_configs=ap_configs,
            metadata={'source': 'current'}
        )
    
    def build_channel_config(self, ap_id: int, channel: int) -> APConfig:
        """
        Build configuration for changing an AP's channel.
        
        Args:
            ap_id: Access point identifier
            channel: New channel number
            
        Returns:
            APConfig with new channel
        """
        if ap_id not in self.aps:
            raise ValueError(f"AP {ap_id} not found")
        
        ap = self.aps[ap_id]
        return APConfig(
            ap_id=ap_id,
            channel=channel,
            tx_power=ap.tx_power,
            bandwidth=ap.bandwidth,
            obss_pd_threshold=ap.obss_pd_threshold
        )
    
    def build_power_config(self, ap_id: int, power: float) -> APConfig:
        """
        Build configuration for changing an AP's transmit power.
        
        Args:
            ap_id: Access point identifier
            power: New transmit power (dBm)
            
        Returns:
            APConfig with new power
        """
        if ap_id not in self.aps:
            raise ValueError(f"AP {ap_id} not found")
        
        ap = self.aps[ap_id]
        return APConfig(
            ap_id=ap_id,
            channel=ap.channel,
            tx_power=power,
            bandwidth=ap.bandwidth,
            obss_pd_threshold=ap.obss_pd_threshold
        )
    
    def build_network_config(self, ap_configs: List[APConfig], metadata: Dict[str, Any] = None) -> NetworkConfig:
        """
        Build complete network configuration from list of AP configs.
        
        Args:
            ap_configs: List of APConfig instances
            metadata: Optional metadata
            
        Returns:
            NetworkConfig instance
        """
        import time
        
        config_dict = {cfg.ap_id: cfg for cfg in ap_configs}
        
        return NetworkConfig(
            timestamp=time.time(),
            ap_configs=config_dict,
            metadata=metadata or {}
        )
    
    def validate_config(self, config: NetworkConfig) -> tuple[bool, List[str]]:
        """
        Validate network configuration against constraints.
        
        Args:
            config: NetworkConfig to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        for ap_id, ap_config in config.ap_configs.items():
            # Check AP exists
            if ap_id not in self.aps:
                errors.append(f"AP {ap_id} does not exist")
                continue
            
            # Validate channel
            if ap_config.channel not in self.allowed_channels:
                errors.append(
                    f"AP {ap_id}: Invalid channel {ap_config.channel}. "
                    f"Allowed: {self.allowed_channels}"
                )
            
            # Validate power
            if not (self.min_power <= ap_config.tx_power <= self.max_power):
                errors.append(
                    f"AP {ap_id}: Power {ap_config.tx_power} dBm out of range "
                    f"[{self.min_power}, {self.max_power}]"
                )
            
            # Validate bandwidth
            if ap_config.bandwidth not in self.allowed_bandwidths:
                errors.append(
                    f"AP {ap_id}: Invalid bandwidth {ap_config.bandwidth} MHz. "
                    f"Allowed: {self.allowed_bandwidths}"
                )
        
        is_valid = len(errors) == 0
        return (is_valid, errors)
    
    def apply_config(self, config: NetworkConfig) -> bool:
        """
        Apply network configuration to APs.
        
        Args:
            config: NetworkConfig to apply
            
        Returns:
            True if applied successfully, False otherwise
        """
        # Validate first
        is_valid, errors = self.validate_config(config)
        if not is_valid:
            print(f"Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        # Save current config to history before applying
        current_config = self.get_current_config()
        self.config_history.append(current_config)
        
        # Trim history if too large
        if len(self.config_history) > self.max_history_size:
            self.config_history = self.config_history[-self.max_history_size:]
        
        # Apply configuration to APs
        for ap_id, ap_config in config.ap_configs.items():
            ap = self.aps[ap_id]
            ap.channel = ap_config.channel
            ap.tx_power = ap_config.tx_power
            ap.bandwidth = ap_config.bandwidth
            ap.obss_pd_threshold = ap_config.obss_pd_threshold
        
        return True
    
    def rollback(self, steps: int = 1) -> bool:
        """
        Rollback to a previous configuration.
        
        Args:
            steps: Number of steps to rollback (default: 1)
            
        Returns:
            True if rollback successful, False otherwise
        """
        if len(self.config_history) < steps + 1:
            print(f"Cannot rollback {steps} steps. Only {len(self.config_history)-1} configs in history.")
            return False
        
        # Get config from history
        target_config = self.config_history[-(steps + 1)]
        
        # Apply it
        success = self.apply_config(target_config)
        
        if success:
            # Remove the configs we rolled back past (including current)
            self.config_history = self.config_history[:-(steps)]
        
        return success
    
    def get_config_diff(self, config1: NetworkConfig, config2: NetworkConfig) -> Dict[int, Dict[str, Any]]:
        """
        Compute difference between two configurations.
        
        Args:
            config1: First configuration
            config2: Second configuration
            
        Returns:
            Dictionary mapping AP ID to dict of changed fields
        """
        diff = {}
        
        all_ap_ids = set(config1.ap_configs.keys()) | set(config2.ap_configs.keys())
        
        for ap_id in all_ap_ids:
            ap_diff = {}
            
            cfg1 = config1.ap_configs.get(ap_id)
            cfg2 = config2.ap_configs.get(ap_id)
            
            if cfg1 is None:
                ap_diff['added'] = True
                ap_diff['config'] = cfg2.to_dict()
            elif cfg2 is None:
                ap_diff['removed'] = True
            else:
                # Compare fields
                if cfg1.channel != cfg2.channel:
                    ap_diff['channel'] = {'old': cfg1.channel, 'new': cfg2.channel}
                if cfg1.tx_power != cfg2.tx_power:
                    ap_diff['tx_power'] = {'old': cfg1.tx_power, 'new': cfg2.tx_power}
                if cfg1.bandwidth != cfg2.bandwidth:
                    ap_diff['bandwidth'] = {'old': cfg1.bandwidth, 'new': cfg2.bandwidth}
                if cfg1.obss_pd_threshold != cfg2.obss_pd_threshold:
                    ap_diff['obss_pd_threshold'] = {'old': cfg1.obss_pd_threshold, 'new': cfg2.obss_pd_threshold}
            
            if ap_diff:
                diff[ap_id] = ap_diff
        
        return diff
    
    def print_config(self, config: NetworkConfig):
        """
        Pretty-print network configuration.
        
        Args:
            config: NetworkConfig to print
        """
        from datetime import datetime
        
        print("\n" + "="*60)
        print(f"NETWORK CONFIGURATION ({datetime.fromtimestamp(config.timestamp)})")
        print("="*60)
        
        for ap_id in sorted(config.ap_configs.keys()):
            cfg = config.ap_configs[ap_id]
            print(f"\nAP {ap_id}:")
            print(f"  Channel:           {cfg.channel}")
            print(f"  Tx Power:          {cfg.tx_power} dBm")
            print(f"  Bandwidth:         {cfg.bandwidth} MHz")
            print(f"  OBSS PD Threshold: {cfg.obss_pd_threshold} dBm")
        
        if config.metadata:
            print(f"\nMetadata: {config.metadata}")
        print()
    
    def print_diff(self, old_config: NetworkConfig, new_config: NetworkConfig):
        """
        Pretty-print configuration changes.
        
        Args:
            old_config: Old configuration
            new_config: New configuration
        """
        diff = self.get_config_diff(old_config, new_config)
        
        if not diff:
            print("No configuration changes")
            return
        
        print("\n" + "="*60)
        print("CONFIGURATION CHANGES")
        print("="*60)
        
        for ap_id in sorted(diff.keys()):
            changes = diff[ap_id]
            print(f"\nAP {ap_id}:")
            
            if 'added' in changes:
                print("  [ADDED]")
            elif 'removed' in changes:
                print("  [REMOVED]")
            else:
                for field, values in changes.items():
                    print(f"  {field}: {values['old']} → {values['new']}")
        print()
    
    def print_status(self):
        """Print configuration engine status."""
        print("\n" + "="*60)
        print("CONFIGURATION ENGINE STATUS")
        print("="*60)
        print(f"Managing {len(self.aps)} Access Points")
        print(f"Configuration History: {len(self.config_history)} entries")
        print(f"\nConstraints:")
        print(f"  Allowed Channels: {self.allowed_channels}")
        print(f"  Power Range: [{self.min_power}, {self.max_power}] dBm")
        print(f"  Allowed Bandwidths: {self.allowed_bandwidths} MHz")
        print()
