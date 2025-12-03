"""
SLO Catalog module for role-based service level objectives.

This module provides YAML-based configuration for roles with:
- QoS component weights (ws, wt, wr, wl, wa)
- Enforcement rules with thresholds and actions
- Regulatory constraints
- Long-term KPIs
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import yaml
import os


@dataclass
class EnforcementRule:
    """Enforcement rule for a specific metric."""
    operator: str  # ">=", "<=", "==", etc.
    value: float
    authority: str  # "P95", "P50", "mean"
    action: str  # "IncreaseTxPower", "Steer", "Penalty", etc.


@dataclass
class RoleConfig:
    """Configuration for a single role."""
    role_id: str
    display_name: str
    purpose: str
    qos_weights: Dict[str, float]  # ws, wt, wr, wl, wa
    enforcement: Dict[str, EnforcementRule]
    regulatory: Dict[str, Any]
    long_term_kpis: Dict[str, float]


@dataclass
class SLOCatalogConfig:
    """Complete SLO catalog configuration."""
    version: str
    description: str
    global_config: Dict[str, Any]
    roles: Dict[str, RoleConfig]
    long_term_evaluation: Dict[str, Any]
    policy_notes: List[str]


class SLOCatalog:
    """
    SLO Catalog for managing role-based service level objectives.
    
    Loads configuration from YAML file and provides methods to:
    - Get role configurations
    - Get QoS weights for roles
    - Evaluate enforcement rules
    - Check regulatory compliance
    """
    
    def __init__(self, yaml_path: str):
        """
        Initialize SLO Catalog from YAML file.
        
        Args:
            yaml_path: Path to slo_catalog.yml file
        """
        self.yaml_path = yaml_path
        self.config = self._load_yaml(yaml_path)
        self.roles = self.config.roles
        self.global_config = self.config.global_config
        self.long_term_evaluation = self.config.long_term_evaluation
        self.policy_notes = self.config.policy_notes
    
    def _load_yaml(self, path: str) -> SLOCatalogConfig:
        """
        Load and parse YAML SLO catalog.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Parsed SLOCatalogConfig
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"SLO catalog not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return self._parse_config(data)
    
    def _parse_config(self, data: Dict) -> SLOCatalogConfig:
        """
        Parse YAML data into SLOCatalogConfig.
        
        Args:
            data: Parsed YAML dictionary
            
        Returns:
            SLOCatalogConfig object
        """
        # Parse roles
        roles = {}
        for role_id, role_data in data.get('roles', {}).items():
            # Parse enforcement rules
            enforcement = {}
            for metric_name, rule_data in role_data.get('enforcement', {}).items():
                enforcement[metric_name] = EnforcementRule(
                    operator=rule_data['operator'],
                    value=rule_data['value'],
                    authority=rule_data['authority'],
                    action=rule_data['action']
                )
            
            roles[role_id] = RoleConfig(
                role_id=role_id,
                display_name=role_data.get('display_name', role_id),
                purpose=role_data.get('purpose', ''),
                qos_weights=role_data.get('qos_weights', {}),
                enforcement=enforcement,
                regulatory=role_data.get('regulatory', {}),
                long_term_kpis=role_data.get('long_term_kpis', {})
            )
        
        return SLOCatalogConfig(
            version=data.get('version', '1.0'),
            description=data.get('description', ''),
            global_config=data.get('global', {}),
            roles=roles,
            long_term_evaluation=data.get('long_term_evaluation', {}),
            policy_notes=data.get('policy_notes', [])
        )
    
    def get_role(self, role_id: str) -> Optional[RoleConfig]:
        """
        Get role configuration by ID.
        
        Args:
            role_id: Role identifier (e.g., "ExamHallStrict", "VO", "BE")
            
        Returns:
            RoleConfig object or None if not found
        """
        return self.roles.get(role_id)
    
    def list_roles(self) -> List[str]:
        """
        Get list of all available role IDs.
        
        Returns:
            List of role IDs
        """
        return list(self.roles.keys())
    
    def get_qos_weights(self, role_id: str) -> Dict[str, float]:
        """
        Get QoS component weights for a role.
        
        Args:
            role_id: Role identifier
            
        Returns:
            Dictionary with QoS weights: {'ws': ..., 'wt': ..., 'wr': ..., 'wl': ..., 'wa': ...}
            Returns empty dict if role not found
        """
        role = self.get_role(role_id)
        return role.qos_weights if role else {}
    
    def evaluate_enforcement(self, role_id: str, metrics: Dict[str, float]) -> List[str]:
        """
        Evaluate enforcement rules and return triggered actions.
        
        Compares metrics against role's enforcement thresholds and
        returns list of actions that should be triggered.
        
        Args:
            role_id: Role identifier
            metrics: Dictionary of metric name -> value
                     e.g., {'RSSI_dBm': -70, 'Retry_pct': 5.0}
        
        Returns:
            List of action strings that should be triggered
            (e.g., ['IncreaseTxPower', 'Steer'])
        """
        role = self.get_role(role_id)
        if not role:
            return []
        
        triggered_actions = []
        
        for metric_name, rule in role.enforcement.items():
            if metric_name not in metrics:
                continue
            
            metric_value = metrics[metric_name]
            violated = False
            
            # Evaluate operator
            if rule.operator == ">=":
                violated = metric_value < rule.value
            elif rule.operator == "<=":
                violated = metric_value > rule.value
            elif rule.operator == "==":
                violated = abs(metric_value - rule.value) > 1e-6
            elif rule.operator == "!=":
                violated = abs(metric_value - rule.value) < 1e-6
            
            if violated:
                # Parse multiple actions separated by /
                actions = rule.action.split('/')
                triggered_actions.extend(actions)
        
        return triggered_actions
    
    def check_regulatory_compliance(self, role_id: str, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Check if configuration meets regulatory constraints for a role.
        
        Args:
            role_id: Role identifier
            config: Configuration dictionary to validate
                    e.g., {'channel_width_MHz': 80, 'tx_power_dBm': 25}
        
        Returns:
            Tuple of (is_compliant: bool, violations: List[str])
        """
        role = self.get_role(role_id)
        if not role:
            return (True, [])
        
        violations = []
        
        # Check channel width
        if 'channel_width_MHz' in config:
            max_width = role.regulatory.get('max_channel_width_MHz', float('inf'))
            if config['channel_width_MHz'] > max_width:
                violations.append(
                    f"Channel width {config['channel_width_MHz']} MHz exceeds "
                    f"maximum {max_width} MHz for role {role_id}"
                )
        
        # Future: add more regulatory checks (power limits, etc.)
        
        is_compliant = len(violations) == 0
        return (is_compliant, violations)
    
    def get_global_normalizers(self) -> Dict[str, float]:
        """
        Get global normalizer values.
        
        Returns:
            Dictionary of normalizer values (RSSI_min, RSSI_max, etc.)
        """
        return self.global_config.get('normalizers', {})
    
    def get_global_defaults(self) -> Dict[str, Any]:
        """
        Get global default values.
        
        Returns:
            Dictionary of default values
        """
        return self.global_config.get('defaults', {})
    
    def print_role(self, role_id: str) -> None:
        """
        Pretty-print role configuration.
        
        Args:
            role_id: Role identifier
        """
        role = self.get_role(role_id)
        if not role:
            print(f"Role '{role_id}' not found")
            return
        
        print(f"\n{'='*60}")
        print(f"Role: {role.display_name} ({role.role_id})")
        print(f"{'='*60}")
        print(f"Purpose: {role.purpose}")
        
        print(f"\nQoS Weights:")
        for key, value in role.qos_weights.items():
            print(f"  {key}: {value:.2f}")
        
        print(f"\nEnforcement Rules:")
        for metric, rule in role.enforcement.items():
            print(f"  {metric}: {rule.operator} {rule.value} ({rule.authority}) → {rule.action}")
        
        print(f"\nRegulatory:")
        for key, value in role.regulatory.items():
            print(f"  {key}: {value}")
        
        print(f"\nLong-term KPIs:")
        for key, value in role.long_term_kpis.items():
            print(f"  {key}: {value}")
        print()
    
    def print_all_roles(self) -> None:
        """Print all roles in the catalog."""
        print(f"\nSLO Catalog v{self.config.version}")
        print(f"{self.config.description}\n")
        print(f"Available Roles: {', '.join(self.list_roles())}")
        
        for role_id in sorted(self.list_roles()):
            self.print_role(role_id)
