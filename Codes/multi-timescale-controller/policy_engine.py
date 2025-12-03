"""
Policy Engine for RRMEngine.

This module provides policy management with SLO catalog integration:
- Client role assignment
- Role-based QoS weight retrieval
- Enforcement rule evaluation
- Optimization objective management
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from slo_catalog import SLOCatalog, RoleConfig
from clientview import ClientQoEResult


@dataclass
class OptimizationObjective:
    """Defines optimization goals for the RRM engine."""
    name: str  # "max_throughput", "max_fairness", "max_qoe", "min_interference"
    weight: float  # Weight for multi-objective optimization
    minimize: bool = False  # If True, minimize; if False, maximize


class PolicyEngine:
    """
    Policy Engine for managing optimization policies and client roles.
    
    Responsibilities:
    - Assign roles to clients
    - Retrieve role-specific QoS weights
    - Evaluate client compliance against SLOs
    - Determine optimization objectives
    """
    
    def __init__(self, slo_catalog: SLOCatalog, default_role: str = "BE"):
        """
        Initialize Policy Engine.
        
        Args:
            slo_catalog: SLO catalog instance
            default_role: Default role for clients without assigned role
        """
        self.slo_catalog = slo_catalog
        self.default_role = default_role
        self.client_roles: Dict[int, str] = {}  # client_id -> role_id
        
        # Optimization objectives (can be configured)
        self.objectives: List[OptimizationObjective] = [
            OptimizationObjective(name="max_qoe", weight=1.0, minimize=False)
        ]
    
    def set_client_role(self, client_id: int, role_id: str) -> bool:
        """
        Assign a role to a client.
        
        Args:
            client_id: Client identifier
            role_id: Role identifier (must exist in catalog)
            
        Returns:
            True if role assigned successfully, False if role doesn't exist
        """
        # Validate role exists
        if self.slo_catalog.get_role(role_id) is None:
            return False
        
        self.client_roles[client_id] = role_id
        return True
    
    def get_client_role(self, client_id: int) -> str:
        """
        Get role assigned to a client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Role ID (returns default_role if not assigned)
        """
        return self.client_roles.get(client_id, self.default_role)
    
    def get_client_qos_weights(self, client_id: int) -> Dict[str, float]:
        """
        Get QoS weights for a client based on their assigned role.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dictionary with QoS weights: {'ws': ..., 'wt': ..., 'wr': ..., 'wl': ..., 'wa': ...}
        """
        role_id = self.get_client_role(client_id)
        return self.slo_catalog.get_qos_weights(role_id)
    
    def get_role_config(self, client_id: int) -> Optional[RoleConfig]:
        """
        Get complete role configuration for a client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            RoleConfig object or None
        """
        role_id = self.get_client_role(client_id)
        return self.slo_catalog.get_role(role_id)
    
    def evaluate_client_compliance(self, client_id: int, metrics: Dict[str, float]) -> List[str]:
        """
        Evaluate if client meets SLO and return triggered actions.
        
        Args:
            client_id: Client identifier
            metrics: Dictionary of metric values
                     e.g., {'RSSI_dBm': -70, 'Retry_pct': 5.0, 'CCA_busy': 65}
        
        Returns:
            List of action strings that should be triggered
            (e.g., ['IncreaseTxPower', 'Steer'])
        """
        role_id = self.get_client_role(client_id)
        return self.slo_catalog.evaluate_enforcement(role_id, metrics)
    
    def check_config_compliance(self, client_id: int, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Check if configuration complies with client's role regulatory constraints.
        
        Args:
            client_id: Client identifier
            config: Configuration to validate
            
        Returns:
            Tuple of (is_compliant, violations)
        """
        role_id = self.get_client_role(client_id)
        return self.slo_catalog.check_regulatory_compliance(role_id, config)
    
    def set_optimization_objective(self, objective: OptimizationObjective):
        """
        Set single optimization objective.
        
        Args:
            objective: OptimizationObjective to use
        """
        self.objectives = [objective]
    
    def set_multi_objective(self, objectives: List[OptimizationObjective]):
        """
        Set multiple optimization objectives with weights.
        
        Args:
            objectives: List of OptimizationObjective instances
        """
        self.objectives = objectives
    
    def get_objectives(self) -> List[OptimizationObjective]:
        """Get current optimization objectives."""
        return self.objectives
    
    def compute_weighted_qoe(self, client_id: int, qoe_components: Dict[str, float]) -> float:
        """
        Compute weighted QoE for a client using their role's weights.
        
        Args:
            client_id: Client identifier
            qoe_components: Dictionary with component scores
                           {'signal_quality': ..., 'throughput': ..., etc.}
        
        Returns:
            Weighted QoE score (0.0-1.0)
        """
        weights = self.get_client_qos_weights(client_id)
        
        # Map component names to weight keys
        component_map = {
            'signal_quality': 'ws',
            'throughput': 'wt',
            'reliability': 'wr',
            'latency': 'wl',
            'activity': 'wa'
        }
        
        qoe = 0.0
        for comp_name, weight_key in component_map.items():
            if comp_name in qoe_components and weight_key in weights:
                qoe += weights[weight_key] * qoe_components[comp_name]
        
        return qoe
    
    def list_client_roles(self) -> Dict[int, str]:
        """
        Get all client role assignments.
        
        Returns:
            Dictionary mapping client_id to role_id
        """
        return self.client_roles.copy()
    
    def reset_client_role(self, client_id: int):
        """
        Reset client to default role.
        
        Args:
            client_id: Client identifier
        """
        if client_id in self.client_roles:
            del self.client_roles[client_id]
    
    def get_role_summary(self) -> Dict[str, int]:
        """
        Get count of clients per role.
        
        Returns:
            Dictionary mapping role_id to count
        """
        summary = {}
        for role_id in self.client_roles.values():
            summary[role_id] = summary.get(role_id, 0) + 1
        return summary
    
    def print_status(self):
        """Print policy engine status."""
        print("\n" + "="*60)
        print("POLICY ENGINE STATUS")
        print("="*60)
        print(f"Default Role: {self.default_role}")
        print(f"Total Clients with Assigned Roles: {len(self.client_roles)}")
        
        summary = self.get_role_summary()
        if summary:
            print("\nRole Distribution:")
            for role_id, count in sorted(summary.items()):
                role = self.slo_catalog.get_role(role_id)
                display_name = role.display_name if role else role_id
                print(f"  {display_name} ({role_id}): {count} clients")
        
        print("\nOptimization Objectives:")
        for obj in self.objectives:
            direction = "minimize" if obj.minimize else "maximize"
            print(f"  {obj.name}: {direction} (weight={obj.weight:.2f})")
        print()
