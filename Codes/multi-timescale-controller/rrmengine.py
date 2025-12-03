"""
RRM Engine - Radio Resource Management Engine.

Multi-timescale control for WiFi network optimization:
- PolicyEngine: Role-based policy management
- ConfigEngine: AP configuration management
- SlowLoopController: Long-term optimization (channels, power)
- FastLoopController: Real-time optimization (client steering, load balancing)
"""

from typing import Optional, List, Tuple, Dict
from datatype import AccessPoint, Client
from slo_catalog import SLOCatalog
from policy_engine import PolicyEngine
from config_engine import ConfigEngine
from sensing import SensingAPI
from clientview import ClientViewAPI


class RRMEngine:
    """
    Radio Resource Management Engine.
    
    Orchestrates multi-timescale network optimization using:
    - SLO Catalog for role-based policies
    - PolicyEngine for client role management
    - ConfigEngine for AP configuration
    - SensingAPI for interferer detection
    - ClientViewAPI for QoE monitoring
    """
    
    def __init__(self,
                 access_points: List[AccessPoint],
                 clients: List[Client],
                 interferers: List = None,
                 prop_model = None,
                 slo_catalog_path: str = "slo_catalog.yml",
                 default_role: str = "BE",
                 slow_loop_period: int = 100):
        """
        Initialize RRM Engine.
        
        Args:
            access_points: List of access points
            clients: List of clients
            interferers: List of interferers (optional)
            prop_model: Propagation model (optional)
            slo_catalog_path: Path to SLO catalog YAML file
            default_role: Default role for clients
            slow_loop_period: Steps between slow loop executions
        """
        self.rrm_enabled = True
        
        # Stored Entities
        self.aps: Dict[int, AccessPoint] = {ap.id: ap for ap in access_points}
        self.stas: Dict[int, Client] = {client.id: client for client in clients}
        
        # Load SLO Catalog
        self.slo_catalog = SLOCatalog(slo_catalog_path)
        
        # Initialize Engines
        self.policy_engine = PolicyEngine(self.slo_catalog, default_role=default_role)
        self.config_engine = ConfigEngine(access_points)
        
        # Initialize APIs
        self.sensing_api: Optional[SensingAPI] = None
        if interferers and prop_model:
            self.sensing_api = SensingAPI(access_points, interferers, prop_model)
        
        self.client_view_api = ClientViewAPI(access_points, clients)
        
        # Controllers
        from slow_loop_controller import SlowLoopController
        from fast_loop_controller import FastLoopController
        
        self.slow_loop_engine = SlowLoopController(
            self.policy_engine,
            self.config_engine,
            self.sensing_api,
            self.client_view_api,
            period=slow_loop_period
        )
        
        self.fast_loop_engine = FastLoopController(
            self.policy_engine,
            self.config_engine,
            self.client_view_api,
            clients
        )
        
        # Configuration
        self.slow_loop_period = slow_loop_period
        self.current_step = 0
    
    def execute(self, step: int) -> Dict[str, any]:
        """
        Main RRM execution loop.
        
        Args:
            step: Current simulation step
            
        Returns:
            Dictionary with execution results
        """
        if not self.rrm_enabled:
            return {}
        
        self.current_step = step
        results = {}
        
        # Update sensing data (if available)
        if self.sensing_api:
            sensing_results = self.sensing_api.compute_sensing_results()
            results['sensing'] = sensing_results
        
        # Update client view (QoE)
        qoe_views = self.client_view_api.compute_all_views()
        results['qoe'] = qoe_views
        
        # Fast loop (every step)
        steering_actions = self.fast_loop_engine.execute()
        if steering_actions:
            results['steering'] = [
                {'client_id': cid, 'old_ap': old, 'new_ap': new}
                for cid, old, new in steering_actions
            ]
        
        # Slow loop (periodic)
        if self.slow_loop_engine.should_execute(step):
            config = self.slow_loop_engine.execute(step)
            if config:
                success = self.config_engine.apply_config(config)
                if success:
                    results['config_update'] = config.to_dict()
                    results['optimization_type'] = config.metadata.get('optimization', 'unknown')
        
        return results
    
    def set_client_role(self, client_id: int, role_id: str) -> bool:
        """
        Assign a role to a client.
        
        Args:
            client_id: Client identifier
            role_id: Role identifier
            
        Returns:
            True if successful
        """
        return self.policy_engine.set_client_role(client_id, role_id)
    
    def get_client_role(self, client_id: int) -> str:
        """Get role assigned to a client."""
        return self.policy_engine.get_client_role(client_id)
    
    def apply_config(self, config) -> bool:
        """Apply configuration to network."""
        return self.config_engine.apply_config(config)
    
    def rollback_config(self, steps: int = 1) -> bool:
        """Rollback configuration."""
        return self.config_engine.rollback(steps)
    
    def print_status(self):
        """Print RRM Engine status."""
        print("\n" + "="*60)
        print("RRM ENGINE STATUS")
        print("="*60)
        print(f"Enabled: {self.rrm_enabled}")
        print(f"Current Step: {self.current_step}")
        print(f"Slow Loop Period: {self.slow_loop_period}")
        print(f"Access Points: {len(self.aps)}")
        print(f"Clients: {len(self.stas)}")
        
        # Print sub-component status
        self.policy_engine.print_status()
        self.config_engine.print_status()
        if self.slow_loop_engine:
            self.slow_loop_engine.print_status()
        if self.fast_loop_engine:
            self.fast_loop_engine.print_status()

