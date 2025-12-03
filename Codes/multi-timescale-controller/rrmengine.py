"""
RRM Engine - Radio Resource Management Engine.

Multi-timescale control for WiFi network optimization:
- PolicyEngine: Role-based policy management
- ConfigEngine: AP configuration management
- SlowLoopController: Long-term optimization (channels, power)
- FastLoopController: Real-time optimization (client steering, load balancing)
"""

from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import time

from datatype import AccessPoint, Client
from slo_catalog import SLOCatalog
from policy_engine import PolicyEngine
from config_engine import ConfigEngine
from sensing import SensingAPI
from clientview import ClientViewAPI


@dataclass
class RRMState:
    """Tracks RRM engine operational state"""
    locked: bool = False
    lock_reason: Optional[str] = None
    lock_timestamp: Optional[float] = None
    
    last_config_change_step: int = -1000  # Allow immediate first change
    cooldown_active: bool = False
    cooldown_remaining_steps: int = 0
    
    total_config_changes: int = 0
    total_events_handled: int = 0
    total_steering_actions: int = 0


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
                 slow_loop_period: int = 100,
                 cooldown_steps: int = 50):
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
            cooldown_steps: Steps between configuration changes (cooldown period)
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
        
        # NEW: Events Loop Controller
        from events_loop_controller import EventsLoopController
        from event_detector import EventDetector
        
        self.events_loop_engine = EventsLoopController(
            self.policy_engine,
            self.config_engine,
            self.sensing_api,
            self.client_view_api
        )
        
        # NEW: Event Detector
        self.event_detector = EventDetector()
        
        # NEW: RRM State Management
        self.state = RRMState()
        
        # Configuration
        self.cooldown_steps = cooldown_steps
        self.slow_loop_period = slow_loop_period
        self.current_step = 0
    
    def execute(self, step: int) -> Dict[str, any]:
        """
        Main RRM execution loop with priority-based control.
        
        Priority Order:
        1. Lock Check → Skip all if locked
        2. Events Loop → Highest priority
        3. Cooldown Check → Blocks slow loop only
        4. Slow Loop → Periodic long-term
        5. Fast Loop → Every step real-time
        
        Args:
            step: Current simulation step
            
        Returns:
            Dictionary with execution results
        """
        if not self.rrm_enabled:
            return {}
        
        self.current_step = step
        results = {}
        
        # ========== PRIORITY 1: CHECK LOCK ==========
        if self.state.locked:
            results['locked'] = True
            results['lock_reason'] = self.state.lock_reason
            # Continue monitoring only
            if self.sensing_api:
                results['sensing'] = self.sensing_api.compute_sensing_results()
            results['qoe'] = self.client_view_api.compute_all_views()
            return results
        
        # ========== UPDATE MONITORING DATA ==========
        sensing_results = None
        if self.sensing_api:
            sensing_results = self.sensing_api.compute_sensing_results()
            results['sensing'] = sensing_results
        
        qoe_views = self.client_view_api.compute_all_views()
        results['qoe'] = qoe_views
        
        # ========== DETECT EVENTS ==========
        if sensing_results:
            detected_events = self.event_detector.detect_events(
                sensing_results, qoe_views, step
            )
            for event in detected_events:
                self.events_loop_engine.register_event(event)
        
        # ========== PRIORITY 2: EVENTS LOOP ==========
        event_config = self.events_loop_engine.execute(step)
        if event_config:
            success = self.config_engine.apply_config(event_config)
            if success:
                self.state.last_config_change_step = step
                self.state.total_config_changes += 1
                self.state.total_events_handled += 1
                results['event_action'] = event_config.to_dict()
                results['event_metadata'] = event_config.metadata
                # Event handled, skip other loops this step
                return results
        
        # ========== PRIORITY 3: CHECK COOLDOWN ==========
        in_cooldown = self.is_in_cooldown(step)
        self.state.cooldown_active = in_cooldown
        if in_cooldown:
            self.state.cooldown_remaining_steps = (
                self.cooldown_steps - (step - self.state.last_config_change_step)
            )
            results['in_cooldown'] = True
            results['cooldown_remaining'] = self.state.cooldown_remaining_steps
        
        # ========== PRIORITY 4: SLOW LOOP ==========
        if not in_cooldown and self.slow_loop_engine.should_execute(step):
            config = self.slow_loop_engine.execute(step)
            if config:
                success = self.config_engine.apply_config(config)
                if success:
                    self.state.last_config_change_step = step
                    self.state.total_config_changes += 1
                    results['slow_loop'] = config.to_dict()
                    results['optimization_type'] = config.metadata.get('optimization')
        
        # ========== PRIORITY 5: FAST LOOP ==========
        steering_actions = self.fast_loop_engine.execute()
        if steering_actions:
            self.state.total_steering_actions += len(steering_actions)
            results['steering'] = [
                {'client_id': cid, 'old_ap': old, 'new_ap': new}
                for cid, old, new in steering_actions
            ]
        
        return results
    
    def lock(self, reason: str = "Manual lock"):
        """
        Lock the RRM system (prevents all optimizations).
        
        Args:
            reason: Reason for locking
        """
        self.state.locked = True
        self.state.lock_reason = reason
        self.state.lock_timestamp = time.time()
    
    def unlock(self):
        """Unlock the RRM system."""
        self.state.locked = False
        self.state.lock_reason = None
        self.state.lock_timestamp = None
    
    def is_in_cooldown(self, current_step: int) -> bool:
        """
        Check if system is in cooldown period.
        
        Args:
            current_step: Current simulation step
            
        Returns:
            True if in cooldown
        """
        steps_since_change = current_step - self.state.last_config_change_step
        return steps_since_change < self.cooldown_steps
    
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
        print(f"Locked: {self.state.locked}")
        if self.state.locked:
            print(f"  Reason: {self.state.lock_reason}")
        print(f"Cooldown: {self.cooldown_steps} steps")
        if self.state.cooldown_active:
            print(f"  IN COOLDOWN ({self.state.cooldown_remaining_steps} steps remaining)")
        
        print(f"\nStatistics:")
        print(f"  Config Changes: {self.state.total_config_changes}")
        print(f"  Events Handled: {self.state.total_events_handled}")
        print(f"  Steering Actions: {self.state.total_steering_actions}")
        
        # Print sub-component status
        self.policy_engine.print_status()
        self.config_engine.print_status()
        if self.slow_loop_engine:
            self.slow_loop_engine.print_status()
        if self.fast_loop_engine:
            self.fast_loop_engine.print_status()
        if self.events_loop_engine:
            self.events_loop_engine.print_status()

