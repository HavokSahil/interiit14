"""
Enhanced RRM Engine with new Event Loop integration.

Combines:
- Enhanced Event Loop (with rollback and audit)
- Existing Fast Loop and Slow Loop controllers
- Event injection for testing
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import time

from datatype import AccessPoint, Client, Interferer
from slo_catalog import SLOCatalog
from policy_engine import PolicyEngine
from config_engine import ConfigEngine
from sensing import SensingAPI
from clientview import ClientViewAPI

# Import enhanced event loop
from models import (
    EnhancedEventLoop, Event, EventType, Severity,
    SensingSource, PostActionMetrics
)


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


class EnhancedRRMEngine:
    """
    Enhanced RRM Engine with integrated Event Loop rollback and audit.
    
    Features:
    - Enhanced Event Loop with automatic rollback
    - Audit trail with HMAC signatures
    - Event injection for testing (DFS, interference)
    - Integration with existing Fast/Slow loops
    """
    
    def __init__(self,
                 access_points: List[AccessPoint],
                 clients: List[Client],
                 interferers: Optional[List[Interferer]] = None,
                 prop_model=None,
                 slo_catalog_path: str = "slo_catalog.yml",
                 default_role: str = "BE",
                 slow_loop_period: int = 100,
                 fast_loop_period: int = 60,  # 10 minutes at 360 steps/hour
                 cooldown_steps: int = 50,
                 audit_log_dir: str = "audit_logs"):
        """
        Initialize Enhanced RRM Engine.
        
        Args:
            access_points: List of access points
            clients: List of clients
            interferers: List of interferers (optional)
            prop_model: Propagation model (optional)
            slo_catalog_path: Path to SLO catalog YAML file
            default_role: Default role for clients
            slow_loop_period: Steps between slow loop executions
            fast_loop_period: Steps between fast loop executions (default: 60 = 10 min)
            cooldown_steps: Steps between configuration changes
            audit_log_dir: Directory for audit logs
        """
        self.rrm_enabled = True
        
        # Stored Entities
        self.aps: Dict[int, AccessPoint] = {ap.id: ap for ap in access_points}
        self.stas: Dict[int, Client] = {client.id: client for client in clients}
        self.interferers = interferers or []
        
        # Load SLO Catalog
        self.slo_catalog = SLOCatalog(slo_catalog_path)
        
        # Initialize Engines
        self.policy_engine = PolicyEngine(self.slo_catalog, default_role=default_role)
        self.config_engine = ConfigEngine(access_points)
        
        # Initialize APIs
        self.sensing_api: Optional[SensingAPI] = None
        if interferers and prop_model:
            self.sensing_api = SensingAPI(access_points, interferers, prop_model)
        
        # Initialize Interference Graph Builder for Fast Loop
        if prop_model:
            from metrics import InterferenceGraphBuilder
            self.graph_builder = InterferenceGraphBuilder(
                propagation_model=prop_model,
                interference_threshold_dbm=-75.0
            )
        else:
            self.graph_builder = None
        
        self.client_view_api = ClientViewAPI(access_points, clients)
        
        # ========== NEW: ENHANCED EVENT LOOP ==========
        self.event_loop = EnhancedEventLoop(
            config_engine=self.config_engine,
            audit_log_dir=audit_log_dir,
            monitoring_window_sec=300,  # 5 minutes
            cooldown_sec=10
        )
        
        # Controllers (Fast/Slow loops - optional integration)
        try:
            from slow_loop_controller import SlowLoopController
            self.slow_loop_engine = SlowLoopController(
                self.policy_engine,
                self.config_engine,
                self.sensing_api,
                self.client_view_api,
                period=slow_loop_period
            )
        except ImportError:
            self.slow_loop_engine = None
        
        # Fast Loop Controller (interference-based optimization)
        try:
            from fast_loop_controller import FastLoopController
            self.fast_loop_engine = FastLoopController(
                config_engine=self.config_engine,
                policy_engine=self.policy_engine,
                access_points=access_points,
                config_path="fast_loop_config.yml"
            )
            print("[RRM] Fast Loop Controller initialized (interference-based)")
        except ImportError as e:
            self.fast_loop_engine = None
            print(f"[RRM] Fast Loop Controller not available: {e}")
        
        # State Management
        self.state = RRMState()
        
        # Configuration
        self.cooldown_steps = cooldown_steps
        self.last_slow_loop_step = 0
        self.slow_loop_period = slow_loop_period
        self.last_fast_loop_step = 0
        self.fast_loop_period = fast_loop_period
        self.current_step = 0
        
        # Event injection tracking
        self.injected_events = []
    
    def execute(self, step: int) -> Dict[str, Any]:
        """
        Main RRM execution loop with Enhanced Event Loop priority.
        
        Priority Order:
        1. Lock Check → Skip all if locked
        2. Enhanced Event Loop → Highest priority (DFS, interference, etc.)
        3. Cooldown Check → Blocks slow loop only
        4. Slow Loop → Periodic long-term (if available)
        5. Fast Loop → Every step real-time (if available)
        
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
        
        # ========== AUTO-DETECT EVENTS ==========
        self._auto_detect_events(step, sensing_results, qoe_views)
        
        # ========== PRIORITY 2: ENHANCED EVENT LOOP ==========
        access_points_list = list(self.aps.values())
        clients_list = list(self.stas.values())
        
        event_config = self.event_loop.execute(
            step=step,
            access_points=access_points_list,
            clients=clients_list,
            interferers=self.interferers
        )
        
        if event_config:
            # Event action was taken
            self.state.last_config_change_step = step
            self.state.total_config_changes += 1
            self.state.total_events_handled += 1
            results['event_action'] = event_config.to_dict()
            results['event_metadata'] = event_config.metadata
            results['event_loop_stats'] = self.event_loop.get_statistics()
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
        if self.slow_loop_engine and not in_cooldown:
            if self.slow_loop_engine.should_execute(step):
                config = self.slow_loop_engine.execute(step)
                if config:
                    success = self.config_engine.apply_config(config)
                    if success:
                        self.state.last_config_change_step = step
                        self.state.total_config_changes += 1
                        results['slow_loop'] = config.to_dict()
                        results['optimization_type'] = config.metadata.get('optimization')
        
        
        # ========== PRIORITY 5: FAST LOOP (Every 10 minutes) ==========
        if self.fast_loop_engine and (step - self.last_fast_loop_step >= self.fast_loop_period):
            try:
                # Build interference graph
                if self.graph_builder:
                    interference_graph = self.graph_builder.build_graph(list(self.aps.values()))
                    
                    # Execute Fast Loop with graph and current step
                    actions = self.fast_loop_engine.execute(interference_graph, current_step=step)
                    
                    if actions:
                        # Track successful actions
                        for action in actions:
                            if action.get('success'):
                                self.state.total_config_changes += 1
                        
                        results['fast_loop_actions'] = actions
                        
                        # Get statistics
                        if hasattr(self.fast_loop_engine, 'get_statistics'):
                            results['fast_loop_stats'] = self.fast_loop_engine.get_statistics()
                
                self.last_fast_loop_step = step
                
            except Exception as e:
                print(f"[RRM] Fast Loop error: {e}")
                results['fast_loop_error'] = str(e)
        
        return results
    
    def _auto_detect_events(self, step: int, sensing_results, qoe_views):
        """Auto-detect events from sensing and QoE data"""
        if not sensing_results or not qoe_views:
            return
        
        # Detect interference bursts
        for ap_id, sensing in sensing_results.items():
            if sensing.confidence >= 0.80:  # High confidence interferer
                # Create interference burst event
                event = Event(
                    event_id=f"auto_intf_{ap_id}_{step}",
                    event_type=EventType.NON_WIFI_BURST,
                    severity=Severity.HIGH,
                    ap_id=f"ap_{ap_id}",
                    radio="2g",  # Assume 2.4 GHz for now
                    timestamp_utc=datetime.utcnow(),
                    detection_confidence=sensing.confidence,
                    metadata={
                        'interferer_type': sensing.major_interferer_type,
                        'interferer_channel': 6,  # Would come from sensing
                        'duty_cycle_pct': 80,  # Estimate
                        'center_freq': sensing.center_frequency
                    },
                    sensing_source=SensingSource.ADDITIONAL_RADIO
                )
                self.event_loop.register_event(event)
        
        # Detect critical QoE degradation
        for ap_id, view in qoe_views.items():
            if view.num_clients > 0 and view.avg_qoe < 0.3:
                event = Event(
                    event_id=f"auto_qoe_{ap_id}_{step}",
                    event_type=EventType.EMERGENCY_QOE,
                    severity=Severity.HIGH,
                    ap_id=f"ap_{ap_id}",
                    radio="2g",
                    timestamp_utc=datetime.utcnow(),
                    detection_confidence=1.0,
                    metadata={
                        'avg_qoe': view.avg_qoe,
                        'min_qoe': view.min_qoe,
                        'num_clients': view.num_clients
                    }
                )
                self.event_loop.register_event(event)
    
    def inject_dfs_event(self, ap_id: int, channel: int):
        """
        Inject a DFS radar detection event for testing.
        
        Args:
            ap_id: Access Point ID
            channel: Channel where radar was detected
        """
        event = Event(
            event_id=f"dfs_{ap_id}_{self.current_step}",
            event_type=EventType.DFS_RADAR,
            severity=Severity.CRITICAL,
            ap_id=f"ap_{ap_id}",
            radio="5g",
            timestamp_utc=datetime.utcnow(),
            detection_confidence=1.0,
            metadata={
                'channel': channel,
                'pulse_width_us': 1.0,
                'repetition_interval_us': 1000
            },
            sensing_source=SensingSource.SERVING_RADIO
        )
        
        self.event_loop.register_event(event)
        self.injected_events.append(event)
        print(f"[RRM] DFS event injected: AP {ap_id}, Channel {channel}")
    
    def inject_interference_event(self, ap_id: int, interferer_type: str = "Microwave"):
        """
        Inject an interference burst event for testing.
        
        Args:
            ap_id: Access Point ID
            interferer_type: Type of interferer
        """
        ap = self.aps.get(ap_id)
        if not ap:
            return
        
        event = Event(
            event_id=f"intf_{ap_id}_{self.current_step}",
            event_type=EventType.NON_WIFI_BURST,
            severity=Severity.HIGH,
            ap_id=f"ap_{ap_id}",
            radio="2g",
            timestamp_utc=datetime.utcnow(),
            detection_confidence=0.85,
            metadata={
                'interferer_type': interferer_type,
                'interferer_channel': ap.channel,
                'duty_cycle_pct': 80,
                'center_freq': 2.437
            },
            sensing_source=SensingSource.ADDITIONAL_RADIO
        )
        
        self.event_loop.register_event(event)
        self.injected_events.append(event)
        print(f"[RRM] Interference event injected: AP {ap_id}, Type {interferer_type}")
    
    def inject_spectrum_saturation_event(self, ap_id: int, cca_busy_pct: float = 96):
        """
        Inject a spectrum saturation event for testing.
        
        Args:
            ap_id: Access Point ID
            cca_busy_pct: CCA busy percentage
        """
        event = Event(
            event_id=f"spectrum_{ap_id}_{self.current_step}",
            event_type=EventType.SPECTRUM_SAT,
            severity=Severity.HIGH,
            ap_id=f"ap_{ap_id}",
            radio="2g",
            timestamp_utc=datetime.utcnow(),
            detection_confidence=1.0,
            metadata={'cca_busy_pct': cca_busy_pct}
        )
        
        self.event_loop.register_event(event)
        self.injected_events.append(event)
        print(f"[RRM] Spectrum saturation event injected: AP {ap_id}, CCA {cca_busy_pct}%")
    
    def is_in_cooldown(self, current_step: int) -> bool:
        """Check if system is in cooldown period"""
        steps_since_change = current_step - self.state.last_config_change_step
        return steps_since_change < self.cooldown_steps
    
    def lock(self, reason: str = "Manual lock"):
        """Lock the RRM system"""
        self.state.locked = True
        self.state.lock_reason = reason
        self.state.lock_timestamp = time.time()
    
    def unlock(self):
        """Unlock the RRM system"""
        self.state.locked = False
        self.state.lock_reason = None
        self.state.lock_timestamp = None
    
    def set_client_role(self, client_id: int, role_id: str) -> bool:
        """Assign a role to a client"""
        return self.policy_engine.set_client_role(client_id, role_id)
    
    def get_client_role(self, client_id: int) -> str:
        """Get role assigned to a client"""
        return self.policy_engine.get_client_role(client_id)
    
    def apply_config(self, config) -> bool:
        """Apply configuration to network"""
        return self.config_engine.apply_config(config)
    
    def rollback_config(self, steps: int = 1) -> bool:
        """Rollback configuration"""
        return self.config_engine.rollback(steps)
    
    def print_status(self):
        """Print RRM Engine status"""
        print("\n" + "="*70)
        print("ENHANCED RRM ENGINE STATUS")
        print("="*70)
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
        print(f"  Injected Events: {len(self.injected_events)}")
        
        # Print Event Loop status
        self.event_loop.print_status()
        
        # Print other components
        print("\n" + "-"*70)
        self.policy_engine.print_status()
        self.config_engine.print_status()
        
        if self.slow_loop_engine:
            self.slow_loop_engine.print_status()
        
        # Print refactored fast loop stats if available
        if self.fast_loop_engine and hasattr(self.fast_loop_engine, 'print_status'):
            self.fast_loop_engine.print_status()
        elif self.fast_loop_engine and hasattr(self.fast_loop_engine, 'get_statistics'):
            # Refactored controller
            stats = self.fast_loop_engine.get_statistics()
            print("\n" + "="*60)
            print("REFACTORED FAST LOOP STATUS")
            print("="*60)
            print(f"Actions Executed: {stats['actions_executed']}")
            print(f"Actions Succeeded: {stats['actions_succeeded']}")
            print(f"Actions Rolled Back: {stats['actions_rolled_back']}")
            print(f"Rollback Rate: {stats['rollback_rate']:.1%}")
            print(f"Active Penalties: {stats['active_penalties']}")
            print()

