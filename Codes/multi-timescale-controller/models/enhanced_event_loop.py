"""
Enhanced Event Loop Controller with Rollback and Audit.

High-priority event-driven optimization with:
- Automatic rollback on degradation
- HMAC-signed audit trail
- Emergency channel selection
- Post-action monitoring
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import time
import uuid

from models.event_models import (
    Event, EventType, Severity, ActionType, ExecutionStatus,
    ConfigurationChange, PostActionMetrics, EVENT_ACTION_MATRIX
)
from models.rollback_manager import RollbackManager
from models.audit_logger import AuditLogger
from models.channel_selector import EmergencyChannelSelector

from datatype import AccessPoint, Client, Interferer
from config_engine import ConfigEngine, NetworkConfig


class EnhancedEventLoop:
    """
    Enhanced Event Loop with rollback, audit, and monitoring.
    """
    
    def __init__(self,
                 config_engine: ConfigEngine,
                 audit_log_dir: str = "audit_logs",
                 monitoring_window_sec: int = 300,
                 cooldown_sec: int = 10):
        """
        Initialize Enhanced Event Loop.
        
        Args:
            config_engine: Configuration engine for applying changes
            audit_log_dir: Directory for audit logs
            monitoring_window_sec: Post-action monitoring window
            cooldown_sec: Cooldown between actions per AP
        """
        self.config_engine = config_engine
        self.monitoring_window_sec = monitoring_window_sec
        self.cooldown_sec = cooldown_sec
        
        # Core components
        self.rollback_manager = RollbackManager(monitoring_window_sec)
        self.audit_logger = AuditLogger(audit_log_dir)
        self.channel_selector = EmergencyChannelSelector()
        
        # Event queue (priority-based)
        self.event_queue: List[Event] = []
        
        # Active monitoring: token_id -> step counter
        self.active_monitoring: Dict[str, int] = {}
        
        # Statistics
        self.events_processed = 0
        self.actions_executed = 0
        self.rollbacks_triggered = 0
    
    def register_event(self, event: Event):
        """
        Add event to priority queue.
        
        Args:
            event: Event to process
        """
        self.event_queue.append(event)
        # Sort by severity (lower = higher priority)
        self.event_queue.sort(key=lambda e: e.severity)
    
    def execute(self, step: int, 
                access_points: List[AccessPoint],
                clients: List[Client],
                interferers: Optional[List[Interferer]] = None) -> Optional[NetworkConfig]:
        """
        Execute highest priority event if any exist.
        
        Args:
            step: Current simulation step
            access_points: List of access points
            clients: List of clients
            interferers: List of interferers
            
        Returns:
            NetworkConfig if action taken, None otherwise
        """
        # 1. Process post-action monitoring for active tokens
        self._check_monitoring(step, access_points, clients)
        
        # 2. Cleanup expired tokens
        self.rollback_manager.cleanup_expired_tokens()
        
        # 3. Process next event from queue
        if not self.event_queue:
            return None
        
        event = self.event_queue.pop(0)
        
        # Log event reception
        # print(f"\n[Event Loop] Processing event: {event.event_type} "
        #       f"(Severity: {event.severity}) for AP {event.ap_id}")
        
        # Execute event handler
        result = self._handle_event(event, step, access_points, clients, interferers)
        
        self.events_processed += 1
        
        return result
    
    def _handle_event(self, event: Event, step: int,
                     access_points: List[AccessPoint],
                     clients: List[Client],
                     interferers: Optional[List[Interferer]]) -> Optional[NetworkConfig]:
        """
        Handle a specific event.
        
        Args:
            event: Event to handle
            step: Current step
            access_points: List of APs
            clients: List of clients
            interferers: List of interferers
            
        Returns:
            NetworkConfig if action taken
        """
        # Get action matrix for event type
        action_config = EVENT_ACTION_MATRIX.get(event.event_type)
        if not action_config:
            # print(f"[Event Loop] No action configured for {event.event_type}")
            return None
        
        # Check confidence threshold
        threshold = action_config['confidence_threshold']
        if threshold and event.detection_confidence < threshold:
            print(f"[Event Loop] Confidence {event.detection_confidence:.2f} "
                  f"below threshold {threshold:.2f}, skipping")
            return None
        
        # Check AP cooldown
        ap_id_int = int(event.ap_id.split('_')[-1]) if '_' in event.ap_id else int(event.ap_id)
        if self.rollback_manager.check_ap_cooldown(event.ap_id, self.cooldown_sec):
            # print(f"[Event Loop] AP {event.ap_id} in cooldown, deferring")
            return None
        
        # Get action type
        action_type = action_config['primary_action']
        
        # Execute action based on type
        if event.event_type == EventType.DFS_RADAR:
            return self._handle_dfs_radar(event, step, access_points, interferers)
        elif event.event_type == EventType.NON_WIFI_BURST:
            return self._handle_non_wifi_burst(event, step, access_points, interferers)
        elif event.event_type == EventType.SPECTRUM_SAT:
            return self._handle_spectrum_saturation(event, step, access_points)
        elif event.event_type == EventType.DENSITY_SPIKE:
            return self._handle_density_spike(event, step, access_points, clients)
        else:
            print(f"[Event Loop] Handler not implemented for {event.event_type}")
            return None
    
    def _handle_dfs_radar(self, event: Event, step: int,
                         access_points: List[AccessPoint],
                         interferers: Optional[List[Interferer]]) -> Optional[NetworkConfig]:
        """Handle DFS radar detection"""
        ap_id_int = int(event.ap_id.split('_')[-1]) if '_' in event.ap_id else int(event.ap_id)
        
        # Find AP
        ap = next((a for a in access_points if a.id == ap_id_int), None)
        if not ap:
            return None
        
        current_channel = ap.channel
        
        # Select emergency channel
        new_channel = self.channel_selector.select_channel(
            current_channel=current_channel,
            radio="5g" if current_channel > 14 else "2g",
            access_points=access_points,
            interferers=interferers,
            excluded_channels=[current_channel]
        )
        
        # Create snapshot for rollback
        snapshot = {
            'channel': current_channel,
            'tx_power': ap.tx_power,
            'bandwidth': ap.bandwidth
        }
        
        # Create rollback token
        token = self.rollback_manager.create_token(
            event.ap_id, snapshot, event, ttl_seconds=3600
        )
        
        # Create configuration change
        config_changes = [
            ConfigurationChange(
                param="channel",
                old_value=current_channel,
                new_value=new_channel,
                radio="5g" if current_channel > 14 else "2g"
            )
        ]
        
        # Create audit record
        audit_record = self.audit_logger.create_event_action_record(
            event=event,
            ap_id=event.ap_id,
            step=step,
            action_type=ActionType.CHANNEL_CHANGE,
            config_changes=config_changes,
            rollback_token=token,
            reason=f"DFS radar detected on channel {current_channel}"
        )
        
        # Log audit record
        self.audit_logger.log_action(audit_record)
        
        # Execute channel change
        start_time = time.time()
        ap.channel = new_channel
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Update audit record
        self.audit_logger.update_execution_status(
            audit_record.audit_id,
            ExecutionStatus.SUCCESS,
            latency_ms,
            f"Channel changed to {new_channel}"
        )
        
        # Set cooldown
        self.rollback_manager.set_ap_cooldown(event.ap_id)
        
        # Start post-action monitoring
        baseline = self._collect_baseline_metrics(ap, access_points)
        self.rollback_manager.start_monitoring(token.token_id, baseline)
        self.active_monitoring[token.token_id] = step
        
        self.actions_executed += 1
        
        print(f"[Event Loop] DFS: Changed AP {ap.id} from ch{current_channel} to ch{new_channel}")
        
        # Return network config
        return self.config_engine.build_network_config(
            [self.config_engine.build_channel_config(ap.id, new_channel)],
            metadata={'event_type': 'dfs_radar', 'step': step}
        )
    
    def _handle_non_wifi_burst(self, event: Event, step: int,
                              access_points: List[AccessPoint],
                              interferers: Optional[List[Interferer]]) -> Optional[NetworkConfig]:
        """Handle non-WiFi interference burst"""
        ap_id_int = int(event.ap_id.split('_')[-1]) if '_' in event.ap_id else int(event.ap_id)
        
        ap = next((a for a in access_points if a.id == ap_id_int), None)
        if not ap:
            return None
        
        # Check duty cycle from metadata
        duty_cycle = event.metadata.get('duty_cycle_pct', 0)
        
        if duty_cycle < 70:
            print(f"[Event Loop] Interference duty cycle {duty_cycle}% < 70%, monitoring only")
            return None
        
        # Select new channel avoiding interferer
        interferer_channel = event.metadata.get('interferer_channel', ap.channel)
        
        new_channel = self.channel_selector.select_channel(
            current_channel=ap.channel,
            radio="2g",
            access_points=access_points,
            interferers=interferers,
            excluded_channels=[interferer_channel]
        )
        
        if new_channel == ap.channel:
            print(f"[Event Loop] No better channel available for AP {ap.id}")
            return None
        
        # Similar to DFS handling
        snapshot = {'channel': ap.channel, 'tx_power': ap.tx_power, 'bandwidth': ap.bandwidth}
        token = self.rollback_manager.create_token(event.ap_id, snapshot, event, ttl_seconds=300)
        
        config_changes = [ConfigurationChange("channel", ap.channel, new_channel, "2g")]
        audit_record = self.audit_logger.create_event_action_record(
            event, event.ap_id, step, ActionType.CHANNEL_CHANGE, config_changes, token,
            f"Non-WiFi interference burst detected (duty={duty_cycle}%)"
        )
        self.audit_logger.log_action(audit_record)
        
        # Execute
        old_channel = ap.channel
        ap.channel = new_channel
        
        self.audit_logger.update_execution_status(
            audit_record.audit_id, ExecutionStatus.SUCCESS, 0,
            f"Channel changed to {new_channel}"
        )
        
        self.rollback_manager.set_ap_cooldown(event.ap_id)
        baseline = self._collect_baseline_metrics(ap, access_points)
        self.rollback_manager.start_monitoring(token.token_id, baseline)
        self.active_monitoring[token.token_id] = step
        
        self.actions_executed += 1
        
        print(f"[Event Loop] Interference: Changed AP {ap.id} ch{old_channel}→ch{new_channel}")
        
        return self.config_engine.build_network_config(
            [self.config_engine.build_channel_config(ap.id, new_channel)],
            metadata={'event_type': 'non_wifi_burst', 'step': step}
        )
    
    def _handle_spectrum_saturation(self, event: Event, step: int,
                                   access_points: List[AccessPoint]) -> Optional[NetworkConfig]:
        """Handle spectrum saturation with OBSS-PD adjustment"""
        ap_id_int = int(event.ap_id.split('_')[-1]) if '_' in event.ap_id else int(event.ap_id)
        
        ap = next((a for a in access_points if a.id == ap_id_int), None)
        if not ap:
            return None
        
        # Increase OBSS-PD threshold (more aggressive spatial reuse)
        old_obss_pd = ap.obss_pd_threshold
        new_obss_pd = min(old_obss_pd + 8, -62.0)  # Increase by 8 dB, max -62
        
        if new_obss_pd == old_obss_pd:
            print(f"[Event Loop] OBSS-PD already at maximum for AP {ap.id}")
            return None
        
        snapshot = {'obss_pd_threshold': old_obss_pd}
        token = self.rollback_manager.create_token(event.ap_id, snapshot, event, ttl_seconds=600)
        
        config_changes = [ConfigurationChange("obss_pd_threshold", old_obss_pd, new_obss_pd)]
        audit_record = self.audit_logger.create_event_action_record(
            event, event.ap_id, step, ActionType.OBSS_PD_TUNE, config_changes, token,
            "Spectrum saturation detected, increasing OBSS-PD"
        )
        self.audit_logger.log_action(audit_record)
        
        # Execute
        ap.obss_pd_threshold = new_obss_pd
        
        self.audit_logger.update_execution_status(
            audit_record.audit_id, ExecutionStatus.SUCCESS, 0,
            f"OBSS-PD adjusted to {new_obss_pd} dBm"
        )
        
        self.rollback_manager.set_ap_cooldown(event.ap_id)
        baseline = self._collect_baseline_metrics(ap, access_points)
        self.rollback_manager.start_monitoring(token.token_id, baseline)
        self.active_monitoring[token.token_id] = step
        
        self.actions_executed += 1
        
        print(f"[Event Loop] OBSS-PD: AP {ap.id} {old_obss_pd}→{new_obss_pd} dBm")
        
        return None  # OBSS-PD change doesn't need NetworkConfig
    
    def _handle_density_spike(self, event: Event, step: int,
                            access_points: List[AccessPoint],
                            clients: List[Client]) -> Optional[NetworkConfig]:
        """Handle client density spike"""
        # For now, just log it
        print(f"[Event Loop] Density spike detected for AP {event.ap_id}")
        return None
    
    def _check_monitoring(self, step: int,
                         access_points: List[AccessPoint],
                         clients: List[Client]):
        """Check post-action monitoring and trigger rollback if needed"""
        tokens_to_remove = []
        
        for token_id, monitoring_start_step in self.active_monitoring.items():
            # Check if monitoring window expired
            if not self.rollback_manager.is_monitoring_active(token_id):
                tokens_to_remove.append(token_id)
                continue
            
            # Collect current metrics
            token = self.rollback_manager.get_token(token_id)
            if not token:
                tokens_to_remove.append(token_id)
                continue
            
            ap_id_int = int(token.ap_id.split('_')[-1]) if '_' in token.ap_id else int(token.ap_id)
            ap = next((a for a in access_points if a.id == ap_id_int), None)
            if not ap:
                tokens_to_remove.append(token_id)
                continue
            
            current_metrics = self._collect_current_metrics(ap, access_points)
            
            # Check for rollback condition
            if self.rollback_manager.check_auto_rollback(token_id, current_metrics):
                print(f"\n[Event Loop] AUTO-ROLLBACK triggered for token {token_id}")
                
                # Execute rollback
                result = self.rollback_manager.execute_rollback(token_id, "auto")
                
                if result['success']:
                    # Restore configuration
                    snapshot = result['snapshot']
                    if 'channel' in snapshot:
                        ap.channel = snapshot['channel']
                    if 'obss_pd_threshold' in snapshot:
                        ap.obss_pd_threshold = snapshot['obss_pd_threshold']
                    
                    # Find audit record and mark as rolled back
                    for record in self.audit_logger.recent_records:
                        if record.rollback_token == token_id:
                            self.audit_logger.mark_rollback(
                                record.audit_id,
                                self.rollback_manager.monitoring[token_id].get('rollback_reason', 'Unknown')
                            )
                            break
                    
                    self.rollbacks_triggered += 1
                    print(f"[Event Loop] Rollback completed for AP {ap.id}")
                
                tokens_to_remove.append(token_id)
        
        # Remove completed monitoring
        for token_id in tokens_to_remove:
            self.active_monitoring.pop(token_id, None)
    
    def _collect_baseline_metrics(self, ap: AccessPoint,
                                  access_points: List[AccessPoint]) -> PostActionMetrics:
        """Collect baseline metrics before action"""
        return PostActionMetrics(
            per_p95=0.0,  # Would come from metrics manager
            retry_rate_p95=ap.p95_retry_rate,
            client_disconnection_rate=0.0,
            throughput_degradation_pct=0.0,
            new_critical_events=0
        )
    
    def _collect_current_metrics(self, ap: AccessPoint,
                                access_points: List[AccessPoint]) -> PostActionMetrics:
        """Collect current metrics during monitoring"""
        return PostActionMetrics(
            per_p95=0.0,
            retry_rate_p95=ap.p95_retry_rate,
            client_disconnection_rate=0.0,
            throughput_degradation_pct=0.0,
            new_critical_events=0
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event loop statistics"""
        return {
            'events_processed': self.events_processed,
            'actions_executed': self.actions_executed,
            'rollbacks_triggered': self.rollbacks_triggered,
            'pending_events': len(self.event_queue),
            'active_monitoring': len(self.active_monitoring),
            'rollback_stats': self.rollback_manager.get_statistics(),
            'audit_stats': self.audit_logger.get_statistics()
        }
    
    def print_status(self):
        """Print event loop status"""
        print("\n" + "="*60)
        print("ENHANCED EVENT LOOP STATUS")
        print("="*60)
        stats = self.get_statistics()
        print(f"Events Processed: {stats['events_processed']}")
        print(f"Actions Executed: {stats['actions_executed']}")
        print(f"Rollbacks Triggered: {stats['rollbacks_triggered']}")
        print(f"Pending Events: {stats['pending_events']}")
        print(f"Active Monitoring: {stats['active_monitoring']}")
        print()
        
        self.rollback_manager.print_status()
        self.audit_logger.print_status()
