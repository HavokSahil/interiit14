"""
Events Loop Controller for RRMEngine.

High-priority event-driven optimization:
- DFS radar detection
- Interference bursts
- Scheduled events (exam hours, etc.)
- Emergency situations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import IntEnum
import heapq
import time
from datetime import datetime

from datatype import AccessPoint, Client
from policy_engine import PolicyEngine
from config_engine import ConfigEngine, NetworkConfig
from sensing import SensingAPI
from clientview import ClientViewAPI


class EventPriority(IntEnum):
    """Event priority levels (lower number = higher priority)"""
    CRITICAL = 1      # DFS radar, AP failure
    HIGH = 2          # Interference burst, critical QoE
    MEDIUM = 3        # Scheduled events
    LOW = 4           # Non-urgent events


@dataclass
class RRMEvent:
    """Represents a network event requiring RRM action"""
    event_id: str                    # Unique identifier
    event_type: str                  # "dfs_radar", "interference_burst", etc.
    priority: EventPriority
    timestamp: float                 # When event was detected
    metadata: Dict[str, Any]         # Event-specific data
    action_type: str                 # "channel_switch", "power_adjust", etc.
    target_ap_ids: List[int] = field(default_factory=list)  # Affected APs
    handled: bool = False            # Whether event was processed
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority < other.priority


@dataclass
class ScheduledEvent:
    """Scheduled network events (exam hours, etc.)"""
    name: str
    start_hour: int              # 0-23
    start_minute: int            # 0-59
    end_hour: int
    end_minute: int
    days_of_week: List[int]      # 0=Monday, 6=Sunday
    action_type: str
    config_changes: Dict[str, Any]
    active: bool = True


class EventsLoopController:
    """
    High-priority event-driven controller for RRMEngine.
    
    Handles:
    - DFS radar detection
    - Interference bursts
    - Scheduled events (exam hours, etc.)
    - Emergency situations
    """
    
    def __init__(self,
                 policy_engine: PolicyEngine,
                 config_engine: ConfigEngine,
                 sensing_api: Optional[SensingAPI],
                 client_view_api: ClientViewAPI):
        """
        Initialize Events Loop Controller.
        
        Args:
            policy_engine: Policy engine for role-based decisions
            config_engine: Config engine for applying changes
            sensing_api: Sensing API for interference detection
            client_view_api: Client view API for QoE monitoring
        """
        self.policy_engine = policy_engine
        self.config_engine = config_engine
        self.sensing_api = sensing_api
        self.client_view_api = client_view_api
        
        # Event queue (priority-based)
        self.event_queue: List[Tuple[EventPriority, float, RRMEvent]] = []
        
        # Event handlers registry
        self.event_handlers: Dict[str, Callable] = {
            'dfs_radar': self._handle_dfs_radar,
            'interference_burst': self._handle_interference_burst,
            'scheduled_event': self._handle_scheduled_event,
            'emergency_qoe': self._handle_emergency_qoe,
            'ap_failure': self._handle_ap_failure,
        }
        
        # Scheduled events
        self.scheduled_events: List[ScheduledEvent] = []
        
        # Statistics
        self.events_processed = 0
        self.events_by_type: Dict[str, int] = {}
    
    def register_event(self, event: RRMEvent) -> None:
        """
        Add event to priority queue.
        
        Args:
            event: RRMEvent to process
        """
        # Queue entry: (priority, timestamp, event)
        # Priority is primary sort, timestamp breaks ties (FIFO for same priority)
        heapq.heappush(
            self.event_queue,
            (event.priority, event.timestamp, event)
        )
    
    def execute(self, step: int) -> Optional[NetworkConfig]:
        """
        Execute highest priority event if any exist.
        
        Args:
            step: Current simulation step
            
        Returns:
            NetworkConfig if event handled, None otherwise
        """
        # Check scheduled events
        self._check_scheduled_events(step)
        
        # Process next event from queue
        if not self.event_queue:
            return None
        
        priority, timestamp, event = heapq.heappop(self.event_queue)
        
        # Get handler for event type
        handler = self.event_handlers.get(event.event_type)
        if not handler:
            print(f"Warning: No handler for event type '{event.event_type}'")
            return None
        
        # Execute handler
        try:
            config = handler(event, step)
            event.handled = True
            
            # Update statistics
            self.events_processed += 1
            self.events_by_type[event.event_type] = \
                self.events_by_type.get(event.event_type, 0) + 1
            
            return config
        except Exception as e:
            print(f"Error handling event {event.event_id}: {e}")
            return None
    
    def _handle_dfs_radar(self, event: RRMEvent, step: int) -> NetworkConfig:
        """
        Handle DFS radar detection - IMMEDIATE channel switch required.
        
        Args:
            event: DFS radar event
            step: Current step
            
        Returns:
            NetworkConfig with emergency channel change
        """
        affected_ap_id = event.metadata['ap_id']
        detected_channel = event.metadata['channel']
        
        # Non-DFS safe channels (2.4 GHz)
        safe_channels = [1, 6, 11]
        
        # Find best safe channel based on interference
        best_channel = self._find_best_safe_channel(
            affected_ap_id,
            safe_channels
        )
        
        # Build emergency config
        ap_config = self.config_engine.build_channel_config(
            affected_ap_id,
            best_channel
        )
        
        return self.config_engine.build_network_config(
            [ap_config],
            metadata={
                'event_type': 'dfs_radar',
                'step': step,
                'old_channel': detected_channel,
                'new_channel': best_channel,
                'reason': 'DFS radar detected'
            }
        )
    
    def _handle_interference_burst(self, event: RRMEvent, step: int) -> Optional[NetworkConfig]:
        """
        Handle sudden interference burst (microwave, BLE).
        
        Args:
            event: Interference burst event
            step: Current step
            
        Returns:
            NetworkConfig with channel change to avoid interferer
        """
        affected_ap_id = event.metadata['ap_id']
        interferer_channel = event.metadata.get('interferer_channel', 6)
        interferer_type = event.metadata.get('interferer_type', 'Unknown')
        
        # Find channel furthest from interferer
        current_channel = self.config_engine.aps[affected_ap_id].channel
        allowed_channels = [1, 6, 11]
        
        # Remove interferer's channel and current channel
        candidate_channels = [
            ch for ch in allowed_channels 
            if ch != interferer_channel and ch != current_channel
        ]
        
        if not candidate_channels:
            candidate_channels = [ch for ch in allowed_channels if ch != interferer_channel]
        
        # Pick best from candidates
        best_channel = candidate_channels[0] if candidate_channels else current_channel
        
        if best_channel == current_channel:
            return None  # No better channel available
        
        ap_config = self.config_engine.build_channel_config(
            affected_ap_id,
            best_channel
        )
        
        return self.config_engine.build_network_config(
            [ap_config],
            metadata={
                'event_type': 'interference_burst',
                'interferer_type': interferer_type,
                'step': step
            }
        )
    
    def _handle_scheduled_event(self, event: RRMEvent, step: int) -> Optional[NetworkConfig]:
        """
        Handle scheduled events (exam hours, high-density scenarios).
        
        Args:
            event: Scheduled event
            step: Current step
            
        Returns:
            NetworkConfig with scheduled changes
        """
        event_name = event.metadata['event_name']
        action = event.metadata.get('action', 'unknown')
        
        if action == 'reduce_power':
            # Reduce all AP power (e.g., during exam)
            target_power = event.metadata.get('target_power', 15.0)
            ap_configs = [
                self.config_engine.build_power_config(ap.id, target_power)
                for ap in self.config_engine.aps.values()
            ]
            
            return self.config_engine.build_network_config(
                ap_configs,
                metadata={'event': event_name, 'step': step}
            )
        
        # Add more scheduled event actions as needed
        return None
    
    def _handle_emergency_qoe(self, event: RRMEvent, step: int) -> Optional[NetworkConfig]:
        """
        Handle critical QoE degradation.
        
        Args:
            event: Emergency QoE event
            step: Current step
            
        Returns:
            NetworkConfig with emergency optimization
        """
        affected_ap_id = event.metadata['ap_id']
        avg_qoe = event.metadata['avg_qoe']
        
        # Try power increase first
        current_power = self.config_engine.aps[affected_ap_id].tx_power
        new_power = min(current_power + 5, 30.0)  # Increase by 5dBm, max 30
        
        if new_power > current_power:
            ap_config = self.config_engine.build_power_config(
                affected_ap_id,
                new_power
            )
            
            return self.config_engine.build_network_config(
                [ap_config],
                metadata={'event_type': 'emergency_qoe', 'step': step}
            )
        
        return None
    
    def _handle_ap_failure(self, event: RRMEvent, step: int) -> Optional[NetworkConfig]:
        """
        Handle AP failure - redistribute load.
        
        Args:
            event: AP failure event
            step: Current step
            
        Returns:
            None (client redistribution handled by fast loop)
        """
        # AP failure requires client reassociation, not config change
        # This is handled by fast loop steering
        # Event just triggers awareness
        return None
    
    def register_scheduled_event(self, scheduled_event: ScheduledEvent):
        """Register a recurring scheduled event."""
        self.scheduled_events.append(scheduled_event)
    
    def _check_scheduled_events(self, step: int):
        """Check if any scheduled events should trigger."""
        now = datetime.now()
        
        for sched in self.scheduled_events:
            if not sched.active:
                continue
            
            # Check if current time matches schedule
            if (now.hour == sched.start_hour and 
                now.minute >= sched.start_minute and
                now.weekday() in sched.days_of_week):
                
                # Create event
                event = RRMEvent(
                    event_id=f"scheduled_{sched.name}_{step}",
                    event_type='scheduled_event',
                    priority=EventPriority.MEDIUM,
                    timestamp=time.time(),
                    metadata={
                        'event_name': sched.name,
                        'action': sched.action_type,
                        **sched.config_changes
                    },
                    action_type=sched.action_type
                )
                
                self.register_event(event)
    
    def _find_best_safe_channel(self, ap_id: int, safe_channels: List[int]) -> int:
        """Find best channel from safe options based on interference."""
        # Use sensing if available
        if self.sensing_api:
            sensing_results = self.sensing_api.compute_sensing_results()
            ap_sensing = sensing_results.get(ap_id)
            
            if ap_sensing:
                # Avoid channel with interferer
                interferer_channel = self._freq_to_channel(ap_sensing.center_frequency)
                safe_channels = [ch for ch in safe_channels if ch != interferer_channel]
        
        # Default to first safe channel
        return safe_channels[0] if safe_channels else 1
    
    def _freq_to_channel(self, freq_ghz: float) -> int:
        """Convert frequency to WiFi channel."""
        if 2.4 <= freq_ghz <= 2.5:
            channel = int((freq_ghz - 2.407) / 0.005)
            if channel <= 3:
                return 1
            elif channel <= 8:
                return 6
            else:
                return 11
        return 6
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get events loop statistics."""
        return {
            'events_processed': self.events_processed,
            'events_by_type': self.events_by_type,
            'pending_events': len(self.event_queue),
            'scheduled_events': len(self.scheduled_events)
        }
    
    def print_status(self):
        """Print events loop status."""
        print("\n" + "="*60)
        print("EVENTS LOOP CONTROLLER STATUS")
        print("="*60)
        print(f"Pending Events: {len(self.event_queue)}")
        print(f"Events Processed: {self.events_processed}")
        print(f"Scheduled Events: {len(self.scheduled_events)}")
        
        if self.events_by_type:
            print("\nEvents by Type:")
            for event_type, count in self.events_by_type.items():
                print(f"  {event_type}: {count}")
        print()
