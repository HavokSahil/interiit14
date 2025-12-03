"""
Event Detector for RRMEngine.

Automatically detects network events from sensing and QoE data:
- Interference bursts
- Critical QoE degradation
- Sudden changes requiring immediate action
"""

from typing import Dict, List, Any
import time

from events_loop_controller import RRMEvent, EventPriority


class EventDetector:
    """
    Automatically detect events from sensing and QoE data.
    """
    
    def __init__(self,
                 interference_threshold: float = 0.7,
                 qoe_critical_threshold: float = 0.3):
        """
        Initialize Event Detector.
        
        Args:
            interference_threshold: Confidence threshold for interference events
            qoe_critical_threshold: QoE threshold for emergency events
        """
        self.interference_threshold = interference_threshold
        self.qoe_critical_threshold = qoe_critical_threshold
        
        # Track previous state to detect changes
        self.previous_interference = {}
        self.previous_qoe = {}
    
    def detect_events(self,
                     sensing_results: Dict[int, Any],
                     qoe_views: Dict[int, Any],
                     step: int) -> List[RRMEvent]:
        """
        Detect events from current network state.
        
        Args:
            sensing_results: Current sensing data
            qoe_views: Current QoE views
            step: Current simulation step
            
        Returns:
            List of detected RRMEvents
        """
        events = []
        
        # 1. Detect interference bursts
        events.extend(self._detect_interference_bursts(sensing_results, step))
        
        # 2. Detect critical QoE degradation
        events.extend(self._detect_qoe_emergencies(qoe_views, step))
        
        return events
    
    def _detect_interference_bursts(self,
                                   sensing_results: Dict[int, Any],
                                   step: int) -> List[RRMEvent]:
        """Detect interference bursts from sensing data."""
        events = []
        
        for ap_id, sensing in sensing_results.items():
            # Check if high-confidence interferer detected
            if sensing.confidence >= self.interference_threshold:
                # Check if this is a new burst (not detected before)
                prev_confidence = self.previous_interference.get(ap_id, 0.0)
                
                if prev_confidence < self.interference_threshold:
                    # New burst detected
                    event = RRMEvent(
                        event_id=f"intrf_burst_{ap_id}_{step}",
                        event_type='interference_burst',
                        priority=EventPriority.HIGH,
                        timestamp=time.time(),
                        metadata={
                            'ap_id': ap_id,
                            'interferer_type': sensing.major_interferer_type,
                            'interferer_channel': self._freq_to_channel(sensing.center_frequency),
                            'confidence': sensing.confidence
                        },
                        action_type='channel_switch',
                        target_ap_ids=[ap_id]
                    )
                    events.append(event)
            
            # Update previous state
            self.previous_interference[ap_id] = sensing.confidence
        
        return events
    
    def _detect_qoe_emergencies(self,
                               qoe_views: Dict[int, Any],
                               step: int) -> List[RRMEvent]:
        """Detect critical QoE degradation."""
        events = []
        
        for ap_id, view in qoe_views.items():
            if view.num_clients == 0:
                continue
            
            # Check for critical QoE
            if view.avg_qoe < self.qoe_critical_threshold:
                # Check if this is sudden degradation
                prev_qoe = self.previous_qoe.get(ap_id, 1.0)
                
                if prev_qoe >= self.qoe_critical_threshold:
                    # Sudden critical degradation
                    event = RRMEvent(
                        event_id=f"qoe_emergency_{ap_id}_{step}",
                        event_type='emergency_qoe',
                        priority=EventPriority.HIGH,
                        timestamp=time.time(),
                        metadata={
                            'ap_id': ap_id,
                            'avg_qoe': view.avg_qoe,
                            'min_qoe': view.min_qoe,
                            'num_clients': view.num_clients
                        },
                        action_type='power_adjust',
                        target_ap_ids=[ap_id]
                    )
                    events.append(event)
            
            # Update previous state
            self.previous_qoe[ap_id] = view.avg_qoe
        
        return events
    
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
