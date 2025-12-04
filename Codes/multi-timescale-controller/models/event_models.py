"""
Event Loop Data Models.

Models for events, rollback tokens, and audit records
for the RRM Engine Event Loop.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum, IntEnum
from datetime import datetime
import uuid
import hashlib
import hmac


class EventType(str, Enum):
    """Types of network events"""
    DFS_RADAR = "dfs_radar"
    NON_WIFI_BURST = "non_wifi_burst"
    HW_FAILURE = "hw_failure"
    DENSITY_SPIKE = "density_spike"
    SPECTRUM_SAT = "spectrum_saturation"
    SECURITY = "security"
    EMERGENCY_QOE = "emergency_qoe"
    AP_FAILURE = "ap_failure"
    SCHEDULED = "scheduled_event"


class Severity(IntEnum):
    """Event severity (lower = more critical)"""
    CRITICAL = 1  # DFS radar, hardware failure
    HIGH = 2      # Non-Wi-Fi >80% duty cycle
    MEDIUM = 3    # Density spike, thermal throttling
    LOW = 4       # Informational


class SensingSource(str, Enum):
    """Source of event detection"""
    ADDITIONAL_RADIO = "additional_radio"
    SERVING_RADIO = "serving_radio"
    EXTERNAL = "external"


class ActionType(str, Enum):
    """Types of RRM actions"""
    CHANNEL_CHANGE = "channel_change"
    POWER_ADJUST = "power_adjust"
    OBSS_PD_TUNE = "obss_pd_tune"
    ADMISSION_CONTROL = "admission_control"
    WIDTH_REDUCTION = "width_reduction"
    FAILOVER = "failover"
    CLIENT_STEERING = "client_steering"


class ExecutionStatus(str, Enum):
    """Status of action execution"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    ROLLED_BACK = "rolled_back"
    PENDING = "pending"


@dataclass
class Event:
    """Represents a network event requiring RRM action"""
    event_id: str
    event_type: EventType
    severity: Severity
    ap_id: str  # hashed site-specific ID
    radio: str  # "2g", "5g", "6g"
    timestamp_utc: datetime
    detection_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    sensing_source: SensingSource = SensingSource.SERVING_RADIO
    
    def __post_init__(self):
        """Validate event data"""
        if not 0.0 <= self.detection_confidence <= 1.0:
            raise ValueError("Confidence must be in [0, 1]")
    
    def __lt__(self, other):
        """For priority queue ordering (by severity)"""
        return self.severity < other.severity


@dataclass
class RollbackToken:
    """Token for rolling back configuration changes"""
    token_id: str
    created_at: datetime
    expires_at: datetime
    loop_type: str  # "EVENT", "FAST", "SLOW"
    ap_id: str
    snapshot: Dict[str, Any]  # Full AP configuration before change
    trigger_event: Optional[Event] = None
    action_taken: str = ""
    
    @classmethod
    def create(cls, ap_id: str, snapshot: Dict[str, Any], 
               trigger_event: Optional[Event] = None,
               ttl_seconds: int = 3600) -> 'RollbackToken':
        """
        Create a new rollback token.
        
        Args:
            ap_id: Access Point identifier
            snapshot: Configuration snapshot before change
            trigger_event: Event that triggered the change
            ttl_seconds: Time-to-live in seconds (default 1 hour)
            
        Returns:
            RollbackToken instance
        """
        now = datetime.utcnow()
        expires = datetime.fromtimestamp(now.timestamp() + ttl_seconds)
        
        token_id = f"evtloop-{ap_id}-{int(now.timestamp())}-{uuid.uuid4().hex[:8]}"
        
        return cls(
            token_id=token_id,
            created_at=now,
            expires_at=expires,
            loop_type="EVENT",
            ap_id=ap_id,
            snapshot=snapshot,
            trigger_event=trigger_event
        )
    
    def is_expired(self) -> bool:
        """Check if token has expired"""
        return datetime.utcnow() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if token is valid (not expired)"""
        return not self.is_expired()


@dataclass
class ConfigurationChange:
    """Represents a single parameter change"""
    param: str
    old_value: Any
    new_value: Any
    radio: Optional[str] = None


@dataclass
class AuditRecord:
    """Audit record for Event Loop actions"""
    audit_id: str
    record_type: str = "EVENT_LOOP_ACTION"
    timestamp_utc: datetime = field(default_factory=datetime.utcnow)
    
    # Event details
    event: Optional[Event] = None
    
    # Action details
    ap_id: str = ""
    action_type: Optional[ActionType] = None
    configuration_changes: List[ConfigurationChange] = field(default_factory=list)
    
    # Rollback information
    rollback_token: str = ""
    rollback_expires_at: Optional[datetime] = None
    rollback_eligible: bool = True
    
    # Justification
    reason: str = ""
    confidence_score: Optional[float] = None
    regulatory_check_passed: bool = False
    
    # Execution
    execution_status: ExecutionStatus = ExecutionStatus.PENDING
    execution_latency_ms: int = 0
    ap_response: str = ""
    
    # Post-action monitoring
    post_action_monitoring_window_sec: int = 300  # 5 minutes
    degradation_detected: bool = False
    auto_rollback_triggered: bool = False
    
    # Attribution
    actor: str = "EVENT_LOOP_AUTOMATED"
    blast_radius_id: Optional[str] = None
    
    # Privacy
    no_pii: bool = True
    hashed_identifiers: List[str] = field(default_factory=lambda: ["ap_id"])
    
    # Signature (computed after creation)
    signature: str = ""
    signature_key_version: int = 1
    
    def generate_signature(self, secret_key: str) -> str:
        """
        Generate HMAC-SHA256 signature for audit record.
        
        Args:
            secret_key: Secret key for HMAC
            
        Returns:
            Hex-encoded signature
        """
        # Create canonical representation
        data_to_sign = f"{self.audit_id}|{self.timestamp_utc.isoformat()}|{self.ap_id}|{self.action_type}|{self.execution_status}"
        
        # Generate HMAC
        signature = hmac.new(
            secret_key.encode('utf-8'),
            data_to_sign.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        self.signature = signature
        return signature
    
    def verify_signature(self, secret_key: str) -> bool:
        """
        Verify the HMAC signature.
        
        Args:
            secret_key: Secret key for HMAC
            
        Returns:
            True if signature is valid
        """
        expected = self.generate_signature(secret_key)
        return hmac.compare_digest(self.signature, expected)


@dataclass
class PostActionMetrics:
    """Metrics collected after an action for rollback decision"""
    per_p95: float  # 95th percentile packet error rate
    retry_rate_p95: float  # 95th percentile retry rate
    client_disconnection_rate: float  # clients/minute
    throughput_degradation_pct: float  # percentage
    new_critical_events: int  # count of new critical events
    
    def should_rollback(self, baseline: 'PostActionMetrics') -> bool:
        """
        Determine if rollback is needed based on degradation.
        
        Args:
            baseline: Metrics before the action
            
        Returns:
            True if rollback should be triggered
        """
        # Rollback if ANY condition met:
        conditions = [
            # PER increased by >30%
            self.per_p95 > baseline.per_p95 * 1.30,
            
            # Retry rate increased by >30%
            self.retry_rate_p95 > baseline.retry_rate_p95 * 1.30,
            
            # Client disconnections >10/min for >2 min
            self.client_disconnection_rate > 10.0,
            
            # Throughput degradation >40%
            self.throughput_degradation_pct > 40.0,
            
            # New critical event on new channel
            self.new_critical_events > 0
        ]
        
        return any(conditions)


def hash_identifier(identifier: str, site_secret: str) -> str:
    """
    Hash an identifier (AP MAC, Client MAC) for privacy.
    
    Args:
        identifier: Raw identifier (e.g., MAC address)
        site_secret: Site-specific secret
        
    Returns:
        Hashed identifier (hex string)
    """
    return hmac.new(
        site_secret.encode('utf-8'),
        identifier.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()[:16]  # Use first 16 chars for readability


# Decision Matrix: Event → Action mapping with confidence thresholds
EVENT_ACTION_MATRIX = {
    EventType.DFS_RADAR: {
        "confidence_threshold": None,  # N/A - regulatory requirement
        "primary_action": ActionType.CHANNEL_CHANGE,
        "secondary_action": ActionType.POWER_ADJUST,
        "rollback_eligible": True,
        "rollback_delay_sec": 3600  # After 1 hour
    },
    EventType.NON_WIFI_BURST: {
        "confidence_threshold": 0.80,
        "primary_action": ActionType.CHANNEL_CHANGE,
        "secondary_action": ActionType.OBSS_PD_TUNE,
        "rollback_eligible": True,
        "rollback_delay_sec": 300  # After 5 minutes
    },
    EventType.HW_FAILURE: {
        "confidence_threshold": None,
        "primary_action": ActionType.FAILOVER,
        "secondary_action": None,
        "rollback_eligible": False,
        "rollback_delay_sec": 0
    },
    EventType.DENSITY_SPIKE: {
        "confidence_threshold": None,
        "primary_action": ActionType.ADMISSION_CONTROL,
        "secondary_action": ActionType.CLIENT_STEERING,
        "rollback_eligible": True,
        "rollback_delay_sec": 600  # After spike ends
    },
    EventType.SPECTRUM_SAT: {
        "confidence_threshold": None,
        "primary_action": ActionType.OBSS_PD_TUNE,
        "secondary_action": ActionType.WIDTH_REDUCTION,
        "rollback_eligible": True,
        "rollback_delay_sec": 600
    }
}
