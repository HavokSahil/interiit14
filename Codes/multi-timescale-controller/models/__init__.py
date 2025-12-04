"""
Models package for RRM Engine Event Loop.

Contains:
- event_models: Event, RollbackToken, AuditRecord data structures
- rollback_manager: Rollback token management and automatic rollback
- audit_logger: Tamper-proof audit logging with HMAC signatures
- channel_selector: Emergency channel selection algorithm
- enhanced_event_loop: Main Event Loop controller
"""

from models.event_models import (
    Event, EventType, Severity, SensingSource, ActionType,
    ExecutionStatus, RollbackToken, AuditRecord, ConfigurationChange,
    PostActionMetrics, EVENT_ACTION_MATRIX, hash_identifier
)

from models.rollback_manager import RollbackManager
from models.audit_logger import AuditLogger
from models.channel_selector import EmergencyChannelSelector, select_emergency_channel
from models.enhanced_event_loop import EnhancedEventLoop

__all__ = [
    # Event models
    'Event', 'EventType', 'Severity', 'SensingSource', 'ActionType',
    'ExecutionStatus', 'RollbackToken', 'AuditRecord', 'ConfigurationChange',
    'PostActionMetrics', 'EVENT_ACTION_MATRIX', 'hash_identifier',
    
    # Managers
    'RollbackManager', 'AuditLogger', 'EmergencyChannelSelector',
    
    # Main controller
    'EnhancedEventLoop',
    
    # Helper functions
    'select_emergency_channel',
]
