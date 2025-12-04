"""
Audit Logger for Event Loop.

Provides tamper-proof audit logging with HMAC signatures,
append-only storage, and compliance-ready audit trails.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os
from pathlib import Path

from models.event_models import (
    AuditRecord, Event, RollbackToken, ConfigurationChange,
    ActionType, ExecutionStatus
)


class AuditLogger:
    """
    Append-only audit logger with HMAC signatures.
    """
    
    def __init__(self, log_dir: str = "audit_logs", 
                 secret_key: Optional[str] = None):
        """
        Initialize Audit Logger.
        
        Args:
            log_dir: Directory for audit log files
            secret_key: Secret key for HMAC signatures
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Use environment variable or default (INSECURE - use KMS in production)
        self.secret_key = secret_key or os.getenv(
            'RRM_AUDIT_SECRET',
            'default_secret_key_CHANGE_IN_PRODUCTION'
        )
        
        # Current log file (rotated daily)
        self.current_log_file = None
        self.rotate_log_file()
        
        # In-memory cache for recent records
        self.recent_records: List[AuditRecord] = []
        self.max_recent = 1000
        
        # Statistics
        self.total_records = 0
        self.records_by_status: Dict[str, int] = {}
    
    def rotate_log_file(self):
        """Rotate log file (daily rotation)"""
        date_str = datetime.utcnow().strftime("%Y%m%d")
        self.current_log_file = self.log_dir / f"audit_{date_str}.jsonl"
    
    def log_action(self, record: AuditRecord) -> str:
        """
        Log an audit record with signature.
        
        Args:
            record: AuditRecord to log
            
        Returns:
            Audit ID
        """
        # Generate signature
        record.generate_signature(self.secret_key)
        
        # Rotate log file if needed (daily)
        self.rotate_log_file()
        
        # Write to append-only log
        with open(self.current_log_file, 'a') as f:
            json_record = self._serialize_record(record)
            f.write(json.dumps(json_record) + '\n')
        
        # Add to recent cache
        self.recent_records.append(record)
        if len(self.recent_records) > self.max_recent:
            self.recent_records.pop(0)
        
        # Update statistics
        self.total_records += 1
        status_key = str(record.execution_status)
        self.records_by_status[status_key] = \
            self.records_by_status.get(status_key, 0) + 1
        
        return record.audit_id
    
    def create_event_action_record(self,
                                   event: Event,
                                   ap_id: str,
                                   action_type: ActionType,
                                   config_changes: List[ConfigurationChange],
                                   rollback_token: RollbackToken,
                                   reason: str = "") -> AuditRecord:
        """
        Create an audit record for an event-driven action.
        
        Args:
            event: Triggering event
            ap_id: Access Point ID
            action_type: Type of action taken
            config_changes: List of configuration changes
            rollback_token: Associated rollback token
            reason: Human-readable justification
            
        Returns:
            AuditRecord
        """
        import uuid
        
        record = AuditRecord(
            audit_id=str(uuid.uuid4()),
            event=event,
            ap_id=ap_id,
            action_type=action_type,
            configuration_changes=config_changes,
            rollback_token=rollback_token.token_id,
            rollback_expires_at=rollback_token.expires_at,
            rollback_eligible=True,
            reason=reason,
            confidence_score=event.detection_confidence,
            regulatory_check_passed=True,  # Set externally
            execution_status=ExecutionStatus.PENDING
        )
        
        return record
    
    def update_execution_status(self, audit_id: str, 
                               status: ExecutionStatus,
                               latency_ms: int = 0,
                               ap_response: str = ""):
        """
        Update execution status of an audit record.
        
        Args:
            audit_id: Audit record ID
            status: New execution status
            latency_ms: Execution latency in milliseconds
            ap_response: Response from AP
        """
        # Find in recent records
        for record in self.recent_records:
            if record.audit_id == audit_id:
                record.execution_status = status
                record.execution_latency_ms = latency_ms
                record.ap_response = ap_response
                
                # Re-log with updated status
                self.log_action(record)
                break
    
    def mark_rollback(self, audit_id: str, rollback_reason: str):
        """
        Mark an audit record as rolled back.
        
        Args:
            audit_id: Audit record ID
            rollback_reason: Reason for rollback
        """
        for record in self.recent_records:
            if record.audit_id == audit_id:
                record.execution_status = ExecutionStatus.ROLLED_BACK
                record.auto_rollback_triggered = True
                record.degradation_detected = True
                
                # Update reason
                record.reason += f" | ROLLBACK: {rollback_reason}"
                
                # Re-log
                self.log_action(record)
                break
    
    def verify_record(self, record: AuditRecord) -> bool:
        """
        Verify HMAC signature of a record.
        
        Args:
            record: AuditRecord to verify
            
        Returns:
            True if signature is valid
        """
        return record.verify_signature(self.secret_key)
    
    def query_by_ap(self, ap_id: str, limit: int = 100) -> List[AuditRecord]:
        """
        Query audit records for a specific AP.
        
        Args:
            ap_id: Access Point ID
            limit: Maximum number of records
            
        Returns:
            List of AuditRecords
        """
        results = [r for r in self.recent_records if r.ap_id == ap_id]
        return results[-limit:]
    
    def query_by_event_type(self, event_type: str, 
                           limit: int = 100) -> List[AuditRecord]:
        """
        Query audit records by event type.
        
        Args:
            event_type: Event type to filter
            limit: Maximum number of records
            
        Returns:
            List of AuditRecords
        """
        results = [
            r for r in self.recent_records 
            if r.event and r.event.event_type == event_type
        ]
        return results[-limit:]
    
    def export_audit_trail(self, ap_id: Optional[str] = None,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> str:
        """
        Export audit trail for compliance.
        
        Args:
            ap_id: Optional AP filter
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            Path to exported file
        """
        # For MVP, just copy relevant records
        # In production, query compressed Parquet files
        
        export_file = self.log_dir / f"export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        with open(export_file, 'w') as f:
            for record in self.recent_records:
                # Apply filters
                if ap_id and record.ap_id != ap_id:
                    continue
                
                if start_date and record.timestamp_utc < start_date:
                    continue
                
                if end_date and record.timestamp_utc > end_date:
                    continue
                
                # Write record
                json_record = self._serialize_record(record)
                f.write(json.dumps(json_record) + '\n')
        
        return str(export_file)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit logger statistics"""
        return {
            'total_records': self.total_records,
            'records_by_status': self.records_by_status,
            'recent_cache_size': len(self.recent_records),
            'log_file': str(self.current_log_file)
        }
    
    def _serialize_record(self, record: AuditRecord) -> Dict[str, Any]:
        """Serialize audit record to JSON-compatible dict"""
        return {
            'audit_id': record.audit_id,
            'record_type': record.record_type,
            'timestamp_utc': record.timestamp_utc.isoformat(),
            'event': self._serialize_event(record.event) if record.event else None,
            'ap_id': record.ap_id,
            'action_type': str(record.action_type) if record.action_type else None,
            'configuration_changes': [
                {
                    'param': c.param,
                    'old_value': c.old_value,
                    'new_value': c.new_value,
                    'radio': c.radio
                }
                for c in record.configuration_changes
            ],
            'rollback_token': record.rollback_token,
            'rollback_expires_at': record.rollback_expires_at.isoformat() 
                if record.rollback_expires_at else None,
            'rollback_eligible': record.rollback_eligible,
            'reason': record.reason,
            'confidence_score': record.confidence_score,
            'regulatory_check_passed': record.regulatory_check_passed,
            'execution_status': str(record.execution_status),
            'execution_latency_ms': record.execution_latency_ms,
            'ap_response': record.ap_response,
            'post_action_monitoring_window_sec': record.post_action_monitoring_window_sec,
            'degradation_detected': record.degradation_detected,
            'auto_rollback_triggered': record.auto_rollback_triggered,
            'actor': record.actor,
            'blast_radius_id': record.blast_radius_id,
            'no_pii': record.no_pii,
            'hashed_identifiers': record.hashed_identifiers,
            'signature': record.signature,
            'signature_key_version': record.signature_key_version
        }
    
    def _serialize_event(self, event: Event) -> Dict[str, Any]:
        """Serialize event to JSON-compatible dict"""
        return {
            'event_id': event.event_id,
            'event_type': str(event.event_type),
            'severity': int(event.severity),
            'ap_id': event.ap_id,
            'radio': event.radio,
            'timestamp_utc': event.timestamp_utc.isoformat(),
            'detection_confidence': event.detection_confidence,
            'metadata': event.metadata,
            'sensing_source': str(event.sensing_source)
        }
    
    def print_status(self):
        """Print audit logger status"""
        print("\n" + "="*60)
        print("AUDIT LOGGER STATUS")
        print("="*60)
        stats = self.get_statistics()
        print(f"Total Records: {stats['total_records']}")
        print(f"Recent Cache: {stats['recent_cache_size']}")
        print(f"Current Log File: {stats['log_file']}")
        
        if stats['records_by_status']:
            print("\nRecords by Status:")
            for status, count in stats['records_by_status'].items():
                print(f"  {status}: {count}")
        print()
