"""
Rollback Manager for Event Loop.

Handles rollback tokens, automatic rollback detection,
and rollback execution with comprehensive monitoring.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time
from collections import defaultdict

from models.event_models import (
    RollbackToken, AuditRecord, PostActionMetrics,
    ExecutionStatus, Event
)
from datatype import AccessPoint


class RollbackManager:
    """
    Manages rollback tokens and automatic rollback logic.
    """
    
    def __init__(self, monitoring_window_sec: int = 300):
        """
        Initialize Rollback Manager.
        
        Args:
            monitoring_window_sec: Post-action monitoring window (default 5 min)
        """
        self.monitoring_window_sec = monitoring_window_sec
        
        # Active rollback tokens: token_id -> RollbackToken
        self.tokens: Dict[str, RollbackToken] = {}
        
        # Monitoring state: token_id -> monitoring data
        self.monitoring: Dict[str, Dict[str, Any]] = {}
        
        # Baseline metrics before action: token_id -> PostActionMetrics
        self.baselines: Dict[str, PostActionMetrics] = {}
        
        # Cooldown per AP: ap_id -> timestamp of last action
        self.ap_cooldowns: Dict[str, float] = {}
        
        # Statistics
        self.total_rollbacks = 0
        self.auto_rollbacks = 0
        self.manual_rollbacks = 0
    
    def create_token(self, ap_id: str, snapshot: Dict[str, Any],
                     trigger_event: Optional[Event] = None,
                     ttl_seconds: int = 3600) -> RollbackToken:
        """
        Create and register a new rollback token.
        
        Args:
            ap_id: Access Point identifier
            snapshot: Configuration snapshot
            trigger_event: Event that triggered the change
            ttl_seconds: Time-to-live
            
        Returns:
            RollbackToken
        """
        token = RollbackToken.create(ap_id, snapshot, trigger_event, ttl_seconds)
        self.tokens[token.token_id] = token
        
        # Initialize monitoring
        self.monitoring[token.token_id] = {
            'created_at': time.time(),
            'monitoring_end': time.time() + self.monitoring_window_sec,
            'samples': []
        }
        
        return token
    
    def start_monitoring(self, token_id: str, baseline: PostActionMetrics):
        """
        Start post-action monitoring for a token.
        
        Args:
            token_id: Rollback token ID
            baseline: Baseline metrics before action
        """
        if token_id not in self.tokens:
            raise ValueError(f"Token {token_id} not found")
        
        self.baselines[token_id] = baseline
        self.monitoring[token_id]['baseline'] = baseline
        self.monitoring[token_id]['monitoring_active'] = True
    
    def add_monitoring_sample(self, token_id: str, metrics: PostActionMetrics):
        """
        Add a monitoring sample during post-action window.
        
        Args:
            token_id: Rollback token ID
            metrics: Current metrics
        """
        if token_id not in self.monitoring:
            return
        
        self.monitoring[token_id]['samples'].append({
            'timestamp': time.time(),
            'metrics': metrics
        })
    
    def check_auto_rollback(self, token_id: str, 
                           current_metrics: PostActionMetrics) -> bool:
        """
        Check if automatic rollback should be triggered.
        
        Args:
            token_id: Rollback token ID
            current_metrics: Current network metrics
            
        Returns:
            True if rollback should be triggered
        """
        if token_id not in self.baselines:
            return False
        
        baseline = self.baselines[token_id]
        
        # Check rollback conditions
        should_rollback = current_metrics.should_rollback(baseline)
        
        if should_rollback:
            self.monitoring[token_id]['rollback_triggered'] = True
            self.monitoring[token_id]['rollback_reason'] = \
                self._get_rollback_reason(baseline, current_metrics)
        
        return should_rollback
    
    def is_monitoring_active(self, token_id: str) -> bool:
        """
        Check if monitoring window is still active.
        
        Args:
            token_id: Rollback token ID
            
        Returns:
            True if monitoring is active
        """
        if token_id not in self.monitoring:
            return False
        
        monitoring_end = self.monitoring[token_id]['monitoring_end']
        return time.time() < monitoring_end
    
    def execute_rollback(self, token_id: str, reason: str = "auto") -> Dict[str, Any]:
        """
        Execute rollback to previous configuration.
        
        Args:
            token_id: Rollback token ID
            reason: Reason for rollback ("auto" or manual reason)
            
        Returns:
            Rollback result dictionary
        """
        if token_id not in self.tokens:
            return {'success': False, 'error': 'Token not found'}
        
        token = self.tokens[token_id]
        
        # Check if expired
        if token.is_expired():
            return {'success': False, 'error': 'Token expired'}
        
        # Get snapshot configuration
        snapshot = token.snapshot
        
        # Track rollback
        self.total_rollbacks += 1
        if reason == "auto":
            self.auto_rollbacks += 1
        else:
            self.manual_rollbacks += 1
        
        # Mark token as used
        self.monitoring[token_id]['rolled_back'] = True
        self.monitoring[token_id]['rollback_time'] = time.time()
        self.monitoring[token_id]['rollback_reason'] = reason
        
        return {
            'success': True,
            'token_id': token_id,
            'ap_id': token.ap_id,
            'snapshot': snapshot,
            'reason': reason
        }
    
    def can_rollback(self, token_id: str) -> bool:
        """Check if rollback is possible for a token"""
        if token_id not in self.tokens:
            return False
        
        token = self.tokens[token_id]
        
        # Check if already rolled back
        if self.monitoring.get(token_id, {}).get('rolled_back', False):
            return False
        
        # Check if expired
        if token.is_expired():
            return False
        
        return True
    
    def check_ap_cooldown(self, ap_id: str, cooldown_seconds: int = 10) -> bool:
        """
        Check if AP is in cooldown period.
        
        Args:
            ap_id: Access Point identifier
            cooldown_seconds: Cooldown period in seconds
            
        Returns:
            True if in cooldown (action should be blocked)
        """
        if ap_id not in self.ap_cooldowns:
            return False
        
        last_action = self.ap_cooldowns[ap_id]
        return (time.time() - last_action) < cooldown_seconds
    
    def set_ap_cooldown(self, ap_id: str):
        """Set cooldown timestamp for AP after an action"""
        self.ap_cooldowns[ap_id] = time.time()
    
    def cleanup_expired_tokens(self):
        """Remove expired tokens and monitoring data"""
        current_time = time.time()
        
        expired_tokens = [
            token_id for token_id, token in self.tokens.items()
            if token.is_expired()
        ]
        
        for token_id in expired_tokens:
            del self.tokens[token_id]
            if token_id in self.monitoring:
                del self.monitoring[token_id]
            if token_id in self.baselines:
                del self.baselines[token_id]
    
    def get_token(self, token_id: str) -> Optional[RollbackToken]:
        """Get rollback token by ID"""
        return self.tokens.get(token_id)
    
    def get_active_tokens(self) -> List[RollbackToken]:
        """Get all active (non-expired) tokens"""
        return [token for token in self.tokens.values() if token.is_valid()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rollback manager statistics"""
        return {
            'total_tokens': len(self.tokens),
            'active_tokens': len(self.get_active_tokens()),
            'total_rollbacks': self.total_rollbacks,
            'auto_rollbacks': self.auto_rollbacks,
            'manual_rollbacks': self.manual_rollbacks,
            'monitoring_active_count': sum(
                1 for m in self.monitoring.values()
                if m.get('monitoring_active', False)
            )
        }
    
    def _get_rollback_reason(self, baseline: PostActionMetrics,
                            current: PostActionMetrics) -> str:
        """Generate human-readable rollback reason"""
        reasons = []
        
        if baseline.per_p95 > 0 and current.per_p95 > baseline.per_p95 * 1.30:
            delta = ((current.per_p95 / baseline.per_p95) - 1) * 100
            reasons.append(f"PER increased by {delta:.1f}%")
        
        if baseline.retry_rate_p95 > 0 and current.retry_rate_p95 > baseline.retry_rate_p95 * 1.30:
            delta = ((current.retry_rate_p95 / baseline.retry_rate_p95) - 1) * 100
            reasons.append(f"Retry rate increased by {delta:.1f}%")
        
        if current.client_disconnection_rate > 10.0:
            reasons.append(f"Disconnect rate: {current.client_disconnection_rate:.1f} clients/min")
        
        if current.throughput_degradation_pct > 40.0:
            reasons.append(f"Throughput degraded by {current.throughput_degradation_pct:.1f}%")
        
        if current.new_critical_events > 0:
            reasons.append(f"{current.new_critical_events} new critical events")
        
        return "; ".join(reasons) if reasons else "Unknown degradation"
    
    def print_status(self):
        """Print rollback manager status"""
        print("\n" + "="*60)
        print("ROLLBACK MANAGER STATUS")
        print("="*60)
        stats = self.get_statistics()
        print(f"Active Tokens: {stats['active_tokens']}/{stats['total_tokens']}")
        print(f"Total Rollbacks: {stats['total_rollbacks']}")
        print(f"  - Automatic: {stats['auto_rollbacks']}")
        print(f"  - Manual: {stats['manual_rollbacks']}")
        print(f"Monitoring Active: {stats['monitoring_active_count']}")
        print()
