# RRM Engine Implementation Plan - Simplified MVP

## Overview
This plan implements a minimal viable Multi-Timescale RRM Controller with three control loops (Event, Fast, Slow) for AI-assisted radio resource management.

---

## Phase 1: Core Data Models & Infrastructure (Week 1)

### 1.1 Event & Audit Models (`models.py`)
**Priority: CRITICAL**

```python
@dataclass
class Event:
    event_id: str
    event_type: EventType  # DFS_RADAR, NON_WIFI_BURST, HW_FAILURE, etc.
    severity: Severity  # CRITICAL, HIGH, MEDIUM, LOW
    ap_id: str  # hashed
    radio: RadioBand  # 2g, 5g, 6g
    timestamp_utc: datetime
    detection_confidence: float
    metadata: dict

@dataclass
class RollbackToken:
    token_id: str
    created_at: datetime
    expires_at: datetime
    loop_type: str  # EVENT, FAST, SLOW
    ap_id: str
    snapshot: dict  # full config before change
    trigger_event: Optional[Event]
    
@dataclass
class AuditRecord:
    audit_id: str
    record_type: str
    timestamp_utc: datetime
    event: Optional[Event]
    ap_id: str
    action_type: str
    configuration_changes: list
    rollback_token: str
    execution_status: str
    signature: str
```

**Files to Create:**
- `models/events.py`
- `models/rollback.py`
- `models/audit.py`

---

## Phase 2: Event Loop (Weeks 2-3)

### 2.1 Event Ingestion & Priority Queue (`event_loop.py`)
**Priority: HIGH**

**Core Components:**
1. **EventQueue**: Priority-based queue with rate limiting
2. **EventClassifier**: Maps raw events to priority levels
3. **EventActionDecider**: Decision matrix for event → action mapping

**Key Methods:**
```python
class EventLoop:
    def ingest_event(self, event: Event) -> None:
        """Add event to priority queue with coalescing"""
        
    def process_event(self, event: Event) -> Action:
        """Decide action based on event type and confidence"""
        
    def execute_emergency_action(self, action: Action) -> RollbackToken:
        """Execute action with rollback token generation"""
        
    def monitor_post_action(self, token: RollbackToken) -> bool:
        """5-min monitoring, auto-rollback if degraded"""
```

**Integration Points:**
- Input: DFS detector, spectrum analyzer, AP telemetry
- Output: Direct AP config commands (bypass planner)
- Audit: Every action logged with signature

### 2.2 Emergency Channel Selection (`channel_selector.py`)
**Priority: HIGH**

**Algorithm:**
```python
def select_emergency_channel(
    ap_id: str,
    current_channel: int,
    radio: RadioBand,
    interference_data: dict
) -> int:
    """
    1. Exclude: Current, DFS-pending, interfered, regulatory-invalid
    2. Score: Interference + neighbor overlap + client compat
    3. Return: Lowest score or fallback
    """
```

**Files to Create:**
- `event_loop/event_loop.py`
- `event_loop/channel_selector.py`
- `event_loop/rollback_manager.py`

---

## Phase 3: Fast Loop - Bayesian Optimization (Weeks 4-6)

### 3.1 Telemetry Aggregator (`telemetry.py`)
**Priority: HIGH**

**Data Collection:**
```python
class TelemetryAggregator:
    def collect_ap_metrics(self, ap_id: str) -> dict:
        """
        Returns:
        {
            'rssi_p50': float,
            'rssi_p95': float,
            'snr_p95': float,
            'per_p95': float,
            'retry_rate_p95': float,
            'airtime_util_p50': float,
            'cca_busy_p50': float,
            'station_count': int
        }
        """
```

**Integration:**
- Pull from existing `APMetricsManager`
- Add additional radio sensing data (if available)
- Cache with 10-second refresh

### 3.2 Bayesian Optimizer (`fast_loop_optimizer.py`)
**Priority: CRITICAL**

**Using scikit-optimize or GPyOpt:**

```python
class FastLoopBO:
    def __init__(self):
        self.gp = GaussianProcessRegressor(kernel=Matern(nu=2.5))
        self.config_space = {
            'channel': Categorical([36, 40, 44, ...]),
            'tx_power_dbm': Categorical([8, 11, 14, 17, 20, 23]),
            'width_mhz': Categorical([20, 40, 80]),
            'obss_pd_dbm': Integer(-82, -62)
        }
        
    def propose_candidate(self, ap_id: str) -> dict:
        """Use EI or UCB acquisition to propose next config"""
        
    def update_model(self, config: dict, reward: float):
        """Update GP with observed reward"""
```

**Reward Function:**
```python
def compute_reward(qoe: float, penalties: dict) -> float:
    return qoe - 0.3 * sum(penalties.values())
```

### 3.3 Change Planner & Guardrails (`change_planner.py`)
**Priority: HIGH**

**Guardrail Checks:**
```python
class ChangePlanner:
    def validate_proposal(self, proposal: ChangeProposal) -> bool:
        checks = [
            self.check_regulatory_compliance(proposal),
            self.check_blast_radius(proposal),
            self.check_change_budget(proposal),
            self.check_hysteresis(proposal),
            self.check_qoe_gain_threshold(proposal)
        ]
        return all(checks)
        
    def execute_change(self, proposal: ChangeProposal) -> RollbackToken:
        """Execute with observation window and rollback logic"""
```

**Files to Create:**
- `fast_loop/telemetry.py`
- `fast_loop/optimizer.py`
- `fast_loop/change_planner.py`
- `fast_loop/guardrails.py`

---

## Phase 4: Slow Loop - Safe RL (Weeks 7-9)

### 4.1 Graph-Based State Representation (`slow_loop_env.py`)
**Priority: MEDIUM**

**Environment:**
```python
class SlowLoopEnv(gym.Env):
    """
    State: Interference graph + AP features (from GNN)
    Action: Multi-AP channel/power assignment
    Reward: Network-wide QoE - penalty
    """
    
    def step(self, action: dict) -> Tuple[State, float, bool, dict]:
        """Simulate action, compute reward, check constraints"""
```

### 4.2 Conservative Q-Learning Agent (`cql_agent.py`)
**Priority: MEDIUM**

**Using Stable-Baselines3 or TF-Agents:**

```python
class CQLAgent:
    def __init__(self):
        # Conservative Q-Learning with constraint penalties
        self.policy = SAC(...)  # or custom CQL
        
    def select_action(self, state: State, safe_mode=True) -> dict:
        """Return multi-AP config changes with safety bounds"""
        
    def train_offline(self, replay_buffer: Buffer):
        """Train on historical data with conservatism penalty"""
```

**Safety Constraints:**
- Max 3 APs changed per iteration
- Rollback if network QoE drops >10%
- Human-in-loop approval for large changes

**Files to Create:**
- `slow_loop/environment.py`
- `slow_loop/cql_agent.py`
- `slow_loop/graph_coloring.py` (alternative heuristic)

---

## Phase 5: Integration & Orchestration (Week 10)

### 5.1 Master Scheduler (`scheduler.py`)
**Priority: CRITICAL**

**Orchestration:**
```python
class RRMScheduler:
    def __init__(self):
        self.event_loop = EventLoop()
        self.fast_loop = FastLoop()
        self.slow_loop = SlowLoop()
        
    async def run(self):
        # Event loop: Always listening (async handler)
        asyncio.create_task(self.event_loop.run())
        
        # Fast loop: Every 60-300s adaptive
        while True:
            await self.fast_loop.execute()
            await asyncio.sleep(self.fast_loop.adaptive_period())
            
        # Slow loop: Every 4-24 hours
        asyncio.create_task(self.slow_loop_scheduler())
```

### 5.2 Audit & Compliance Engine (`audit_logger.py`)
**Priority: HIGH**

**Features:**
- Append-only audit DB (SQLite or PostgreSQL)
- HMAC-SHA256 signature for tamper-proofing
- Query API for compliance reports

```python
class AuditLogger:
    def log_action(self, record: AuditRecord):
        record.signature = self.sign_record(record)
        self.db.insert(record)
        
    def verify_integrity(self, audit_id: str) -> bool:
        """Verify signature of stored record"""
```

**Files to Create:**
- `core/scheduler.py`
- `core/audit_logger.py`
- `core/privacy.py` (MAC hashing, opt-out)

---

## Phase 6: Testing & Validation (Weeks 11-12)

### 6.1 Unit Tests
**Coverage:**
- Event classification and priority
- BO acquisition function correctness
- Guardrail enforcement
- Rollback token lifecycle

### 6.2 Simulation Tests
**Using existing `sim.py` framework:**
```python
# Inject synthetic events
sim.inject_event(Event(type=DFS_RADAR, ap_id="AP01"))

# Run fast loop iteration
proposal = fast_loop.execute(ap_id="AP01")

# Verify rollback
assert rollback_manager.can_rollback(token)
```

### 6.3 Integration Tests
- Multi-loop coordination (event preempts fast loop)
- Blast radius isolation (neighbor APs unaffected)
- End-to-end: Event → Action → Rollback → Audit

**Files to Create:**
- `tests/test_event_loop.py`
- `tests/test_fast_loop.py`
- `tests/test_integration.py`

---

## Simplified Dependency Map

```
main.py
  ├── RRMScheduler (core/scheduler.py)
  │     ├── EventLoop (event_loop/event_loop.py)
  │     │     ├── EventQueue
  │     │     ├── ChannelSelector
  │     │     └── RollbackManager
  │     ├── FastLoop (fast_loop/fast_loop.py)
  │     │     ├── TelemetryAggregator
  │     │     ├── BayesianOptimizer (BO)
  │     │     ├── ChangePlanner
  │     │     └── GuardrailValidator
  │     └── SlowLoop (slow_loop/slow_loop.py)
  │           ├── SlowLoopEnv (Gym)
  │           └── CQLAgent
  ├── PolicyEngine (existing slo_catalog.py)
  └── AuditLogger (core/audit_logger.py)
```

---

## Minimal File Structure

```
multi-timescale-controller/
├── models/
│   ├── __init__.py
│   ├── events.py
│   ├── rollback.py
│   └── audit.py
├── event_loop/
│   ├── __init__.py
│   ├── event_loop.py
│   ├── channel_selector.py
│   └── rollback_manager.py
├── fast_loop/
│   ├── __init__.py
│   ├── fast_loop.py
│   ├── telemetry.py
│   ├── optimizer.py          # Bayesian Optimization
│   ├── change_planner.py
│   └── guardrails.py
├── slow_loop/
│   ├── __init__.py
│   ├── slow_loop.py
│   ├── environment.py        # Gym environment
│   └── cql_agent.py          # Safe RL agent
├── core/
│   ├── __init__.py
│   ├── scheduler.py          # Master orchestrator
│   ├── audit_logger.py
│   └── privacy.py
├── tests/
│   ├── test_event_loop.py
│   ├── test_fast_loop.py
│   └── test_integration.py
├── slo_catalog.py            # Existing policy engine
├── rrmengine.py              # To be refactored/integrated
└── main.py                   # Entry point
```

---

## Key Dependencies

```toml
# requirements.txt additions
scikit-optimize>=0.9.0     # Bayesian Optimization
gpytorch>=1.11             # Gaussian Processes (alternative)
gymnasium>=0.29            # RL environment
stable-baselines3>=2.0     # RL algorithms
protobuf>=4.0              # Telemetry serialization
influxdb-client>=1.36      # Time-series DB (optional)
cryptography>=41.0         # HMAC signing
pydantic>=2.0              # Data validation
```

---

## Implementation Priorities

### Must-Have (MVP)
1. ✅ Event Loop with DFS handling
2. ✅ Fast Loop with BO (single AP optimization)
3. ✅ Rollback mechanism (automatic + manual)
4. ✅ Audit logging with signatures
5. ✅ Integration with existing Policy Engine

### Should-Have (V1.0)
6. ⚠️ Slow Loop with Safe RL
7. ⚠️ Multi-AP coordination (blast radius)
8. ⚠️ Additional radio sensing integration
9. ⚠️ Compliance dashboard

### Nice-to-Have (V2.0)
10. 🔵 Advanced RL (RCPO, offline RL)
11. 🔵 Distributed controller (multi-site)
12. 🔵 Predictive maintenance
13. 🔵 Federated learning

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| BO exploration causes outages | HIGH | Conservative GP kernel, safety baseline, small initial steps |
| Event loop cascades | HIGH | Per-AP rate limiting, cooldown timers, blast radius checks |
| Regulatory violations | CRITICAL | Pre-check all actions, hardware failsafe, audit trail |
| Rollback failures | MEDIUM | Multi-level fallback (rollback → safe config → human intervention) |
| RL divergence | MEDIUM | Offline training first, human-in-loop for deployment, kill switch |

---

## Success Metrics

### Event Loop:
- DFS reaction latency <5 seconds (regulatory: <10s)
- False positive rate <5%
- Rollback success rate >95%

### Fast Loop:
- QoE improvement >10% over baseline
- Change budget utilization <80% (avoid thrashing)
- Prediction error <20% (BO accuracy)

### Slow Loop:
- Network-wide QoE improvement >15%
- Zero regulatory violations
- Constraint satisfaction rate 100%

---

## Next Steps

1. **Week 1**: Set up models and infrastructure
2. **Week 2-3**: Implement Event Loop + test with simulated DFS events
3. **Week 4-6**: Build Fast Loop BO pipeline, integrate with existing sim
4. **Week 7-9**: Prototype Slow Loop (start with graph coloring heuristic)
5. **Week 10**: Integration testing with all three loops
6. **Week 11-12**: Validation, tuning, documentation

---

## Notes

- **Start Simple**: Begin with Event Loop only, then add Fast Loop
- **Leverage Existing**: Reuse `PolicyEngine` from `slo_catalog.py`, metrics from `APMetricsManager`
- **Incremental Deployment**: Test each loop independently before orchestration
- **Privacy First**: Hash all identifiers, no PII in logs
- **Fail-Safe**: Every action must have rollback path

