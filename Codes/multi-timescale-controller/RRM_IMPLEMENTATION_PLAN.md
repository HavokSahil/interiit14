# RRMEngine Implementation Plan

## Overview

This plan outlines the implementation of a comprehensive **Radio Resource Management (RRM) Engine** for WiFi network optimization. The RRMEngine will use a multi-timescale control architecture to manage:
- Channel assignment
- Transmit power control
- AP load balancing
- Client association
- Network QoE optimization

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     RRMEngine                           │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ PolicyEngine │  │ ConfigEngine │  │   SLO        │ │
│  │              │  │              │  │  Catalog     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │         SlowLoopController                       │  │
│  │  (Channel Selection, Power Control)              │  │
│  │  Runs every N steps (e.g., every 100 steps)      │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │         FastLoopController                       │  │
│  │  (Client Steering, Load Balancing)               │  │
│  │  Runs every step                                 │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ┌────────────────┐      ┌───────────────────────┐    │
│  │  SensingAPI    │      │  ClientViewAPI        │    │
│  │  (Interferers) │      │  (QoE)                │    │
│  └────────────────┘      └───────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. PolicyEngine

**Purpose:** Defines optimization policies and objectives.

**Responsibilities:**
- Define optimization objectives (max throughput, fairness, QoE, etc.)
- Set constraints (power limits, channel restrictions, etc.)
- Evaluate policy compliance
- Provide decision guidance to controllers

**Key Classes:**

#### Policy Dataclass
```python
@dataclass
class OptimizationPolicy:
    name: str
    objective: str  # "max_throughput", "max_fairness", "max_qoe", etc.
    constraints: Dict[str, any]
    weights: Dict[str, float]  # Component weights for multi-objective
```

#### PolicyEngine Class
```python
class PolicyEngine:
    def __init__(self, policy: OptimizationPolicy)
    def evaluate_objective(self, state: NetworkState) -> float
    def check_constraints(self, config: NetworkConfig) -> bool
    def get_decision_weights(self) -> Dict[str, float]
```

**Optimization Objectives:**

1. **Max Throughput**: Maximize total network throughput
   ```python
   objective = sum(client.throughput_mbps for client in clients)
   ```

2. **Max Fairness**: Maximize Jain's fairness index
   ```python
   sum_tput = sum(c.throughput_mbps for c in clients)
   sum_sq = sum(c.throughput_mbps**2 for c in clients)
   fairness = (sum_tput**2) / (n * sum_sq)
   ```

3. **Max QoE**: Maximize average client QoE
   ```python
   objective = avg([client_qoe.qoe_ap for client_qoe in qoe_results])
   ```

4. **Min Interference**: Minimize network-wide interference
   ```python
   objective = -sum(ap.inc_energy for ap in aps)
   ```

**Constraints:**
- Power limits: `min_power ≤ tx_power ≤ max_power`
- Channel restrictions: `allowed_channels = [1, 6, 11]`
- QoE thresholds: `min_qoe ≥ threshold`
- Load limits: `clients_per_ap ≤ max_clients`

---

### 2. ConfigEngine

**Purpose:** Manages and applies network configurations to APs.

**Responsibilities:**
- Store current and proposed configurations
- Validate configuration changes
- Apply configurations to access points
- Track configuration history
- Rollback support

**Key Classes:**

#### APConfig Dataclass
```python
@dataclass
class APConfig:
    ap_id: int
    channel: int
    tx_power: float
    bandwidth: float
    obss_pd_threshold: float
    # Future: other 802.11 parameters
```

#### NetworkConfig Dataclass
```python
@dataclass
class NetworkConfig:
    timestamp: float
    ap_configs: Dict[int, APConfig]
    metadata: Dict[str, any]
```

#### ConfigEngine Class
```python
class ConfigEngine:
    def __init__(self, access_points: List[AccessPoint])
    
    def get_current_config(self) -> NetworkConfig
    def apply_config(self, config: NetworkConfig) -> bool
    def validate_config(self, config: NetworkConfig) -> bool
    def rollback(self) -> bool
    
    # Configuration builders
    def build_channel_config(self, ap_id: int, channel: int) -> APConfig
    def build_power_config(self, ap_id: int, power: float) -> APConfig
```

**Configuration Validation:**
- Channel in allowed list: [1, 6, 11]
- Power within bounds: [10, 30] dBm
- No conflicting settings
- Hardware capability checks

**Apply Process:**
1. Validate new configuration
2. Store current config as backup
3. Apply changes to AP objects
4. Log configuration change
5. Return success/failure

---

### 3. SlowLoopController

**Purpose:** Long-term optimization (channel selection, power control).

**Execution:** Periodic (e.g., every 100 simulation steps)

**Responsibilities:**
- Channel assignment optimization
- Transmit power optimization
- Long-term network planning
- Uses sensing data and QoE metrics

**Key Methods:**

```python
class SlowLoopController:
    def __init__(self, policy_engine, config_engine, sensing_api, client_view_api)
    
    def should_execute(self, step: int) -> bool
    def optimize_channels(self) -> NetworkConfig
    def optimize_power(self) -> NetworkConfig
    def execute(self, step: int) -> Optional[NetworkConfig]
```

**Channel Optimization Algorithm:**

1. **Analyze interference** using SensingAPI:
   - For each AP, identify major interferers on each channel
   - Calculate interference score per channel

2. **Evaluate channel options**:
   ```python
   for ap in aps:
       for channel in [1, 6, 11]:
           score = calculate_channel_score(ap, channel)
           # Score based on:
           # - Interference level
           # - Client QoE on that channel
           # - Neighboring AP channels (avoid same channel)
   ```

3. **Assign channels** using graph coloring or greedy algorithm:
   - Prioritize APs with highest interference
   - Select channel with best score
   - Update interference graph

4. **Validate and apply**:
   - Check policy constraints
   - Build NetworkConfig
   - Apply via ConfigEngine

**Power Optimization Algorithm:**

1. **Analyze coverage and interference**:
   - Too high power → more interference
   - Too low power → poor coverage

2. **Optimize per AP**:
   ```python
   for ap in aps:
       # Evaluate power levels [10, 15, 20, 25, 30] dBm
       for power in power_levels:
           # Calculate:
           # - Coverage area
           # - Client SINR
           # - Interference to neighbors
           score = evaluate_power(ap, power)
   ```

3. **Apply** power settings that maximize policy objective

---

### 4. FastLoopController

**Purpose:** Real-time optimization (client steering, load balancing).

**Execution:** Every simulation step

**Responsibilities:**
- Client association optimization
- Load balancing across APs
- Fast response to QoE degradation
- Client steering decisions

**Key Methods:**

```python
class FastLoopController:
    def __init__(self, policy_engine, config_engine, client_view_api)
    
    def should_steer_client(self, client: Client, qoe_result: ClientQoEResult) -> bool
    def find_best_ap(self, client: Client) -> int
    def execute(self) -> List[Tuple[int, int, int]]  # (client_id, old_ap, new_ap)
```

**Client Steering Algorithm:**

1. **Identify candidates** for steering:
   ```python
   for client in clients:
       qoe = clientview_api.compute_client_qoe(client)
       
       # Steer if:
       # - QoE below threshold (e.g., < 0.5)
       # - AP overloaded (too many clients)
       # - Better AP available (RSSI + QoE)
       
       if should_steer(client, qoe):
           candidates.append(client)
   ```

2. **Find best target AP**:
   ```python
   for candidate in candidates:
       best_ap = None
       best_score = -inf
       
       for ap in aps:
           # Calculate score based on:
           # - RSSI to AP
           # - AP current load
           # - Expected QoE on that AP
           score = calculate_ap_score(candidate, ap)
           
           if score > best_score:
               best_score = score
               best_ap = ap
   ```

3. **Apply steering**:
   - Force client reassociation
   - Update connection tracking
   - Log steering decision

**Load Balancing:**
- Monitor AP utilization (airtime, client count)
- Move clients from overloaded APs to underutilized APs
- Use hysteresis to avoid ping-pong

---

### 5. SLO Catalog (Role-Based)

**Purpose:** Role-based Service Level Objective definitions with YAML configuration support.

**Key Concept:** Each role (e.g., ExamHallStrict, VO, VI, BE) defines a complete policy profile including:
- QoS component weights
- Enforcement thresholds
- Regulatory constraints
- Long-term KPIs

**YAML Structure:**

```yaml
version: 1.1
description: "Hybrid SLO catalog for Policy, SLOs & Compliance"

global:
  percentiles:
    tail: "P95"        # Tail-sensitive metrics
    congestion: "P50"  # Congestion/typical metrics
  
  normalizers:
    RSSI_min: -95
    RSSI_max: -30
    SNR_min: 0
    SNR_max: 50
    PER_max_pct: 20
    Retry_max_pct: 30
  
  defaults:
    monitoring_window_seconds: 300
    penalty_clip_max: 1.0
    warning_threshold: 0.03
    critical_threshold: 0.10

roles:
  ExamHallStrict:
    display_name: "Exam Hall (Strict)"
    purpose: "High-density exam mode. Enforce strict fairness and stability."
    qos_weights: { ws: 0.10, wt: 0.15, wr: 0.30, wl: 0.30, wa: 0.15 }
    enforcement:
      RSSI_dBm: { operator: ">=", value: -66, authority: "P95", action: "IncreaseTxPower/Steer" }
      SNR_dB: { operator: ">=", value: 20, authority: "P95", action: "Notify/Steer" }
      PER_pct: { operator: "<=", value: 3.0, authority: "P95", action: "Penalty/Throttle" }
      Retry_pct: { operator: "<=", value: 3.0, authority: "P95", action: "Penalty/Throttle" }
      Airtime_util: { operator: "<=", value: 60, authority: "P50", action: "EnforceAirtimeFairness" }
      CCA_busy: { operator: "<=", value: 60, authority: "P50", action: "Steer/ChannelChange" }
    regulatory:
      max_channel_width_MHz: 20
    long_term_kpis:
      median_throughput_lift_pct: 20
      post_steer_sinr_gain_db: 3

  VO:
    display_name: "Voice (VO)"
    purpose: "Real-time voice; prioritize low latency & reliability"
    qos_weights: { ws: 0.25, wt: 0.10, wr: 0.35, wl: 0.25, wa: 0.05 }
    enforcement:
      RSSI_dBm: { operator: ">=", value: -67, authority: "P95", action: "IncreaseTxPower" }
      SNR_dB: { operator: ">=", value: 25, authority: "P95", action: "Notify" }
      PER_pct: { operator: "<=", value: 2.0, authority: "P95", action: "Penalty" }
      Retry_pct: { operator: "<=", value: 5.0, authority: "P95", action: "Penalty" }
      Airtime_util: { operator: "<=", value: 70, authority: "P50", action: "Steer" }
      CCA_busy: { operator: "<=", value: 70, authority: "P50", action: "Steer" }
    regulatory:
      max_channel_width_MHz: 20
    long_term_kpis:
      median_throughput_lift_pct: 15
      post_steer_sinr_gain_db: 3

  VI:
    display_name: "Video (VI)"
    purpose: "Video streaming; prioritize throughput"
    qos_weights: { ws: 0.20, wt: 0.40, wr: 0.20, wl: 0.15, wa: 0.05 }
    
  BE:
    display_name: "Best Effort (BE)"
    purpose: "General user traffic"
    qos_weights: { ws: 0.15, wt: 0.35, wr: 0.25, wl: 0.15, wa: 0.10 }
  
  # ... additional roles: BK, Guest, IoT
```

**Python Data Structures:**

```python
@dataclass
class EnforcementRule:
    operator: str  # ">=", "<=", "==", etc.
    value: float
    authority: str  # "P95", "P50", "mean"
    action: str  # "IncreaseTxPower", "Steer", "Penalty", etc.

@dataclass
class RoleConfig:
    role_id: str
    display_name: str
    purpose: str
    qos_weights: Dict[str, float]  # ws, wt, wr, wl, wa
    enforcement: Dict[str, EnforcementRule]
    regulatory: Dict[str, any]
    long_term_kpis: Dict[str, float]

@dataclass
class SLOCatalogConfig:
    version: str
    description: str
    global_config: Dict[str, any]
    roles: Dict[str, RoleConfig]

class SLOCatalog:
    def __init__(self, yaml_path: str):
        self.config = self._load_yaml(yaml_path)
        self.roles = self.config.roles
        self.global_config = self.config.global_config
    
    def _load_yaml(self, path: str) -> SLOCatalogConfig:
        \"\"\"Load and parse YAML SLO catalog.\"\"\"
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return self._parse_config(data)
    
    def get_role(self, role_id: str) -> Optional[RoleConfig]:
        \"\"\"Get role configuration by ID.\"\"\"
        return self.roles.get(role_id)
    
    def get_qos_weights(self, role_id: str) -> Dict[str, float]:
        \"\"\"Get QoS weights for a role.\"\"\"
        role = self.get_role(role_id)
        return role.qos_weights if role else {}
    
    def evaluate_enforcement(self, role_id: str, metrics: Dict[str, float]) -> List[str]:
        \"\"\"Evaluate enforcement rules and return triggered actions.\"\"\"
        role = self.get_role(role_id)
        if not role:
            return []
        
        triggered_actions = []
        for metric_name, rule in role.enforcement.items():
            if metric_name not in metrics:
                continue
            
            metric_value = metrics[metric_name]
            
            # Evaluate operator
            if rule.operator == ">=" and metric_value < rule.value:
                triggered_actions.append(rule.action)
            elif rule.operator == "<=" and metric_value > rule.value:
                triggered_actions.append(rule.action)
        
        return triggered_actions
    
    def check_regulatory_compliance(self, role_id: str, config: Dict[str, any]) -> bool:
        \"\"\"Check if configuration meets regulatory constraints.\"\"\"
        role = self.get_role(role_id)
        if not role:
            return True
        
        # Check channel width
        if 'channel_width_MHz' in config:
            max_width = role.regulatory.get('max_channel_width_MHz', float('inf'))
            if config['channel_width_MHz'] > max_width:
                return False
        
        return True
```

**Integration with PolicyEngine:**

```python
class PolicyEngine:
    def __init__(self, slo_catalog: SLOCatalog, default_role: str = "BE"):
        self.slo_catalog = slo_catalog
        self.default_role = default_role
        self.client_roles = {}  # Map client_id to role_id
    
    def set_client_role(self, client_id: int, role_id: str):
        \"\"\"Assign a role to a client.\"\"\"
        self.client_roles[client_id] = role_id
    
    def get_client_qos_weights(self, client_id: int) -> Dict[str, float]:
        \"\"\"Get QoS weights for a client based on their role.\"\"\"
        role_id = self.client_roles.get(client_id, self.default_role)
        return self.slo_catalog.get_qos_weights(role_id)
    
    def evaluate_client_compliance(self, client_id: int, qoe_result: ClientQoEResult) -> List[str]:
        \"\"\"Evaluate if client meets SLO and return actions.\"\"\"
        role_id = self.client_roles.get(client_id, self.default_role)
        
        # Convert QoE components to metrics
        metrics = {
            'RSSI_dBm': qoe_result.rssi_dbm,  # Need to add to ClientQoEResult
            'PER_pct': qoe_result.per_pct,    # Need to add
            'Retry_pct': qoe_result.retry_pct,
            # ... other metrics
        }
        
        return self.slo_catalog.evaluate_enforcement(role_id, metrics)
```

**Usage Example:**

```python
# Initialize SLO Catalog from YAML
slo_catalog = SLOCatalog("slo_catalog.yml")

# Create policy engine with catalog
policy_engine = PolicyEngine(slo_catalog, default_role="BE")

# Assign roles to clients
policy_engine.set_client_role(client_id=1, role_id="ExamHallStrict")
policy_engine.set_client_role(client_id=2, role_id="VO")

# Get QoS weights for optimization
weights = policy_engine.get_client_qos_weights(client_id=1)
# Returns: {'ws': 0.10, 'wt': 0.15, 'wr': 0.30, 'wl': 0.30, 'wa': 0.15}

# Evaluate compliance and get actions
actions = policy_engine.evaluate_client_compliance(client_id, qoe_result)
# Returns: ['IncreaseTxPower', 'Steer'] if thresholds violated
```

**File Location:** `slo_catalog.yml` in project root

---

## Integration Architecture

### Data Flow

```
┌──────────────┐
│ Simulation   │
│   Step       │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│  RRMEngine.execute(step)             │
├──────────────────────────────────────┤
│  1. Update sensing data              │ ─┐
│  2. Update client view (QoE)         │  │ Input Collection
│  3. Evaluate SLOs                    │ ─┘
├──────────────────────────────────────┤
│  4. FastLoopController.execute()     │ ─┐
│     - Check client QoE               │  │ Fast Loop
│     - Perform steering               │  │ (Every Step)
│  5. Return steering actions          │ ─┘
├──────────────────────────────────────┤
│  6. If slow_loop_period:             │ ─┐
│     - SlowLoopController.execute()   │  │ Slow Loop
│       * Optimize channels            │  │ (Periodic)
│       * Optimize power               │  │
│     - Apply config via ConfigEngine  │ ─┘
└──────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│  Apply       │
│  Changes     │
└──────────────┘
```

---

## Proposed File Structure

### [MODIFY] [rrmengine.py](file:///home/rishit/Documents/interiit14/Codes/multi-timescale-controller/rrmengine.py)

Update with complete implementation:

```python
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datatype import AccessPoint, Client
from sensing import SensingAPI
from clientview import ClientViewAPI

# Import components
from policy_engine import PolicyEngine, OptimizationPolicy
from config_engine import ConfigEngine, NetworkConfig
from slow_loop_controller import SlowLoopController
from fast_loop_controller import FastLoopController
from slo_catalog import SLOCatalog, SLO

class RRMEngine:
    def __init__(self,
                 access_points: List[AccessPoint],
                 clients: List[Client],
                 prop_model,
                 policy: OptimizationPolicy,
                 slow_loop_period: int = 100):
        # Core components
        self.aps = {ap.id: ap for ap in access_points}
        self.stas = {client.id: client for client in clients}
        
        # APIs
        self.sensing_api = SensingAPI(access_points, [], prop_model)
        self.client_view_api = ClientViewAPI(access_points, clients)
        
        # Engines
        self.policy_engine = PolicyEngine(policy)
        self.config_engine = ConfigEngine(access_points)
        self.slo_catalog = SLOCatalog([])
        
        # Controllers
        self.slow_loop_engine = SlowLoopController(
            self.policy_engine,
            self.config_engine,
            self.sensing_api,
            self.client_view_api
        )
        self.fast_loop_engine = FastLoopController(
            self.policy_engine,
            self.config_engine,
            self.client_view_api
        )
        
        # Configuration
        self.rrm_enabled = True
        self.slow_loop_period = slow_loop_period
        self.current_step = 0
    
    def execute(self, step: int) -> Dict[str, any]:
        \"\"\"Main RRM execution loop.\"\"\"
        if not self.rrm_enabled:
            return {}
        
        self.current_step = step
        results = {}
        
        # Update APIs
        self.sensing_api.compute_sensing_results()
        qoe_views = self.client_view_api.compute_all_views()
        
        # Fast loop (every step)
        steering_actions = self.fast_loop_engine.execute()
        results['steering'] = steering_actions
        
        # Slow loop (periodic)
        if self.slow_loop_engine.should_execute(step):
            config = self.slow_loop_engine.execute(step)
            if config:
                self.config_engine.apply_config(config)
                results['config_update'] = config
        
        return results
```

---

### [NEW] policy_engine.py

PolicyEngine and policy definitions.

---

### [NEW] config_engine.py

ConfigEngine and configuration management.

---

### [NEW] slow_loop_controller.py

SlowLoopController with channel and power optimization.

---

### [NEW] fast_loop_controller.py

FastLoopController with client steering and load balancing.

---

### [NEW] slo_catalog.py

SLO definitions and evaluation.

---

## Dependencies

### Python Packages

Add to `requirements.txt`:
```
pyyaml>=6.0  # For SLO catalog YAML parsing
```

Install with:
```bash
pip install pyyaml
```

---

## Implementation Phases

### Phase 0: SLO Catalog Setup
- [ ] Create `slo_catalog.yml` with role definitions
- [ ] Implement `SLOCatalog` class with YAML parsing
- [ ] Implement role-based enforcement evaluation
- [ ] Unit tests for catalog loading and evaluation

### Phase 1: Core Infrastructure
- [ ] Implement PolicyEngine with SLO catalog integration
- [ ] Implement ConfigEngine with validation
- [ ] Integrate with RRMEngine
- [ ] Add client role assignment capability

### Phase 2: Slow Loop
- [ ] Implement SlowLoopController
- [ ] Channel optimization algorithm
- [ ] Power optimization algorithm
- [ ] Use role-specific regulatory constraints

### Phase 3: Fast Loop
- [ ] Implement FastLoopController
- [ ] Client steering logic using role-based thresholds
- [ ] Load balancing with role priorities

### Phase 4: Integration & Testing
- [ ] Integrate all components
- [ ] End-to-end testing with multiple roles
- [ ] Performance tuning
- [ ] Validate against SLO compliance

---

## Testing Strategy

### Unit Tests
- Test each component independently
- Test policy evaluation
- Test configuration validation
- Test optimization algorithms

### Integration Tests
- Test RRMEngine execution flow
- Test component interactions
- Test with realistic scenarios

### Simulation Tests
- Long-running simulations
- Various traffic patterns
- Interference scenarios
- Client mobility

---

## Next Steps

1. Review and approve this plan
2. Prioritize components to implement first
3. Define specific optimization algorithms
4. Begin implementation phase by phase
