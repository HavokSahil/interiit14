# Fast Loop Controller - Correct Implementation Plan

## Objective
Create a Fast Loop Controller that uses the **real-time interference graph** to optimize:
1. **Channel** assignment
2. **Bandwidth** (20/40/80 MHz)
3. **OBSS-PD threshold** (spatial reuse)

The controller runs every **10 minutes** and makes reactive decisions based on current interference patterns.

---

## What the Fast Loop ACTUALLY Does

### Purpose
**Medium-term reactive optimization** based on observed interference patterns in the network graph.

### Inputs
- **Interference Graph** (`nx.DiGraph`) - computed in real-time
  - Nodes: APs with channel, load, clients
  - Edges: Interference relationships with weights (0-1)
- **AP Metrics** (from metrics manager)
  - CCA busy percentage
  - Retry rate
  - Throughput
- **Client QoE** (optional for validation)

### Actions
1. **Channel Change** - Move to less congested channel
2. **Bandwidth Adjustment** - Reduce (40→20) to avoid interference or increase (20→40) if clear
3. **OBSS-PD Tuning** - Adjust spatial reuse threshold based on interference density

---

## Algorithm Design

### Phase 1: Analyze Interference Graph

For each AP, compute:
```python
def analyze_interference(ap_id, graph):
    # Get all interfering neighbors
    interferers = graph.predecessors(ap_id)
    
    # Calculate interference metrics
    total_interference = sum(
        graph[i][ap_id]['weight'] 
        for i in interferers
    )
    
    # Count interferers per channel
    channel_interference = {}
    for i in interferers:
        ch = graph.nodes[i]['channel']
        weight = graph[i][ap_id]['weight']
        channel_interference[ch] = channel_interference.get(ch, 0) + weight
    
    return {
        'total_interference': total_interference,
        'num_interferers': len(list(interferers)),
        'channel_interference': channel_interference,
        'worst_interference_channel': max(channel_interference, key=channel_interference.get)
    }
```

### Phase 2: Decide Action

#### 1. **Channel Change Decision**
```python
# Threshold: High interference
if total_interference > 0.6 or num_interferers > 3:
    # Find best channel
    candidate_channels = [1, 6, 11] if current_band == '2.4GHz' else [36, 40, 44, 48, ...]
    
    # Evaluate each channel
    best_channel = min(candidate_channels, key=lambda ch: 
        estimate_interference_on_channel(ch, graph, ap_id)
    )
    
    # Change if significantly better
    if interference_on(best_channel) < interference_on(current) * 0.7:
        ACTION: change_channel(ap_id, best_channel)
```

#### 2. **Bandwidth Adjustment**
```python
# Reduce bandwidth if:
# - High interference AND high retry rate
if total_interference > 0.5 and retry_rate > 15%:
    if current_bandwidth == 40:
        ACTION: set_bandwidth(ap_id, 20)  # Reduce to avoid more overlap
    
# Increase bandwidth if:
# - Low interference AND low retry rate AND low CCA busy
if total_interference < 0.2 and retry_rate < 5% and cca_busy < 30%:
    if current_bandwidth == 20:
        ACTION: set_bandwidth(ap_id, 40)  # Take advantage of clean spectrum
```

#### 3. **OBSS-PD Adjustment**
```python
# Increase OBSS-PD (more aggressive spatial reuse) if:
# - Moderate interference but from distant APs
# - CCA busy is high but actual retry rate is low
if cca_busy > 60% and retry_rate < 10%:
    new_obss_pd = min(current_obss_pd + 3, -62)  # Max -62 dBm
    ACTION: set_obss_pd(ap_id, new_obss_pd)

# Decrease OBSS-PD (more conservative) if:
# - High retry rate indicates real collisions
if retry_rate > 20%:
    new_obss_pd = max(current_obss_pd - 3, -82)  # Min -82 dBm
    ACTION: set_obss_pd(ap_id, new_obss_pd)
```

---

## Implementation Structure

### FastLoopController Class

```python
class FastLoopController:
    """
    Fast Loop: Reactive optimization using interference graph.
    
    Runs every 10 minutes to adjust channel, bandwidth, and OBSS-PD
    based on observed interference patterns.
    """
    
    def __init__(self,
                 config_engine: ConfigEngine,
                 policy_engine: PolicyEngine,
                 access_points: List[AccessPoint]):
        self.config_engine = config_engine
        self.policy_engine = policy_engine
        self.aps = {ap.id: ap for ap in access_points}
        
        # Thresholds
        self.interference_threshold = 0.6
        self.cca_busy_threshold = 0.6
        self.retry_rate_threshold = 15.0
        
        # Statistics
        self.stats = {
            'channel_changes': 0,
            'bandwidth_changes': 0,
            'obss_pd_changes': 0
        }
    
    def execute(self, interference_graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        Execute Fast Loop optimization.
        
        Args:
            interference_graph: Real-time interference graph
            
        Returns:
            List of configuration actions
        """
        actions = []
        
        for ap_id in self.aps.keys():
            # Analyze interference for this AP
            analysis = self._analyze_interference(ap_id, interference_graph)
            
            # Decide action based on analysis
            action = self._decide_action(ap_id, analysis)
            
            if action:
                # Apply action
                result = self._apply_action(ap_id, action)
                if result['success']:
                    actions.append(result)
        
        return actions
    
    def _analyze_interference(self, ap_id, graph):
        """Analyze interference graph for this AP"""
        # Implementation from Phase 1
        pass
    
    def _decide_action(self, ap_id, analysis):
        """Decide what action to take"""
        # Priority 1: Channel change (if very high interference)
        if analysis['total_interference'] > 0.7:
            return {'type': 'channel_change', ...}
        
        # Priority 2: Bandwidth adjustment
        elif analysis['total_interference'] > 0.5:
            return {'type': 'bandwidth_adjust', ...}
        
        # Priority 3: OBSS-PD tuning
        elif analysis['cca_busy'] > 0.6:
            return {'type': 'obss_pd_adjust', ...}
        
        return None
    
    def _apply_action(self, ap_id, action):
        """Apply configuration change"""
        pass
```

---

## Data Flow

```
┌─────────────────────────────────────────────────┐
│  Simulation / RRM Engine                        │
│                                                  │
│  Every Step:                                     │
│    - Update metrics                             │
│    - Build interference graph (real-time)       │
│                                                  │
│  Every 60 Steps (10 minutes):                   │
│    ┌──────────────────────────────────┐        │
│    │  FAST LOOP TRIGGERED             │        │
│    │                                  │        │
│    │  Input: interference_graph       │        │
│    │  ↓                               │        │
│    │  For each AP:                    │        │
│    │    - Analyze interference        │        │
│    │    - Check thresholds            │        │
│    │    - Decide action               │        │
│    │    - Apply config change         │        │
│    │  ↓                               │        │
│    │  Output: config_actions[]        │        │
│    └──────────────────────────────────┘        │
│                                                  │
│  Update:                                         │
│    - AP.channel                                 │
│    - AP.bandwidth                               │
│    - AP.obss_pd_threshold                      │
└─────────────────────────────────────────────────┘
```

---

## Integration with Enhanced RRM Engine

### Modified execute() method:

```python
def execute(self, step: int) -> Dict[str, Any]:
    # ... Event Loop, Cooldown, Slow Loop ...
    
    # ========== FAST LOOP (Every 10 minutes) ==========
    if self.fast_loop_engine and (step - self.last_fast_loop_step >= self.fast_loop_period):
        
        # Get current interference graph (already computed by sim)
        interference_graph = self.get_interference_graph()
        
        # Execute Fast Loop with graph
        actions = self.fast_loop_engine.execute(interference_graph)
        
        if actions:
            # Apply each action
            for action in actions:
                if action['type'] == 'channel_change':
                    config = self.config_engine.build_channel_config(
                        action['ap_id'], action['new_channel']
                    )
                elif action['type'] == 'bandwidth_adjust':
                    config = self.config_engine.build_bandwidth_config(
                        action['ap_id'], action['new_bandwidth']
                    )
                elif action['type'] == 'obss_pd_adjust':
                    # Direct AP update (no ConfigEngine method yet)
                    ap = self.aps[action['ap_id']]
                    ap.obss_pd_threshold = action['new_obss_pd']
                
                self.config_engine.apply_config(config)
            
            results['fast_loop_actions'] = actions
        
        self.last_fast_loop_step = step
    
    return results
```

---

## Example Scenario

### Initial State
```
AP0: Channel 1, BW=20, OBSS-PD=-82
  - Interferers from graph: AP1(weight=0.7), AP2(weight=0.4)
  - Total interference: 1.1 (HIGH)
  - Retry rate: 18%
  - CCA busy: 75%
```

### Fast Loop Analysis
```python
analysis = {
    'total_interference': 1.1,
    'num_interferers': 2,
    'channel_interference': {1: 1.1, 6: 0.2, 11: 0.3},
    'worst_channel': 1
}
```

### Decision
```python
# High interference on channel 1
# Channel 6 has least interference (0.2)
# ACTION: Change to channel 6
```

### Result
```
AP0: Channel 6, BW=20, OBSS-PD=-82
  - New interference: 0.2 (LOW)
  - Expected: Retry rate ↓, CCA busy ↓
```

---

## Next Steps

1. **Create new `fast_loop_controller.py`** with interference-based logic
2. **Update `enhanced_rrm_engine.py`** to pass interference graph
3. **Add helper methods** to estimate interference on different channels
4. **Test with interference scenarios**
5. **Update documentation and diagrams**

---

## Questions to Resolve

1. Should Fast Loop have access to the full simulation to call `get_interference_graph()`?
2. Or should the graph be passed as a parameter from Enhanced RRM Engine?
3. What channels are available for 5GHz? (36, 40, 44, 48, 52, 56, 60, 64, ...)
4. Should bandwidth changes be gradual (40→20) or can we jump (20→80)?

---

**Status:** Implementation Plan - Ready for Review
