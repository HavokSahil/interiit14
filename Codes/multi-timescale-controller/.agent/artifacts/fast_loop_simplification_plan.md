# Fast Loop Simplification Implementation Plan

## Objective
Simplify the Fast Loop algorithm and implement 10-minute periodic execution for better clarity and maintainability.

---

## Current Issues

### 1. **Algorithm Complexity**
- Multiple nested conditions for triggers (weak_fraction, cca_busy, retry_rate)
- Complex EWMA baseline tracking with adaptive tolerances
- Scheduled evaluation with threading timers
- Obscure decision logic for TX power adjustments
- Penalty and cooldown management spread across multiple functions

### 2. **Execution Model**
- Currently runs every step (too frequent)
- No built-in periodicity control
- Evaluation happens asynchronously via timers (hard to debug)

---

## Proposed Changes

### Phase 1: Add Periodic Execution (10 Minutes)

**Files to Modify:**
- `enhanced_rrm_engine.py`

**Changes:**
1. Add `fast_loop_period` parameter to `__init__` (default: 60 steps = 10 min)
2. Add `last_fast_loop_step` tracking variable
3. Modify `execute()` to check if enough steps have passed:
   ```python
   # Fast Loop - runs every 10 minutes
   if step - self.last_fast_loop_step >= self.fast_loop_period:
       fast_loop_actions = self.fast_loop_engine.execute()
       self.last_fast_loop_step = step
   ```

**Benefits:**
- More realistic execution interval
- Reduces overhead
- Easier to reason about behavior

---

### Phase 2: Simplify Fast Loop Algorithm

**Files to Modify:**
- `fast_loop_refactored.py`

**Current Complex Elements to Remove/Simplify:**

#### A. **Remove Asynchronous Evaluation**
- **Current:** Uses `schedule_timer()` with threading for delayed evaluation
- **Proposed:** Synchronous evaluation - check metrics immediately or on next execution
- **Why:** Simpler to debug, no threading issues, clearer flow

#### B. **Simplify Baseline Tracking**
- **Current:** Complex EWMA with mean + variance for adaptive tolerance
- **Proposed:** Simple moving average or just use current metrics vs thresholds
- **Why:** EWMA adds complexity without clear benefit in 10-min intervals

#### C. **Simplify TX Power Logic**
- **Current:**
  ```python
  # Complex conditions
  decrease_cond = (cca_busy > 0.5) or (retry_rate > 10.0)
  weak_fraction = calculate_weak_clients()
  increase_cond = (weak_fraction >= 0.15)
  
  # Complex step calculation
  d = compute_distance_from_baseline(...)
  proposed_step = min(max_step, base_step * (1 + 0.2 * d))
  ```
  
- **Proposed:**
  ```python
  # Clear, simple thresholds
  if cca_busy > 0.7 or retry_rate > 15.0:
      # Reduce power by 2 dB
      action = decrease_power(ap_id, step=2.0)
  elif median_rssi < -75 or weak_clients_pct > 0.3:
      # Increase power by 2 dB
      action = increase_power(ap_id, step=2.0)
  ```

#### D. **Simplify Rollback Mechanism**
- **Current:** Automatic rollback with scheduled evaluation, complex metric comparison
- **Proposed:** 
  - Track last action per AP
  - On next Fast Loop execution (10 min later), compare metrics
  - If degraded, revert to previous config
- **Why:** Simpler, synchronous, transparent

#### E. **Unified Action Structure**
- **Current:** Separate methods for `tx_power_refine`, `qoe_rapid_correction`, etc.
- **Proposed:** Single `optimize_ap(ap_id)` method with clear prioritized checks
  ```python
  def optimize_ap(self, ap_id: int) -> Dict[str, Any]:
      # 1. Check if in cooldown
      # 2. Get current metrics
      # 3. Check if previous action needs rollback
      # 4. Decide new action based on simple thresholds
      # 5. Apply and record action
  ```

---

### Phase 3: Update Documentation

**Files to Update:**
- `FAST_LOOP_REFACTORED_README.md`
- `GENERATE_3DAY_FASTLOOP_README.md`
- Add inline comments explaining simplified logic

---

## Detailed Implementation Steps

### Step 1: Update Enhanced RRM Engine (10-min periodicity)

**File:** `enhanced_rrm_engine.py`

1. **Add parameter:**
   ```python
   def __init__(self, ..., fast_loop_period: int = 60, ...):
   ```

2. **Initialize tracking:**
   ```python
   self.fast_loop_period = fast_loop_period
   self.last_fast_loop_step = 0
   ```

3. **Modify execute():**
   ```python
   # Fast Loop (Optimization) - Priority 5, runs every 10 min
   fast_loop_actions = []
   if self.fast_loop_engine and (step - self.last_fast_loop_step >= self.fast_loop_period):
       fast_loop_actions = self.fast_loop_engine.execute()
       self.last_fast_loop_step = step
   ```

**Test:** Verify Fast Loop only executes every 60 steps

---

### Step 2: Simplify Fast Loop Core Algorithm

**File:** `fast_loop_refactored.py`

#### 2.1 Remove Complex Utilities
**Delete/Simplify:**
- `schedule_timer()` - no longer needed
- `fast_alpha_update_after_success()` - no longer needed
- `compute_distance_from_baseline()` - no longer needed
- `compute_adaptive_tolerance()` - no longer needed
- Most EWMA helper functions

**Keep:**
- `clamp()` - useful utility
- `pct_drop()` - useful for comparison
- Basic state management

#### 2.2 Simplify State Store
**Current:**
```python
self.state_store: Dict[str, Any] = {
    "penalties": {},
    "last_actions": {},
    "retry_state": {},
    "planned_retries": {},
    "_audit_log": deque(maxlen=10000)
}
```

**Proposed:**
```python
self.state_store: Dict[str, Any] = {
    "last_actions": {},      # AP -> {action, timestamp, config}
    "cooldowns": {},         # AP -> next_allowed_step
    "_audit_log": deque(maxlen=1000)
}
```

#### 2.3 Simplified Main Method

**New `execute()` structure:**
```python
def execute(self) -> List[Dict[str, Any]]:
    """Execute Fast Loop for all APs (called every 10 minutes)"""
    results = []
    
    for ap_id in self.aps.keys():
        result = self.optimize_ap(ap_id)
        if result['status'] in ['acted', 'rolled_back']:
            results.append({
                'ap_id': ap_id,
                'action': result['action_type'],
                'result': result
            })
    
    return results
```

#### 2.4 Simplified `optimize_ap()` Method

```python
def optimize_ap(self, ap_id: int) -> Dict[str, Any]:
    """
    Optimize a single AP with simple, clear logic.
    
    Steps:
    1. Check cooldown
    2. Check if previous action needs rollback
    3. Evaluate current conditions
    4. Take action if needed
    """
    
    # 1. Check cooldown (simple time-based)
    if self._is_in_cooldown(ap_id):
        return {'status': 'skipped', 'reason': 'cooldown'}
    
    # 2. Check rollback of previous action
    rollback_result = self._check_rollback(ap_id)
    if rollback_result:
        return rollback_result
    
    # 3. Get current metrics
    ap = self.get_ap(ap_id)
    metrics = self._get_simple_metrics(ap)
    
    # 4. Decide action based on SIMPLE thresholds
    action = self._decide_action(ap, metrics)
    
    if action['type'] == 'none':
        return {'status': 'no_action_needed'}
    
    # 5. Apply action
    return self._apply_action(ap_id, action)
```

#### 2.5 Simple Threshold-Based Decision

```python
def _decide_action(self, ap, metrics):
    """Simple threshold-based decision - easy to understand and tune"""
    
    # High contention? Reduce power
    if metrics['cca_busy'] > 0.70 or metrics['retry_rate'] > 15.0:
        return {
            'type': 'decrease_power',
            'step_db': 2.0,
            'reason': 'High contention detected'
        }
    
    # Poor coverage? Increase power
    if metrics['median_rssi'] < -75.0 or metrics['weak_clients_pct'] > 0.30:
        return {
            'type': 'increase_power',
            'step_db': 2.0,
            'reason': 'Poor coverage detected'
        }
    
    return {'type': 'none'}
```

#### 2.6 Simplified Rollback Logic

```python
def _check_rollback(self, ap_id):
    """Check if previous action needs rollback (synchronous)"""
    
    last_action = self.state_store['last_actions'].get(ap_id)
    if not last_action:
        return None
    
    # Get current metrics
    current = self._get_simple_metrics(self.get_ap(ap_id))
    
    # Simple comparison: did key metrics degrade?
    baseline = last_action.get('baseline_metrics', {})
    
    degraded = (
        current['throughput'] < baseline.get('throughput', 0) * 0.85 or
        current['retry_rate'] > baseline.get('retry_rate', 100) * 1.30
    )
    
    if degraded:
        # Rollback to previous config
        self._rollback_action(ap_id, last_action)
        return {
            'status': 'rolled_back',
            'reason': 'Metrics degraded',
            'action_type': 'rollback'
        }
    
    return None
```

---

## Testing Plan

### Test 1: Periodic Execution
**File:** Create `test_fast_loop_periodic.py`
```python
# Verify Fast Loop only runs every 60 steps
for step in range(200):
    result = rrm.execute(step)
    if 'fast_loop' in result and result['fast_loop']:
        print(f"Fast Loop ran at step {step}")
# Expected: Runs at steps 60, 120, 180
```

### Test 2: Simplified Logic
**File:** `test_fast_loop_simplified.py`
```python
# Test simple threshold-based decisions
# - High CCA -> decrease power
# - Low RSSI -> increase power
# - Verify rollback on degradation
```

### Test 3: Integration
**File:** Run `test_3day_fastloop.py`
```python
# Verify 3-day simulation still works
# Check that logs show actions every 10 min
```

---

## Migration Path

### Phase 1: Minimal Changes (Periodicity only)
- Add 10-minute periodicity to `enhanced_rrm_engine.py`
- Keep existing algorithm
- Test to ensure behavior is correct

### Phase 2: Algorithm Simplification
- Implement simplified `optimize_ap()` method
- Remove complex EWMA and adaptive tolerance
- Remove async evaluation (threading)
- Simplify rollback to synchronous check

### Phase 3: Cleanup
- Remove unused helpers
- Update documentation
- Add clear inline comments

---

## Expected Benefits

1. **Clarity:** Code is easier to read and understand
2. **Debuggability:** No threading, synchronous flow
3. **Maintainability:** Clear thresholds, easy to tune
4. **Performance:** Less overhead, runs less frequently
5. **Audit Trail:** Still maintains full audit logging

---

## Risks & Mitigation

### Risk 1: Less Responsive
- **Concern:** 10-minute intervals might miss rapid changes
- **Mitigation:** Event Loop handles critical/rapid events (DFS, interference)

### Risk 2: Loss of Adaptive Behavior
- **Concern:** Removing EWMA might reduce adaptability
- **Mitigation:** Simple thresholds can be tuned per deployment, clear and predictable

### Risk 3: Breaking Existing Tests
- **Concern:** Tests expect old behavior
- **Mitigation:** Update tests incrementally, keep integration tests passing

---

## File Modification Summary

| File | Changes | Complexity |
|------|---------|------------|
| `enhanced_rrm_engine.py` | Add periodicity control | Low |
| `fast_loop_refactored.py` | Simplify algorithm, remove EWMA/timers | Medium |
| `FAST_LOOP_REFACTORED_README.md` | Update documentation | Low |
| `test_fast_loop_integration.py` | Update tests | Low |
| `generate_3day_fastloop_logs.py` | Update for new behavior | Low |

---

## Timeline Estimate

- **Phase 1 (Periodicity):** 30 minutes
- **Phase 2 (Simplification):** 2-3 hours
- **Phase 3 (Cleanup & Docs):** 1 hour
- **Total:** ~4 hours

---

## Approval Checklist

Before proceeding, confirm:
- [ ] User approves simplified threshold-based approach
- [ ] User approves 10-minute periodic execution
- [ ] User approves removal of EWMA and adaptive tolerance
- [ ] User approves synchronous (no threading) rollback

---

**Status:** DRAFT - Awaiting User Approval
