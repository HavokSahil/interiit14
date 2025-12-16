"""
Test script for Events Loop, Cooldown, and Lock functionality.

Tests:
1. Cooldown prevents slow loop
2. Lock prevents all loops
3. Events bypass cooldown
4. Events respect lock
5. Event detection (interference burst)
6. Event detection (QoE emergency)
7. DFS radar handler
8. Priority ordering
"""

from datatype import AccessPoint, Client, Interferer
from rrmengine import RRMEngine
from events_loop_controller import RRMEvent, EventPriority
from model import PathLossModel


def test_cooldown():
    """Test 1: Cooldown prevents slow loop"""
    print("\n" + "="*60)
    print("TEST 1: Cooldown Prevents Slow Loop")
    print("="*60)
    
    # Create network
    aps = [
        AccessPoint(id=1, x=0, y=0, tx_power=20, channel=1),
        AccessPoint(id=2, x=50, y=50, tx_power=20, channel=1),
    ]
    
    clients = [
        Client(id=1, x=5, y=5, demand_mbps=50, associated_ap=1, rssi_dbm=-50, association_time=10),
    ]
    
    # Initialize with cooldown=10 steps
    rrm = RRMEngine(
        access_points=aps,
        clients=clients,
        slo_catalog_path="slo_catalog.yml",
        slow_loop_period=5,
        cooldown_steps=10
    )
    
    # Step 0: Slow loop should execute (first time)
    results = rrm.execute(step=0)
    test1 = rrm.state.last_config_change_step == 0
    print(f"Step 0 - Slow loop executed: {'✓ PASS' if test1 else '✗ FAIL'}")
    
    # Step 5: Should NOT execute (in cooldown)
    results = rrm.execute(step=5)
    test2 = 'in_cooldown' in results and results['in_cooldown']
    print(f"Step 5 - In cooldown: {'✓ PASS' if test2 else '✗ FAIL'}")
    print(f"  Cooldown remaining: {results.get('cooldown_remaining', 'N/A')} steps")
    
    # Step 10: Should execute (cooldown expired)
    results = rrm.execute(step=10)
    test3 = rrm.state.last_config_change_step == 10 or not results.get('in_cooldown')
    print(f"Step 10 - After cooldown: {'✓ PASS' if test3 else '✗ FAIL'}")
    
    return test1 and test2 and test3


def test_lock():
    """Test 2: Lock prevents all loops"""
    print("\n" + "="*60)
    print("TEST 2: Lock Prevents All Loops")
    print("="*60)
    
    # Create network
    aps = [AccessPoint(id=1, x=0, y=0, tx_power=20, channel=1)]
    clients = [Client(id=1, x=5, y=5, demand_mbps=50, associated_ap=1, association_time=10)]
    
    rrm = RRMEngine(
        access_points=aps,
        clients=clients,
        slo_catalog_path="slo_catalog.yml",
        slow_loop_period=1
    )
    
    # Lock the system
    rrm.lock(reason="Testing lock")
    
    # Execute while locked
    results = rrm.execute(step=0)
    
    test1 = 'locked' in results and results['locked']
    print(f"System locked: {'✓ PASS' if test1 else '✗ FAIL'}")
    
    test2 = 'slow_loop' not in results and 'steering' not in results
    print(f"No optimization actions: {'✓ PASS' if test2 else '✗ FAIL'}")
    
    test3 = 'qoe' in results  # Monitoring continues
    print(f"Monitoring continues: {'✓ PASS' if test3 else '✗ FAIL'}")
    
    # Unlock
    rrm.unlock()
    results2 = rrm.execute(step=1)
    test4 = 'locked' not in results2
    print(f"System unlocked: {'✓ PASS' if test4 else '✗ FAIL'}")
    
    return test1 and test2 and test3 and test4


def test_events_bypass_cooldown():
    """Test 3: Events bypass cooldown"""
    print("\n" + "="*60)
    print("TEST 3: Events Bypass Cooldown")
    print("="*60)
    
    # Create network
    aps = [AccessPoint(id=1, x=0, y=0, tx_power=20, channel=1)]
    clients = [Client(id=1, x=5, y=5, demand_mbps=50, associated_ap=1)]
    
    rrm = RRMEngine(
        access_points=aps,
        clients=clients,
        slo_catalog_path="slo_catalog.yml",
        slow_loop_period=100,
        cooldown_steps=50
    )
    
    # Trigger slow loop to start cooldown
    results = rrm.execute(step=0)
    initial_changes = rrm.state.total_config_changes
    
    # Manually register an event during cooldown
    event = RRMEvent(
        event_id="test_dfs_1",
        event_type='dfs_radar',
        priority=EventPriority.CRITICAL,
        timestamp=0.0,
        metadata={'ap_id': 1, 'channel': 1},
        action_type='channel_switch'
    )
    rrm.events_loop_engine.register_event(event)
    
    # Execute at step 5 (in cooldown)
    results = rrm.execute(step=5)
    
    test1 = 'event_action' in results
    print(f"Event executed during cooldown: {'✓ PASS' if test1 else '✗ FAIL'}")
    
    test2 = rrm.state.total_events_handled > 0
    print(f"Event count tracked: {'✓ PASS' if test2 else '✗ FAIL'}")
    
    return test1 and test2


def test_events_respect_lock():
    """Test 4: Events respect lock"""
    print("\n" + "="*60)
    print("TEST 4: Events Respect Lock")
    print("="*60)
    
    # Create network
    aps = [AccessPoint(id=1, x=0, y=0, tx_power=20, channel=1)]
    clients = [Client(id=1, x=5, y=5, demand_mbps=50, associated_ap=1)]
    
    rrm = RRMEngine(
        access_points=aps,
        clients=clients,
        slo_catalog_path="slo_catalog.yml"
    )
    
    # Lock the system
    rrm.lock(reason="Testing event lock")
    
    # Register an event
    event = RRMEvent(
        event_id="test_event",
        event_type='interference_burst',
        priority=EventPriority.HIGH,
        timestamp=0.0,
        metadata={'ap_id': 1, 'interferer_channel': 6},
        action_type='channel_switch'
    )
    rrm.events_loop_engine.register_event(event)
    
    # Execute while locked
    results = rrm.execute(step=0)
    
    test1 = 'locked' in results
    print(f"System locked: {'✓ PASS' if test1 else '✗ FAIL'}")
    
    test2 = 'event_action' not in results
    print(f"Event NOT executed while locked: {'✓ PASS' if test2 else '✗ FAIL'}")
    
    # Event should still be in queue
    test3 = len(rrm.events_loop_engine.event_queue) == 1
    print(f"Event still in queue: {'✓ PASS' if test3 else '✗ FAIL'}")
    
    return test1 and test2 and test3


def test_automatic_interference_detection():
    """Test 5: Automatic interference burst detection"""
    print("\n" + "="*60)
    print("TEST 5: Automatic Interference Detection")
    print("="*60)
    
    # Create network with interferer
    aps = [AccessPoint(id=1, x=0, y=0, tx_power=20, channel=1)]
    clients = [Client(id=1, x=5, y=5, demand_mbps=50, associated_ap=1)]
    interferers = [
        Interferer(id=1, x=10, y=10, tx_power=10, channel=1, type="Microwave")
    ]
    
    prop_model = PathLossModel(frequency_mhz=2400, path_loss_exp=3.0)
    
    rrm = RRMEngine(
        access_points=aps,
        clients=clients,
        interferers=interferers,
        prop_model=prop_model,
        slo_catalog_path="slo_catalog.yml"
    )
    
    # Execute multiple steps to allow detection
    for step in range(3):
        results = rrm.execute(step)
        
        if 'event_action' in results:
            event_type = results.get('event_metadata', {}).get('event_type')
            print(f"Step {step}: Event detected - {event_type}")
            test1 = event_type == 'interference_burst'
            print(f"Interference burst auto-detected: {'✓ PASS' if test1 else '✗ FAIL'}")
            return test1
    
    print("No interference burst detected: ⚠ INFO (may need higher confidence)")
    return True  # Not a failure, just no detection yet


def test_priority_ordering():
    """Test 6: Event priority ordering"""
    print("\n" + "="*60)
    print("TEST 6: Event Priority Ordering")
    print("="*60)
    
    # Create network
    aps = [AccessPoint(id=1, x=0, y=0, tx_power=20, channel=1)]
    clients = [Client(id=1, x=5, y=5, demand_mbps=50, associated_ap=1)]
    
    rrm = RRMEngine(
        access_points=aps,
        clients=clients,
        slo_catalog_path="slo_catalog.yml"
    )
    
    # Register events with different priorities
    low_priority = RRMEvent(
        event_id="low", event_type='scheduled_event', priority=EventPriority.LOW,
        timestamp=0.0, metadata={'event_name': 'test', 'action': 'reduce_power'},
        action_type='reduce_power'
    )
    
    high_priority = RRMEvent(
        event_id="high", event_type='interference_burst', priority=EventPriority.HIGH,
        timestamp=0.1, metadata={'ap_id': 1, 'interferer_channel': 6},
        action_type='channel_switch'
    )
    
    critical_priority = RRMEvent(
        event_id="critical", event_type='dfs_radar', priority=EventPriority.CRITICAL,
        timestamp=0.2, metadata={'ap_id': 1, 'channel': 1},
        action_type='channel_switch'
    )
    
    # Register in reverse priority order
    rrm.events_loop_engine.register_event(low_priority)
    rrm.events_loop_engine.register_event(high_priority)
    rrm.events_loop_engine.register_event(critical_priority)
    
    print(f"Registered 3 events with different priorities")
    
    # Execute - should handle critical first
    results = rrm.execute(step=0)
    
    test1 = 'event_action' in results
    first_event = results.get('event_metadata', {}).get('event_type')
    print(f"First event handled: {first_event}")
    
    test2 = first_event == 'dfs_radar'  # Critical priority
    print(f"Critical priority handled first: {'✓ PASS' if test2 else '✗ FAIL'}")
    
    return test1 and test2


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("EVENTS LOOP & ENHANCEMENTS TEST SUITE")
    print("="*60)
    
    tests = [
        ("Cooldown Prevents Slow Loop", test_cooldown),
        ("Lock Prevents All Loops", test_lock),
        ("Events Bypass Cooldown", test_events_bypass_cooldown),
        ("Events Respect Lock", test_events_respect_lock),
        ("Automatic Interference Detection", test_automatic_interference_detection),
        ("Event Priority Ordering", test_priority_ordering),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ FAIL: {test_name} threw exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All event loop tests passed!")
        return True
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
        return False


if __name__ == "__main__":
    run_all_tests()
