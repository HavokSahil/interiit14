"""
Test script for SlowLoopController.

Tests:
1. Periodic execution
2. Channel optimization
3. Power optimization
4. Integration with RRMEngine
"""

from datatype import AccessPoint, Client, Interferer
from slo_catalog import SLOCatalog
from policy_engine import PolicyEngine
from config_engine import ConfigEngine
from clientview import ClientViewAPI
from sensing import SensingAPI
from slow_loop_controller import SlowLoopController
from model import PathLossModel
from rrmengine import RRMEngine


def test_periodic_execution():
    """Test 1: Periodic execution"""
    print("\n" + "="*60)
    print("TEST 1: Periodic Execution")
    print("="*60)
    
    aps = [AccessPoint(id=1, x=0, y=0, tx_power=20, channel=1)]
    clients = [Client(id=1, x=0, y=0, demand_mbps=50, associated_ap=1)]
    
    catalog = SLOCatalog("slo_catalog.yml")
    policy = PolicyEngine(catalog)
    config = ConfigEngine(aps)
    clientview = ClientViewAPI(aps, clients)
    
    controller = SlowLoopController(policy, config, None, clientview, period=10)
    
    # Test should execute on step 0 (first execution)
    test1 = controller.should_execute(0)
    print(f"Should execute on step 0: {'✓ PASS' if test1 else '✗ FAIL'}")
    
    # Simulate execution
    controller.execute(0)
    
    # Should not execute on step 5
    test2 = not controller.should_execute(5)
    print(f"Should NOT execute on step 5: {'✓ PASS' if test2 else '✗ FAIL'}")
    
    # Should execute on step 10
    test3 = controller.should_execute(10)
    print(f"Should execute on step 10: {'✓ PASS' if test3 else '✗ FAIL'}")
    
    return test1 and test2 and test3


def test_channel_optimization():
    """Test 2: Channel optimization"""
    print("\n" + "="*60)
    print("TEST 2: Channel Optimization")
    print("="*60)
    
    # Create 3 APs on same channel
    aps = [
        AccessPoint(id=1, x=0, y=0, tx_power=20, channel=1),
        AccessPoint(id=2, x=30, y=30, tx_power=20, channel=1),
        AccessPoint(id=3, x=60, y=60, tx_power=20, channel=1),
    ]
    
    # Create clients
    clients = [
        Client(id=1, x=0, y=0, demand_mbps=50, associated_ap=1, rssi_dbm=-50),
        Client(id=2, x=30, y=30, demand_mbps=50, associated_ap=2, rssi_dbm=-55),
        Client(id=3, x=60, y=60, demand_mbps=50, associated_ap=3, rssi_dbm=-60),
    ]
    
    catalog = SLOCatalog("slo_catalog.yml")
    policy = PolicyEngine(catalog)
    config = ConfigEngine(aps)
    clientview = ClientViewAPI(aps, clients)
    
    controller = SlowLoopController(policy, config, None, clientview, period=10)
    controller.set_optimization_mode("channel")
    
    # Get initial channels
    initial_channels = {ap.id: ap.channel for ap in aps}
    print(f"Initial channels: {initial_channels}")
    
    # Optimize
    new_config = controller.optimize_channels()
    
    test1 = new_config is not None
    print(f"Optimization returned config: {'✓ PASS' if test1 else '✗ FAIL'}")
    
    if new_config:
        # Apply config
        config.apply_config(new_config)
        
        # Check that channels changed
        final_channels = {ap.id: ap.channel for ap in aps}
        print(f"Final channels: {final_channels}")
        
        # At least some channels should be different from initial
        test2 = final_channels != initial_channels
        print(f"Channels changed: {'✓ PASS' if test2 else '✗ FAIL'}")
        
        # Check diversity (not all on same channel)
        unique_channels = len(set(final_channels.values()))
        test3 = unique_channels > 1
        print(f"Channel diversity ({unique_channels} unique): {'✓ PASS' if test3 else '✗ FAIL'}")
        
        return test1 and test2 and test3
    
    return test1


def test_power_optimization():
    """Test 3: Power optimization"""
    print("\n" + "="*60)
    print("TEST 3: Power Optimization")
    print("="*60)
    
    # Create APs with varied conditions
    aps = [
        AccessPoint(id=1, x=0, y=0, tx_power=30, channel=1),  # High power
        AccessPoint(id=2, x=100, y=100, tx_power=10, channel=6),  # Low power
    ]
    
    # Create clients with different RSSI
    clients = [
        Client(id=1, x=0, y=0, demand_mbps=50, associated_ap=1, rssi_dbm=-40),  # Good
        Client(id=2, x=50, y=50, demand_mbps=50, associated_ap=1, rssi_dbm=-80),  # Poor
        Client(id=3, x=100, y=100, demand_mbps=50, associated_ap=2, rssi_dbm=-45),  # Good
    ]
    
    catalog = SLOCatalog("slo_catalog.yml")
    policy = PolicyEngine(catalog)
    config = ConfigEngine(aps)
    clientview = ClientViewAPI(aps, clients)
    
    controller = SlowLoopController(policy, config, None, clientview, period=10)
    controller.set_optimization_mode("power")
    
    # Get initial powers
    initial_powers = {ap.id: ap.tx_power for ap in aps}
    print(f"Initial powers: {initial_powers}")
    
    # Optimize
    new_config = controller.optimize_power()
    
    test1 = new_config is not None
    print(f"Optimization returned config: {'✓ PASS' if test1 else '✗ FAIL'}")
    
    if new_config:
        # Check that config is valid
        is_valid, errors = config.validate_config(new_config)
        test2 = is_valid
        print(f"Config is valid: {'✓ PASS' if test2 else '✗ FAIL'}")
        if errors:
            print(f"  Errors: {errors}")
        
        return test1 and test2
    
    return test1


def test_rrm_integration():
    """Test 4: Integration with RRMEngine"""
    print("\n" + "="*60)
    print("TEST 4: RRMEngine Integration")
    print("="*60)
    
    # Create network
    aps = [
        AccessPoint(id=1, x=0, y=0, tx_power=20, channel=1),
        AccessPoint(id=2, x=50, y=50, tx_power=20, channel=1),
    ]
    
    clients = [
        Client(id=1, x=0, y=0, demand_mbps=50, associated_ap=1),
        Client(id=2, x=50, y=50, demand_mbps=50, associated_ap=2),
    ]
    
    # Initialize RRM Engine
    rrm = RRMEngine(
        access_points=aps,
        clients=clients,
        slo_catalog_path="slo_catalog.yml",
        slow_loop_period=5
    )
    
    # Set optimization mode
    rrm.slow_loop_engine.set_optimization_mode("channel")
    
    # Test execution
    results_step0 = rrm.execute(step=0)
    test1 = 'qoe' in results_step0
    print(f"Step 0 - QoE computed: {'✓ PASS' if test1 else '✗ FAIL'}")
    
    # Check slow loop was executed (may or may not produce config changes)
    test2 = rrm.slow_loop_engine.last_execution == 0
    print(f"Step 0 - Slow loop executed: {'✓ PASS' if test2 else '✗ FAIL'}")
    
    # Step 4 should not trigger slow loop
    results_step4 = rrm.execute(step=4)
    test3 = rrm.slow_loop_engine.last_execution == 0  # Still at step 0
    print(f"Step 4 - Slow loop NOT executed: {'✓ PASS' if test3 else '✗ FAIL'}")
    
    # Step 5 should trigger slow loop
    results_step5 = rrm.execute(step=5)
    test4 = rrm.slow_loop_engine.last_execution == 5
    print(f"Step 5 - Slow loop executed: {'✓ PASS' if test4 else '✗ FAIL'}")
    
    return test1 and test2 and test3 and test4


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("SLOW LOOP CONTROLLER TEST SUITE")
    print("="*60)
    
    tests = [
        ("Periodic Execution", test_periodic_execution),
        ("Channel Optimization", test_channel_optimization),
        ("Power Optimization", test_power_optimization),
        ("RRMEngine Integration", test_rrm_integration),
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
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
        return False


if __name__ == "__main__":
    run_all_tests()
