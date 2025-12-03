"""
Test script for FastLoopController.

Tests:
1. Client steering based on QoE
2. Client steering based on RSSI  
3. Load balancing
4. Integration with RRMEngine
"""

from datatype import AccessPoint, Client
from slo_catalog import SLOCatalog
from policy_engine import PolicyEngine
from config_engine import ConfigEngine
from clientview import ClientViewAPI
from fast_loop_controller import FastLoopController
from rrmengine import RRMEngine


def test_qoe_based_steering():
    """Test 1: Client steering based on QoE"""
    print("\n" + "="*60)
    print("TEST 1: QoE-Based Steering")
    print("="*60)
    
    # Create 2 APs
    aps = [
        AccessPoint(id=1, x=0, y=0, tx_power=20, channel=1),
        AccessPoint(id=2, x=30, y=30, tx_power=20, channel=6),
    ]
    
    # Create client with poor QoE at AP1, closer to AP2
    clients = [
        Client(id=1, x=25, y=25, demand_mbps=50, associated_ap=1,
               rssi_dbm=-80, throughput_mbps=10, max_rate_mbps=100,
               retry_rate=50, association_time=10),  # Poor metrics + established connection
    ]
    
    catalog = SLOCatalog("slo_catalog.yml")
    policy = PolicyEngine(catalog)
    config = ConfigEngine(aps)
    clientview = ClientViewAPI(aps, clients)
    
    controller = FastLoopController(policy, config, clientview, clients)
    controller.set_qoe_threshold(0.5)
    
    # Get initial association
    initial_ap = clients[0].associated_ap
    print(f"Initial AP: {initial_ap}")
    print(f"Client metrics: RSSI={clients[0].rssi_dbm}, Retry={clients[0].retry_rate}%")
    
    # Execute fast loop
    steering_actions = controller.execute()
    
    test1 = len(steering_actions) > 0
    print(f"Steering action triggered: {'✓ PASS' if test1 else '✗ FAIL'}")
    
    if steering_actions:
        client_id, old_ap, new_ap = steering_actions[0]
        print(f"  Steered: Client {client_id} from AP {old_ap} → AP {new_ap}")
        
        test2 = new_ap != old_ap
        print(f"Changed AP: {'✓ PASS' if test2 else '✗ FAIL'}")
        
        return test1 and test2
    
    return test1


def test_rssi_based_steering():
    """Test 2: Client steering based on RSSI"""
    print("\n" + "="*60)
    print("TEST 2: RSSI-Based Steering")
    print("="*60)
    
    # Create 2 APs
    aps = [
        AccessPoint(id=1, x=0, y=0, tx_power=20, channel=1),
        AccessPoint(id=2, x=20, y=20, tx_power=20, channel=6),
    ]
    
    # Client with poor RSSI at AP1, closer to AP2
    clients = [
        Client(id=1, x=18, y=18, demand_mbps=50, associated_ap=1,
               rssi_dbm=-80, association_time=10),  # Poor RSSI, established connection
    ]
    
    catalog = SLOCatalog("slo_catalog.yml")
    policy = PolicyEngine(catalog)
    config = ConfigEngine(aps)
    clientview = ClientViewAPI(aps, clients)
    
    controller = FastLoopController(policy, config, clientview, clients)
    controller.set_rssi_threshold(-75)
    
    print(f"Client RSSI: {clients[0].rssi_dbm} dBm (threshold: -75 dBm)")
    print(f"Association time: {clients[0].association_time} steps")
    
    # Execute
    steering_actions = controller.execute()
    
    test1 = len(steering_actions) > 0
    print(f"RSSI-based steering triggered: {'✓ PASS' if test1 else '✗ FAIL'}")
    
    if steering_actions:
        print(f"  Action: {steering_actions[0]}")
    
    return test1


def test_load_balancing():
    """Test 3: Load balancing"""
    print("\n" + "="*60)
    print("TEST 3: Load Balancing")
    print("="*60)
    
    # Create 2 APs
    aps = [
        AccessPoint(id=1, x=0, y=0, tx_power=20, channel=1),
        AccessPoint(id=2, x=50, y=50, tx_power=20, channel=6),
    ]
    
    # 5 clients on AP1, 0 on AP2 - imbalanced
    clients = [
        Client(id=i, x=5, y=5, demand_mbps=50, associated_ap=1,
               rssi_dbm=-50, association_time=10)
        for i in range(1, 6)
    ]
    
    catalog = SLOCatalog("slo_catalog.yml")
    policy = PolicyEngine(catalog)
    config = ConfigEngine(aps)
    clientview = ClientViewAPI(aps, clients)
    
    controller = FastLoopController(policy, config, clientview, clients)
    controller.enable_load_balance(True)
    controller.max_load_imbalance = 3  # Trigger if diff > 3
    
    # Count clients per AP
    ap1_clients = len([c for c in clients if c.associated_ap == 1])
    ap2_clients = len([c for c in clients if c.associated_ap == 2])
    
    print(f"Initial load: AP1={ap1_clients}, AP2={ap2_clients}")
    print(f"Load imbalance: {ap1_clients - ap2_clients} (threshold: 3)")
    
    # Execute
    steering_actions = controller.execute()
    
    test1 = len(steering_actions) > 0
    print(f"Load balancing triggered: {'✓ PASS' if test1 else '✗ FAIL'}")
    
    if steering_actions:
        # Check final distribution
        ap1_final = len([c for c in clients if c.associated_ap == 1])
        ap2_final = len([c for c in clients if c.associated_ap == 2])
        print(f"Final load: AP1={ap1_final}, AP2={ap2_final}")
        
        test2 = ap2_final > 0  # At least one client moved
        print(f"Client moved to balance: {'✓ PASS' if test2 else '✗ FAIL'}")
        
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
        AccessPoint(id=2, x=40, y=40, tx_power=20, channel=6),
    ]
    
    # Client with poor metrics
    clients = [
        Client(id=1, x=35, y=35, demand_mbps=50, associated_ap=1,
               rssi_dbm=-85, throughput_mbps=5, max_rate_mbps=100,
               retry_rate=60, association_time=10),
    ]
    
    # Initialize RRM Engine
    rrm = RRMEngine(
        access_points=aps,
        clients=clients,
        slo_catalog_path="slo_catalog.yml",
        slow_loop_period=100
    )
    
    # Configure fast loop
    rrm.fast_loop_engine.set_qoe_threshold(0.5)
    
    # Execute multiple steps
    steering_happened = False
    for step in range(3):
        results = rrm.execute(step=step)
        
        if 'steering' in results:
            steering_happened = True
            print(f"Step {step}: Steering occurred")
            print(f"  Actions: {results['steering']}")
    
    test1 = steering_happened
    print(f"\nSteering integrated with RRMEngine: {'✓ PASS' if test1 else '✗ FAIL'}")
    
    # Get statistics
    stats = rrm.fast_loop_engine.get_statistics()
    test2 = stats['total_steers'] > 0
    print(f"Steering statistics tracked: {'✓ PASS' if test2 else '✗ FAIL'}")
    print(f"  Total steers: {stats['total_steers']}")
    
    return test1 and test2


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("FAST LOOP CONTROLLER TEST SUITE")
    print("="*60)
    
    tests = [
        ("QoE-Based Steering", test_qoe_based_steering),
        ("RSSI-Based Steering", test_rssi_based_steering),
        ("Load Balancing", test_load_balancing),
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
