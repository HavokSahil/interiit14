"""
Test script for PolicyEngine and ConfigEngine.

Tests:
1. PolicyEngine - Client role assignment
2. PolicyEngine - QoS weights retrieval
3. PolicyEngine - Compliance evaluation
4. ConfigEngine - Configuration building
5. ConfigEngine - Validation
6. ConfigEngine - Apply and rollback
"""

from datatype import AccessPoint, Client
from slo_catalog import SLOCatalog
from policy_engine import PolicyEngine, OptimizationObjective
from config_engine import ConfigEngine, APConfig, NetworkConfig


def test_client_role_assignment():
    """Test 1: Client role assignment"""
    print("\n" + "="*60)
    print("TEST 1: Client Role Assignment")
    print("="*60)
    
    catalog = SLOCatalog("slo_catalog.yml")
    policy = PolicyEngine(catalog, default_role="BE")
    
    # Test assigning valid role
    success1 = policy.set_client_role(1, "ExamHallStrict")
    print(f"Assign ExamHallStrict to client 1: {'✓ PASS' if success1 else '✗ FAIL'}")
    
    # Test assigning invalid role
    success2 = policy.set_client_role(2, "InvalidRole")
    test2 = not success2
    print(f"Reject invalid role: {'✓ PASS' if test2 else '✗ FAIL'}")
    
    # Test getting assigned role
    role1 = policy.get_client_role(1)
    test3 = role1 == "ExamHallStrict"
    print(f"Get assigned role: {'✓ PASS' if test3 else '✗ FAIL'} ({role1})")
    
    # Test default role for unassigned client
    role3 = policy.get_client_role(3)
    test4 = role3 == "BE"
    print(f"Default role for unassigned: {'✓ PASS' if test4 else '✗ FAIL'} ({role3})")
    
    return success1 and test2 and test3 and test4


def test_qos_weights_retrieval():
    """Test 2: QoS weights retrieval"""
    print("\n" + "="*60)
    print("TEST 2: QoS Weights Retrieval")
    print("="*60)
    
    catalog = SLOCatalog("slo_catalog.yml")
    policy = PolicyEngine(catalog)
    
    # Assign roles
    policy.set_client_role(1, "ExamHallStrict")
    policy.set_client_role(2, "VO")
    
    # Get weights for ExamHallStrict
    weights1 = policy.get_client_qos_weights(1)
    expected1 = {'ws': 0.10, 'wt': 0.15, 'wr': 0.30, 'wl': 0.30, 'wa': 0.15}
    test1 = weights1 == expected1
    print(f"ExamHallStrict weights: {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"  Expected: {expected1}")
    print(f"  Got:      {weights1}")
    
    # Get weights for VO
    weights2 = policy.get_client_qos_weights(2)
    expected2 = {'ws': 0.25, 'wt': 0.10, 'wr': 0.35, 'wl': 0.25, 'wa': 0.05}
    test2 = weights2 == expected2
    print(f"VO weights: {'✓ PASS' if test2 else '✗ FAIL'}")
    
    return test1 and test2


def test_compliance_evaluation():
    """Test 3: Compliance evaluation"""
    print("\n" + "="*60)
    print("TEST 3: Compliance Evaluation")
    print("="*60)
    
    catalog = SLOCatalog("slo_catalog.yml")
    policy = PolicyEngine(catalog)
    policy.set_client_role(1, "ExamHallStrict")
    
    # Test compliant metrics
    metrics_good = {'RSSI_dBm': -60, 'Retry_pct': 2.0}
    actions1 = policy.evaluate_client_compliance(1, metrics_good)
    test1 = len(actions1) == 0
    print(f"Compliant metrics → no actions: {'✓ PASS' if test1 else '✗ FAIL'}")
    
    # Test violation
    metrics_bad = {'RSSI_dBm': -70, 'Retry_pct': 5.0}
    actions2 = policy.evaluate_client_compliance(1, metrics_bad)
    test2 = len(actions2) > 0
    print(f"Violations → actions triggered: {'✓ PASS' if test2 else '✗ FAIL'}")
    print(f"  Actions: {actions2}")
    
    return test1 and test2


def test_config_building():
    """Test 4: Configuration building"""
    print("\n" + "="*60)
    print("TEST 4: Configuration Building")
    print("="*60)
    
    aps = [
        AccessPoint(id=1, x=0, y=0, tx_power=20, channel=1),
        AccessPoint(id=2, x=100, y=100, tx_power=20, channel=6),
    ]
    
    config_engine = ConfigEngine(aps)
    
    # Build channel config
    channel_cfg = config_engine.build_channel_config(ap_id=1, channel=6)
    test1 = channel_cfg.channel == 6 and channel_cfg.ap_id == 1
    print(f"Build channel config: {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"  AP: {channel_cfg.ap_id}, Channel: {channel_cfg.channel}")
    
    # Build power config
    power_cfg = config_engine.build_power_config(ap_id=2, power=25.0)
    test2 = power_cfg.tx_power == 25.0 and power_cfg.ap_id == 2
    print(f"Build power config: {'✓ PASS' if test2 else '✗ FAIL'}")
    print(f"  AP: {power_cfg.ap_id}, Power: {power_cfg.tx_power} dBm")
    
    return test1 and test2


def test_config_validation():
    """Test 5: Configuration validation"""
    print("\n" + "="*60)
    print("TEST 5: Configuration Validation")
    print("="*60)
    
    aps = [AccessPoint(id=1, x=0, y=0, tx_power=20, channel=1)]
    config_engine = ConfigEngine(aps)
    
    # Valid config
    valid_cfg = config_engine.build_channel_config(ap_id=1, channel=6)
    valid_net_cfg = config_engine.build_network_config([valid_cfg])
    is_valid1, errors1 = config_engine.validate_config(valid_net_cfg)
    test1 = is_valid1 and len(errors1) == 0
    print(f"Valid config passes: {'✓ PASS' if test1 else '✗ FAIL'}")
    
    # Invalid channel
    invalid_cfg = APConfig(ap_id=1, channel=99, tx_power=20, bandwidth=20)
    invalid_net_cfg = config_engine.build_network_config([invalid_cfg])
    is_valid2, errors2 = config_engine.validate_config(invalid_net_cfg)
    test2 = not is_valid2 and len(errors2) > 0
    print(f"Invalid channel rejected: {'✓ PASS' if test2 else '✗ FAIL'}")
    if errors2:
        print(f"  Error: {errors2[0]}")
    
    # Invalid power
    invalid_power = APConfig(ap_id=1, channel=1, tx_power=100, bandwidth=20)
    invalid_power_cfg = config_engine.build_network_config([invalid_power])
    is_valid3, errors3 = config_engine.validate_config(invalid_power_cfg)
    test3 = not is_valid3 and len(errors3) > 0
    print(f"Invalid power rejected: {'✓ PASS' if test3 else '✗ FAIL'}")
    
    return test1 and test2 and test3


def test_apply_and_rollback():
    """Test 6: Apply and rollback"""
    print("\n" + "="*60)
    print("TEST 6: Apply and Rollback")
    print("="*60)
    
    aps = [AccessPoint(id=1, x=0, y=0, tx_power=20, channel=1)]
    config_engine = ConfigEngine(aps)
    
    # Save initial state
    initial_channel = aps[0].channel
    
    # Apply new config
    new_cfg = config_engine.build_channel_config(ap_id=1, channel=6)
    new_net_cfg = config_engine.build_network_config([new_cfg], metadata={'change': 'test'})
    success1 = config_engine.apply_config(new_net_cfg)
    test1 = success1 and aps[0].channel == 6
    print(f"Apply config: {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"  Channel changed: {initial_channel} → {aps[0].channel}")
    
    # Rollback
    success2 = config_engine.rollback(steps=1)
    test2 = success2 and aps[0].channel == initial_channel
    print(f"Rollback: {'✓ PASS' if test2 else '✗ FAIL'}")
    print(f"  Channel restored: {aps[0].channel}")
    
    return test1 and test2


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("POLICY & CONFIG ENGINE TEST SUITE")
    print("="*60)
    
    tests = [
        ("Client Role Assignment", test_client_role_assignment),
        ("QoS Weights Retrieval", test_qos_weights_retrieval),
        ("Compliance Evaluation", test_compliance_evaluation),
        ("Config Building", test_config_building),
        ("Config Validation", test_config_validation),
        ("Apply and Rollback", test_apply_and_rollback),
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
