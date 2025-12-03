"""
Test script for the SLO Catalog.

This script tests:
1. YAML loading and parsing
2. Role configuration retrieval
3. QoS weights extraction
4. Enforcement rule evaluation
5. Regulatory compliance checking
"""

from slo_catalog import SLOCatalog, RoleConfig, EnforcementRule
import os


def test_yaml_loading():
    """Test 1: YAML loading and parsing"""
    print("\n" + "="*60)
    print("TEST 1: YAML Loading and Parsing")
    print("="*60)
    
    catalog_path = "slo_catalog.yml"
    
    try:
        catalog = SLOCatalog(catalog_path)
        print(f"✓ YAML loaded successfully from {catalog_path}")
        print(f"  Version: {catalog.config.version}")
        print(f"  Description: {catalog.config.description}")
        print(f"  Number of roles: {len(catalog.list_roles())}")
        print(f"  Roles: {', '.join(catalog.list_roles())}")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_role_retrieval():
    """Test 2: Role configuration retrieval"""
    print("\n" + "="*60)
    print("TEST 2: Role Configuration Retrieval")
    print("="*60)
    
    catalog = SLOCatalog("slo_catalog.yml")
    
    # Test getting existing role
    role = catalog.get_role("ExamHallStrict")
    test1 = role is not None and role.role_id == "ExamHallStrict"
    print(f"Get ExamHallStrict: {'✓ PASS' if test1 else '✗ FAIL'}")
    if role:
        print(f"  Display Name: {role.display_name}")
        print(f"  Purpose: {role.purpose}")
    
    # Test getting non-existent role
    role2 = catalog.get_role("NonExistent")
    test2 = role2 is None
    print(f"Get NonExistent role returns None: {'✓ PASS' if test2 else '✗ FAIL'}")
    
    # Test listing all roles
    roles = catalog.list_roles()
    expected_roles = ["ExamHallStrict", "ExamHallModerate", "VO", "VI", "BE", "BK", "Guest", "IoT"]
    test3 = all(r in roles for r in expected_roles)
    print(f"All expected roles present: {'✓ PASS' if test3 else '✗ FAIL'}")
    
    return test1 and test2 and test3


def test_qos_weights():
    """Test 3: QoS weights extraction"""
    print("\n" + "="*60)
    print("TEST 3: QoS Weights Extraction")
    print("="*60)
    
    catalog = SLOCatalog("slo_catalog.yml")
    
    # Test ExamHallStrict weights
    weights = catalog.get_qos_weights("ExamHallStrict")
    expected = {'ws': 0.10, 'wt': 0.15, 'wr': 0.30, 'wl': 0.30, 'wa': 0.15}
    
    test1 = weights == expected
    print(f"ExamHallStrict weights: {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"  Expected: {expected}")
    print(f"  Got:      {weights}")
    
    # Test VO weights
    vo_weights = catalog.get_qos_weights("VO")
    vo_expected = {'ws': 0.25, 'wt': 0.10, 'wr': 0.35, 'wl': 0.25, 'wa': 0.05}
    test2 = vo_weights == vo_expected
    print(f"VO weights: {'✓ PASS' if test2 else '✗ FAIL'}")
    print(f"  Expected: {vo_expected}")
    print(f"  Got:      {vo_weights}")
    
    # Test weights sum to 1.0
    total = sum(weights.values())
    test3 = abs(total - 1.0) < 0.01
    print(f"Weights sum to 1.0: {'✓ PASS' if test3 else '✗ FAIL'} (sum={total:.2f})")
    
    return test1 and test2 and test3


def test_enforcement_evaluation():
    """Test 4: Enforcement rule evaluation"""
    print("\n" + "="*60)
    print("TEST 4: Enforcement Rule Evaluation")
    print("="*60)
    
    catalog = SLOCatalog("slo_catalog.yml")
    
    # Test case 1: All metrics compliant (within thresholds)
    metrics_good = {
        'RSSI_dBm': -60.0,    # Above -66 threshold
        'SNR_dB': 25.0,       # Above 20 threshold
        'PER_pct': 1.0,       # Below 3.0 threshold
        'Retry_pct': 2.0,     # Below 3.0 threshold
        'Airtime_util': 50,   # Below 60 threshold
        'CCA_busy': 50        # Below 60 threshold
    }
    
    actions1 = catalog.evaluate_enforcement("ExamHallStrict", metrics_good)
    test1 = len(actions1) == 0
    print(f"Compliant metrics → no actions: {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"  Actions: {actions1}")
    
    # Test case 2: RSSI violation
    metrics_bad_rssi = {
        'RSSI_dBm': -70.0,    # Below -66 threshold → violation
        'SNR_dB': 25.0,
        'PER_pct': 1.0,
        'Retry_pct': 2.0,
    }
    
    actions2 = catalog.evaluate_enforcement("ExamHallStrict", metrics_bad_rssi)
    test2 = "IncreaseTxPower" in actions2 or "Steer" in actions2
    print(f"RSSI violation → IncreaseTxPower/Steer: {'✓ PASS' if test2 else '✗ FAIL'}")
    print(f"  Actions: {actions2}")
    
    # Test case 3: Multiple violations
    metrics_bad_multiple = {
        'RSSI_dBm': -70.0,    # Violation
        'Retry_pct': 5.0,     # Violation (> 3.0)
        'CCA_busy': 70        # Violation (> 60)
    }
    
    actions3 = catalog.evaluate_enforcement("ExamHallStrict", metrics_bad_multiple)
    test3 = len(actions3) > 0
    print(f"Multiple violations → multiple actions: {'✓ PASS' if test3 else '✗ FAIL'}")
    print(f"  Actions: {actions3}")
    
    return test1 and test2 and test3


def test_regulatory_compliance():
    """Test 5: Regulatory compliance checking"""
    print("\n" + "="*60)
    print("TEST 5: Regulatory Compliance Checking")
    print("="*60)
    
    catalog = SLOCatalog("slo_catalog.yml")
    
    # Test case 1: Compliant config for ExamHallStrict (max 20 MHz)
    config1 = {'channel_width_MHz': 20}
    compliant1, violations1 = catalog.check_regulatory_compliance("ExamHallStrict", config1)
    test1 = compliant1 and len(violations1) == 0
    print(f"ExamHallStrict 20MHz: {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"  Compliant: {compliant1}, Violations: {violations1}")
    
    # Test case 2: Non-compliant config for ExamHallStrict
    config2 = {'channel_width_MHz': 80}  # Exceeds 20 MHz limit
    compliant2, violations2 = catalog.check_regulatory_compliance("ExamHallStrict", config2)
    test2 = not compliant2 and len(violations2) > 0
    print(f"ExamHallStrict 80MHz violation: {'✓ PASS' if test2 else '✗ FAIL'}")
    print(f"  Compliant: {compliant2}, Violations: {violations2}")
    
    # Test case 3: Compliant config for BE (max 80 MHz)
    config3 = {'channel_width_MHz': 80}
    compliant3, violations3 = catalog.check_regulatory_compliance("BE", config3)
    test3 = compliant3 and len(violations3) == 0
    print(f"BE 80MHz: {'✓ PASS' if test3 else '✗ FAIL'}")
    print(f"  Compliant: {compliant3}, Violations: {violations3}")
    
    return test1 and test2 and test3


def test_global_config():
    """Test 6: Global configuration retrieval"""
    print("\n" + "="*60)
    print("TEST 6: Global Configuration Retrieval")
    print("="*60)
    
    catalog = SLOCatalog("slo_catalog.yml")
    
    # Test normalizers
    normalizers = catalog.get_global_normalizers()
    test1 = normalizers.get('RSSI_min') == -95
    print(f"RSSI_min normalizer: {'✓ PASS' if test1 else '✗ FAIL'} ({normalizers.get('RSSI_min')})")
    
    # Test defaults
    defaults = catalog.get_global_defaults()
    test2 = defaults.get('monitoring_window_seconds') == 300
    print(f"Monitoring window default: {'✓ PASS' if test2 else '✗ FAIL'} ({defaults.get('monitoring_window_seconds')})")
    
    return test1 and test2


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("SLO CATALOG TEST SUITE")
    print("="*60)
    
    tests = [
        ("YAML Loading", test_yaml_loading),
        ("Role Retrieval", test_role_retrieval),
        ("QoS Weights", test_qos_weights),
        ("Enforcement Evaluation", test_enforcement_evaluation),
        ("Regulatory Compliance", test_regulatory_compliance),
        ("Global Configuration", test_global_config),
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
