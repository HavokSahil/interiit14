"""
Test script for the Sensing API.

This script demonstrates and tests the sensing API functionality:
1. Creates a test scenario with APs and interferers
2. Tests distance calculation
3. Tests major interferer identification
4. Tests confidence calculation
5. Tests complete sensing flow
"""

from datatype import AccessPoint, Interferer
from model import PathLossModel
from sensing import SensingAPI, SensingResult
import math


def test_distance_calculation():
    """Test 1: Distance calculation accuracy"""
    print("\n" + "="*60)
    print("TEST 1: Distance Calculation Accuracy")
    print("="*60)
    
    # Create AP at (0, 0)
    ap = AccessPoint(id=1, x=0.0, y=0.0, tx_power=20.0, channel=6)
    
    # Create interferer at (3, 4) - should be 5.0 meters away
    interferer = Interferer(
        id=1, x=3.0, y=4.0, tx_power=15.0, channel=6, 
        type="Bluetooth", duty_cycle=0.5
    )
    
    # Create propagation model
    prop_model = PathLossModel(frequency_mhz=2400, path_loss_exp=2.0)
    
    # Create API and scan
    api = SensingAPI([ap], [interferer], prop_model)
    results = api.scan_interferers(ap)
    
    # Verify
    interferer_obj, distance, rx_power = results[0]
    expected_distance = 5.0
    
    print(f"AP Position: ({ap.x}, {ap.y})")
    print(f"Interferer Position: ({interferer.x}, {interferer.y})")
    print(f"Calculated Distance: {distance:.2f} meters")
    print(f"Expected Distance: {expected_distance:.2f} meters")
    print(f"Received Power: {rx_power:.2f} dBm")
    
    if abs(distance - expected_distance) < 0.01:
        print("✓ PASS: Distance calculation is correct!")
    else:
        print("✗ FAIL: Distance calculation is incorrect!")
    
    return distance == expected_distance


def test_major_interferer_identification():
    """Test 2: Major interferer identification"""
    print("\n" + "="*60)
    print("TEST 2: Major Interferer Identification")
    print("="*60)
    
    # Create AP at (50, 50)
    ap = AccessPoint(id=1, x=50.0, y=50.0, tx_power=20.0, channel=6)
    
    # Create 3 interferers at different distances
    interferers = [
        # Close interferer (should be major)
        Interferer(id=1, x=52.0, y=52.0, tx_power=15.0, channel=6, 
                  type="Bluetooth", duty_cycle=0.8),
        # Medium distance
        Interferer(id=2, x=60.0, y=60.0, tx_power=15.0, channel=6, 
                  type="BLE", duty_cycle=0.5),
        # Far interferer
        Interferer(id=3, x=80.0, y=80.0, tx_power=15.0, channel=6, 
                  type="Microwave", duty_cycle=1.0),
    ]
    
    prop_model = PathLossModel(frequency_mhz=2400, path_loss_exp=2.0)
    api = SensingAPI([ap], interferers, prop_model)
    
    # Scan interferers
    scan_results = api.scan_interferers(ap)
    
    print("\nScanned Interferers:")
    for i, (interf, dist, rx_power) in enumerate(scan_results, 1):
        print(f"  Interferer {interf.id}: Distance={dist:.2f}m, "
              f"RxPower={rx_power:.2f}dBm, Type={interf.type}, "
              f"DutyCycle={interf.duty_cycle:.2f}")
    
    # Identify major interferer
    major, confidence = api.identify_major_interferer(scan_results)
    
    print(f"\nMajor Interferer: ID={major.id}, Type={major.type}")
    print(f"Confidence: {confidence:.2f}")
    
    # The closest one (ID=1) should be major
    if major.id == 1:
        print("✓ PASS: Closest interferer correctly identified as major!")
        return True
    else:
        print("✗ FAIL: Wrong interferer identified as major!")
        return False


def test_confidence_calculation():
    """Test 3: Confidence calculation"""
    print("\n" + "="*60)
    print("TEST 3: Confidence Calculation")
    print("="*60)
    
    ap = AccessPoint(id=1, x=50.0, y=50.0, tx_power=20.0, channel=6)
    prop_model = PathLossModel(frequency_mhz=2400, path_loss_exp=2.0)
    
    # Test 3a: Single interferer (should give confidence = 1.0)
    print("\nTest 3a: Single interferer")
    interferers_single = [
        Interferer(id=1, x=52.0, y=52.0, tx_power=15.0, channel=6, 
                  type="Bluetooth", duty_cycle=1.0)
    ]
    
    api = SensingAPI([ap], interferers_single, prop_model)
    scan = api.scan_interferers(ap)
    major, confidence = api.identify_major_interferer(scan)
    
    print(f"  Confidence with 1 interferer: {confidence:.2f}")
    test_3a = (confidence == 1.0)
    if test_3a:
        print("  ✓ PASS: Single interferer gives confidence = 1.0")
    else:
        print("  ✗ FAIL: Single interferer should give confidence = 1.0")
    
    # Test 3b: Two interferers with similar power (low confidence)
    print("\nTest 3b: Two interferers with similar power")
    interferers_similar = [
        Interferer(id=1, x=52.0, y=52.0, tx_power=15.0, channel=6, 
                  type="Bluetooth", duty_cycle=1.0),
        Interferer(id=2, x=52.5, y=52.5, tx_power=15.0, channel=6, 
                  type="BLE", duty_cycle=1.0),
    ]
    
    api = SensingAPI([ap], interferers_similar, prop_model)
    scan = api.scan_interferers(ap)
    major, confidence = api.identify_major_interferer(scan)
    
    print(f"  Confidence with similar interferers: {confidence:.2f}")
    test_3b = (confidence < 0.5)  # Should be low confidence
    if test_3b:
        print("  ✓ PASS: Similar interferers give low confidence")
    else:
        print("  ✗ FAIL: Similar interferers should give low confidence")
    
    # Test 3c: Dominant interferer (high confidence)
    print("\nTest 3c: Dominant interferer")
    interferers_dominant = [
        Interferer(id=1, x=51.0, y=51.0, tx_power=20.0, channel=6, 
                  type="Bluetooth", duty_cycle=1.0),
        Interferer(id=2, x=70.0, y=70.0, tx_power=10.0, channel=6, 
                  type="BLE", duty_cycle=0.3),
    ]
    
    api = SensingAPI([ap], interferers_dominant, prop_model)
    scan = api.scan_interferers(ap)
    major, confidence = api.identify_major_interferer(scan)
    
    print(f"  Confidence with dominant interferer: {confidence:.2f}")
    test_3c = (confidence > 0.8)  # Should be high confidence
    if test_3c:
        print("  ✓ PASS: Dominant interferer gives high confidence")
    else:
        print("  ✗ FAIL: Dominant interferer should give high confidence")
    
    return test_3a and test_3b and test_3c


def test_complete_sensing_flow():
    """Test 4: Complete sensing flow with multiple APs"""
    print("\n" + "="*60)
    print("TEST 4: Complete Sensing Flow")
    print("="*60)
    
    # Create 2 APs at different locations
    aps = [
        AccessPoint(id=1, x=20.0, y=20.0, tx_power=20.0, channel=1),
        AccessPoint(id=2, x=80.0, y=80.0, tx_power=20.0, channel=11),
    ]
    
    # Create 3 interferers
    interferers = [
        Interferer(id=1, x=25.0, y=25.0, tx_power=15.0, channel=1, 
                  type="Bluetooth", duty_cycle=0.8, bandwidth=20.0),
        Interferer(id=2, x=50.0, y=50.0, tx_power=18.0, channel=6, 
                  type="BLE", duty_cycle=0.5, bandwidth=10.0),
        Interferer(id=3, x=75.0, y=75.0, tx_power=20.0, channel=11, 
                  type="Microwave", duty_cycle=0.3, bandwidth=22.0),
    ]
    
    prop_model = PathLossModel(frequency_mhz=2400, path_loss_exp=2.0)
    api = SensingAPI(aps, interferers, prop_model)
    
    # Compute sensing results for all APs
    results = api.compute_sensing_results()
    
    print(f"\nNumber of APs: {len(aps)}")
    print(f"Number of Interferers: {len(interferers)}")
    print(f"Number of Sensing Results: {len(results)}")
    
    # Verify each AP has a unique result
    test_passed = True
    
    if len(results) != len(aps):
        print("✗ FAIL: Not all APs have sensing results!")
        test_passed = False
    
    # Display results
    api.print_sensing_results(results)
    
    # Verify all required fields are populated
    for ap_id, result in results.items():
        print(f"\nVerifying AP {ap_id} result:")
        
        checks = [
            (result.major_interferer_type in ["Bluetooth", "BLE", "Microwave"], 
             "Valid interferer type"),
            (0.0 <= result.confidence <= 1.0, 
             f"Confidence in range [0,1]: {result.confidence:.2f}"),
            (2.0 <= result.center_frequency <= 3.0, 
             f"Valid frequency: {result.center_frequency:.3f} GHz"),
            (0.0 <= result.duty_cycle <= 1.0, 
             f"Duty cycle in range [0,1]: {result.duty_cycle:.2f}"),
            (result.bandwidth > 0, 
             f"Valid bandwidth: {result.bandwidth} MHz"),
        ]
        
        for check, desc in checks:
            if check:
                print(f"  ✓ {desc}")
            else:
                print(f"  ✗ {desc}")
                test_passed = False
    
    if test_passed:
        print("\n✓ PASS: Complete sensing flow works correctly!")
    else:
        print("\n✗ FAIL: Some checks failed!")
    
    return test_passed


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("SENSING API TEST SUITE")
    print("="*60)
    
    tests = [
        ("Distance Calculation", test_distance_calculation),
        ("Major Interferer Identification", test_major_interferer_identification),
        ("Confidence Calculation", test_confidence_calculation),
        ("Complete Sensing Flow", test_complete_sensing_flow),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ FAIL: {test_name} threw exception: {e}")
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
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")


if __name__ == "__main__":
    run_all_tests()
