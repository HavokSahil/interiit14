"""
Test script for the ClientView API.

This script demonstrates and tests the ClientView API functionality:
1. Tests signal quality computation (RSSI-based)
2. Tests throughput score computation
3. Tests reliability score computation
4. Tests latency score computation
5. Tests activity score computation
6. Tests client-to-AP matching
7. Tests complete QoE computation
8. Tests aggregate statistics
"""

from datatype import AccessPoint, Client
from clientview import ClientViewAPI, QoEComponents


def test_signal_quality():
    """Test 1: Signal quality computation"""
    print("\n" + "="*60)
    print("TEST 1: Signal Quality Computation")
    print("="*60)
    
    # Test case 1: RSSI = -90 dBm → S = 0
    client1 = Client(id=1, x=0, y=0, demand_mbps=10, rssi_dbm=-90.0)
    s1 = ClientViewAPI.compute_signal_quality(client1)
    print(f"RSSI = -90 dBm → S = {s1:.2f} (expected: 0.00)")
    test1 = abs(s1 - 0.0) < 0.01
    
    # Test case 2: RSSI = -30 dBm → S = 1
    client2 = Client(id=2, x=0, y=0, demand_mbps=10, rssi_dbm=-30.0)
    s2 = ClientViewAPI.compute_signal_quality(client2)
    print(f"RSSI = -30 dBm → S = {s2:.2f} (expected: 1.00)")
    test2 = abs(s2 - 1.0) < 0.01
    
    # Test case 3: RSSI = -50 dBm → S ≈ 0.67
    client3 = Client(id=3, x=0, y=0, demand_mbps=10, rssi_dbm=-50.0)
    s3 = ClientViewAPI.compute_signal_quality(client3)
    expected3 = (-50.0 - (-90.0)) / 60.0  # = 40/60 = 0.667
    print(f"RSSI = -50 dBm → S = {s3:.2f} (expected: {expected3:.2f})")
    test3 = abs(s3 - expected3) < 0.01
    
    if test1 and test2 and test3:
        print("✓ PASS: All signal quality tests passed!")
        return True
    else:
        print("✗ FAIL: Some signal quality tests failed!")
        return False


def test_throughput_score():
    """Test 2: Throughput score computation"""
    print("\n" + "="*60)
    print("TEST 2: Throughput Score Computation")
    print("="*60)
    
    # Test case 1: throughput = max_rate → T = 1.0
    client1 = Client(id=1, x=0, y=0, demand_mbps=10, 
                     throughput_mbps=100, max_rate_mbps=100)
    t1 = ClientViewAPI.compute_throughput_score(client1)
    print(f"Throughput = Max Rate → T = {t1:.2f} (expected: 1.00)")
    test1 = abs(t1 - 1.0) < 0.01
    
    # Test case 2: throughput = 0 → T = 0.0
    client2 = Client(id=2, x=0, y=0, demand_mbps=10,
                     throughput_mbps=0, max_rate_mbps=100)
    t2 = ClientViewAPI.compute_throughput_score(client2)
    print(f"Throughput = 0 → T = {t2:.2f} (expected: 0.00)")
    test2 = abs(t2 - 0.0) < 0.01
    
    # Test case 3: throughput = 0.5 × max_rate → T = 0.5
    client3 = Client(id=3, x=0, y=0, demand_mbps=10,
                     throughput_mbps=50, max_rate_mbps=100)
    t3 = ClientViewAPI.compute_throughput_score(client3)
    print(f"Throughput = 50% of Max → T = {t3:.2f} (expected: 0.50)")
    test3 = abs(t3 - 0.5) < 0.01
    
    if test1 and test2 and test3:
        print("✓ PASS: All throughput tests passed!")
        return True
    else:
        print("✗ FAIL: Some throughput tests failed!")
        return False


def test_reliability_score():
    """Test 3: Reliability score computation"""
    print("\n" + "="*60)
    print("TEST 3: Reliability Score Computation")
    print("="*60)
    
    # Test case 1: retry_rate = 0% → R = 1.0
    client1 = Client(id=1, x=0, y=0, demand_mbps=10, retry_rate=0.0)
    r1 = ClientViewAPI.compute_reliability_score(client1)
    print(f"Retry = 0% → R = {r1:.2f} (expected: 1.00)")
    test1 = abs(r1 - 1.0) < 0.01
    
    # Test case 2: retry_rate = 100% → R = 0.4 (1 - 0.6*1.0 = 0.4)
    client2 = Client(id=2, x=0, y=0, demand_mbps=10, retry_rate=100.0)
    r2 = ClientViewAPI.compute_reliability_score(client2)
    expected2 = 1.0 - 0.6 * 1.0  # = 0.4
    print(f"Retry = 100% → R = {r2:.2f} (expected: {expected2:.2f})")
    test2 = abs(r2 - expected2) < 0.01
    
    # Test case 3: retry_rate = 10% → R = 0.94 (1 - 0.6*0.1 = 0.94)
    client3 = Client(id=3, x=0, y=0, demand_mbps=10, retry_rate=10.0)
    r3 = ClientViewAPI.compute_reliability_score(client3)
    expected3 = 1.0 - 0.6 * 0.1  # = 0.94
    print(f"Retry = 10% → R = {r3:.2f} (expected: {expected3:.2f})")
    test3 = abs(r3 - expected3) < 0.01
    
    if test1 and test2 and test3:
        print("✓ PASS: All reliability tests passed!")
        return True
    else:
        print("✗ FAIL: Some reliability tests failed!")
        return False


def test_latency_score():
    """Test 4: Latency score computation"""
    print("\n" + "="*60)
    print("TEST 4: Latency Score Computation")
    print("="*60)
    
    # Test case 1: inactive = 0 ms → L = 1.0
    client1 = Client(id=1, x=0, y=0, demand_mbps=10, inactive_msec=0.0)
    l1 = ClientViewAPI.compute_latency_score(client1)
    print(f"Inactive = 0 ms → L = {l1:.2f} (expected: 1.00)")
    test1 = abs(l1 - 1.0) < 0.01
    
    # Test case 2: inactive = 5000 ms → L = 0.0
    client2 = Client(id=2, x=0, y=0, demand_mbps=10, inactive_msec=5000.0)
    l2 = ClientViewAPI.compute_latency_score(client2)
    print(f"Inactive = 5000 ms → L = {l2:.2f} (expected: 0.00)")
    test2 = abs(l2 - 0.0) < 0.01
    
    # Test case 3: inactive = 2500 ms → L = 0.5
    client3 = Client(id=3, x=0, y=0, demand_mbps=10, inactive_msec=2500.0)
    l3 = ClientViewAPI.compute_latency_score(client3)
    expected3 = 1.0 - 2500.0 / 5000.0  # = 0.5
    print(f"Inactive = 2500 ms → L = {l3:.2f} (expected: {expected3:.2f})")
    test3 = abs(l3 - expected3) < 0.01
    
    if test1 and test2 and test3:
        print("✓ PASS: All latency tests passed!")
        return True
    else:
        print("✗ FAIL: Some latency tests failed!")
        return False


def test_activity_score():
    """Test 5: Activity score computation"""
    print("\n" + "="*60)
    print("TEST 5: Activity Score Computation")
    print("="*60)
    
    # Test case 1: no packets → A = 0.0
    client1 = Client(id=1, x=0, y=0, demand_mbps=10, tx_packets=0, rx_packets=0)
    a1 = ClientViewAPI.compute_activity_score(client1)
    print(f"Packets = 0 → A = {a1:.2f} (expected: 0.00)")
    test1 = abs(a1 - 0.0) < 0.01
    
    # Test case 2: 1000 packets → A = 1.0
    client2 = Client(id=2, x=0, y=0, demand_mbps=10, tx_packets=500, rx_packets=500)
    a2 = ClientViewAPI.compute_activity_score(client2)
    print(f"Packets = 1000 → A = {a2:.2f} (expected: 1.00)")
    test2 = abs(a2 - 1.0) < 0.01
    
    # Test case 3: 500 packets → A = 0.5
    client3 = Client(id=3, x=0, y=0, demand_mbps=10, tx_packets=250, rx_packets=250)
    a3 = ClientViewAPI.compute_activity_score(client3)
    expected3 = 500.0 / 1000.0  # = 0.5
    print(f"Packets = 500 → A = {a3:.2f} (expected: {expected3:.2f})")
    test3 = abs(a3 - expected3) < 0.01
    
    if test1 and test2 and test3:
        print("✓ PASS: All activity tests passed!")
        return True
    else:
        print("✗ FAIL: Some activity tests failed!")
        return False


def test_client_ap_matching():
    """Test 6: Client-to-AP matching"""
    print("\n" + "="*60)
    print("TEST 6: Client-to-AP Matching")
    print("="*60)
    
    # Create 2 APs
    ap1 = AccessPoint(id=1, x=0, y=0, tx_power=20, channel=1)
    ap2 = AccessPoint(id=2, x=100, y=100, tx_power=20, channel=6)
    aps = [ap1, ap2]
    
    # Create 4 clients - 2 for AP1, 1 for AP2, 1 unassociated
    clients = [
        Client(id=1, x=0, y=0, demand_mbps=10, associated_ap=1),
        Client(id=2, x=10, y=10, demand_mbps=10, associated_ap=1),
        Client(id=3, x=100, y=100, demand_mbps=10, associated_ap=2),
        Client(id=4, x=50, y=50, demand_mbps=10, associated_ap=None),
    ]
    
    api = ClientViewAPI(aps, clients)
    
    # Test AP1 clients
    ap1_clients = api.get_ap_clients(ap1)
    print(f"AP {ap1.id} clients: {[c.id for c in ap1_clients]} (expected: [1, 2])")
    test1 = len(ap1_clients) == 2 and all(c.id in [1, 2] for c in ap1_clients)
    
    # Test AP2 clients
    ap2_clients = api.get_ap_clients(ap2)
    print(f"AP {ap2.id} clients: {[c.id for c in ap2_clients]} (expected: [3])")
    test2 = len(ap2_clients) == 1 and ap2_clients[0].id == 3
    
    if test1 and test2:
        print("✓ PASS: Client-to-AP matching works correctly!")
        return True
    else:
        print("✗ FAIL: Client-to-AP matching failed!")
        return False


def test_complete_qoe_computation():
    """Test 7: Complete QoE computation"""
    print("\n" + "="*60)
    print("TEST 7: Complete QoE Computation")
    print("="*60)
    
    # Create APs
    ap1 = AccessPoint(id=1, x=20, y=20, tx_power=20, channel=1)
    ap2 = AccessPoint(id=2, x=80, y=80, tx_power=20, channel=11)
    aps = [ap1, ap2]
    
    # Create clients with realistic QoE data
    clients = [
        # Good client for AP1
        Client(id=1, x=20, y=20, demand_mbps=50, associated_ap=1,
               rssi_dbm=-40.0, throughput_mbps=80.0, max_rate_mbps=100.0,
               retry_rate=5.0, inactive_msec=100.0,
               tx_packets=400, rx_packets=400),
        
        # Poor client for AP1
        Client(id=2, x=50, y=50, demand_mbps=50, associated_ap=1,
               rssi_dbm=-85.0, throughput_mbps=20.0, max_rate_mbps=100.0,
               retry_rate=50.0, inactive_msec=3000.0,
               tx_packets=50, rx_packets=50),
        
        # Medium client for AP2
        Client(id=3, x=80, y=80, demand_mbps=50, associated_ap=2,
               rssi_dbm=-60.0, throughput_mbps=60.0, max_rate_mbps=100.0,
               retry_rate=15.0, inactive_msec=1000.0,
               tx_packets=300, rx_packets=300),
    ]
    
    api = ClientViewAPI(aps, clients)
    
    # Compute all views
    views = api.compute_all_views()
    
    print(f"\nNumber of APs: {len(aps)}")
    print(f"Number of Views: {len(views)}")
    
    # Verify all APs have views
    test1 = len(views) == len(aps)
    print(f"All APs have views: {test1}")
    
    # Verify AP1 has 2 clients
    ap1_view = views[1]
    test2 = ap1_view.num_clients == 2
    print(f"AP1 has 2 clients: {test2}")
    
    # Verify AP2 has 1 client
    ap2_view = views[2]
    test3 = ap2_view.num_clients == 1
    print(f"AP2 has 1 client: {test3}")
    
    # Verify QoE scores are in valid range [0, 1]
    all_qoe_valid = True
    for ap_id, view in views.items():
        for result in view.client_results:
            if not (0.0 <= result.qoe_ap <= 1.0):
                all_qoe_valid = False
                print(f"Invalid QoE for Client {result.client_id}: {result.qoe_ap}")
    
    print(f"All QoE scores in [0, 1]: {all_qoe_valid}")
    test4 = all_qoe_valid
    
    # Print detailed results
    api.print_all_views(views)
    
    # Verify components are computed
    test5 = True
    for ap_id, view in views.items():
        for result in view.client_results:
            comp = result.components
            if not all(0.0 <= score <= 1.0 for score in [
                comp.signal_quality, comp.throughput, comp.reliability,
                comp.latency, comp.activity
            ]):
                test5 = False
                print(f"Invalid component for Client {result.client_id}")
    
    print(f"All component scores in [0, 1]: {test5}")
    
    # Verify aggregate statistics
    test6 = (0.0 <= ap1_view.avg_qoe <= 1.0 and
             0.0 <= ap1_view.min_qoe <= 1.0 and
             0.0 <= ap1_view.max_qoe <= 1.0)
    print(f"AP1 statistics valid: {test6}")
    
    if all([test1, test2, test3, test4, test5, test6]):
        print("\n✓ PASS: Complete QoE computation works correctly!")
        return True
    else:
        print("\n✗ FAIL: Some QoE computation tests failed!")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("CLIENTVIEW API TEST SUITE")
    print("="*60)
    
    tests = [
        ("Signal Quality", test_signal_quality),
        ("Throughput Score", test_throughput_score),
        ("Reliability Score", test_reliability_score),
        ("Latency Score", test_latency_score),
        ("Activity Score", test_activity_score),
        ("Client-AP Matching", test_client_ap_matching),
        ("Complete QoE Computation", test_complete_qoe_computation),
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
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")


if __name__ == "__main__":
    run_all_tests()
