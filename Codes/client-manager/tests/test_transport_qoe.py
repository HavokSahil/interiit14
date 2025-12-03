#!/usr/bin/env python3
"""
Test script for Transport Layer Stats infrastructure.

This script tests:
1. TransportStats model parsing from iperf3 JSON
2. TransportStatsDB storage and retrieval
3. TransportQoE scoring functions
4. Overall QoE calculation
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transport_stats import TransportStats
from db.transport_stats_db import TransportStatsDB
from metrics.transport_qoe import TransportQoE


# Sample iperf3 JSON output (simplified)
SAMPLE_IPERF3_JSON = {
    "end": {
        "streams": [
            {
                "sender": {
                    "mean_rtt": 50000,  # 50ms in microseconds
                    "retransmits": 20,
                    "seconds": 10,
                    "bits_per_second": 100000000,  # 100 Mbps
                    "bytes": 125000000,
                    "max_snd_cwnd": 65536,
                    "max_rtt": 80000,
                    "min_rtt": 20000
                }
            }
        ]
    }
}


def test_transport_stats_parsing():
    """Test TransportStats.from_iperf3_json()"""
    print("\n" + "="*60)
    print("TEST 1: TransportStats Parsing")
    print("="*60)
    
    try:
        stats = TransportStats.from_iperf3_json(SAMPLE_IPERF3_JSON, "aa:bb:cc:dd:ee:ff")
        
        print(f"✓ Successfully parsed iperf3 JSON")
        print(f"  MAC: {stats.sta_mac}")
        print(f"  Mean RTT: {stats.mean_rtt_ms:.1f} ms")
        print(f"  Retrans/sec: {stats.retrans_per_sec:.2f}")
        print(f"  Mean Mbps: {stats.mean_mbps:.1f}")
        print(f"  Total retransmits: {stats.total_retransmits}")
        
        # Verify values
        assert stats.mean_rtt_ms == 50.0, f"Expected RTT 50ms, got {stats.mean_rtt_ms}"
        assert stats.retrans_per_sec == 2.0, f"Expected 2.0 retrans/s, got {stats.retrans_per_sec}"
        assert stats.mean_mbps == 100.0, f"Expected 100 Mbps, got {stats.mean_mbps}"
        
        print("✓ All assertions passed")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_transport_stats_db():
    """Test TransportStatsDB storage and retrieval"""
    print("\n" + "="*60)
    print("TEST 2: TransportStatsDB Operations")
    print("="*60)
    
    try:
        db = TransportStatsDB()
        db.clear()  # Clean slate
        
        # Create sample stats
        stats1 = TransportStats.from_iperf3_json(SAMPLE_IPERF3_JSON, "aa:bb:cc:dd:ee:01")
        stats2 = TransportStats.from_iperf3_json(SAMPLE_IPERF3_JSON, "aa:bb:cc:dd:ee:02")
        
        # Add to database
        db.add(stats1)
        db.add(stats2)
        
        print(f"✓ Added 2 stats to database")
        print(f"  Count: {db.count()}")
        
        # Retrieve
        retrieved = db.get("aa:bb:cc:dd:ee:01")
        assert retrieved is not None, "Failed to retrieve stats"
        assert retrieved.sta_mac == "aa:bb:cc:dd:ee:01"
        
        print(f"✓ Retrieved stats successfully")
        
        # Test all()
        all_stats = db.all()
        assert len(all_stats) == 2, f"Expected 2 stats, got {len(all_stats)}"
        
        print(f"✓ all() returned {len(all_stats)} stations")
        
        # Test to_dict()
        dict_export = db.to_dict()
        assert "aa:bb:cc:dd:ee:01" in dict_export
        
        print(f"✓ to_dict() export successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transport_qoe_scoring():
    """Test TransportQoE scoring functions"""
    print("\n" + "="*60)
    print("TEST 3: TransportQoE Scoring Functions")
    print("="*60)
    
    try:
        # Test latency scoring
        assert TransportQoE.score_latency(15) == 100, "Excellent latency should score 100"
        assert TransportQoE.score_latency(35) == 85, "Good latency should score 85"
        assert TransportQoE.score_latency(250) == 10, "Poor latency should score 10"
        print("✓ Latency scoring correct")
        
        # Test reliability scoring
        assert TransportQoE.score_reliability(0.05) == 100, "Low retrans should score 100"
        assert TransportQoE.score_reliability(0.3) == 85, "Medium retrans should score 85"
        assert TransportQoE.score_reliability(10) == 10, "High retrans should score 10"
        print("✓ Reliability scoring correct")
        
        # Test throughput scoring
        assert TransportQoE.score_throughput(250) == 100, "High throughput should score 100"
        assert TransportQoE.score_throughput(75) == 80, "Medium throughput should score 80"
        assert TransportQoE.score_throughput(3) == 10, "Low throughput should score 10"
        print("✓ Throughput scoring correct")
        
        return True
        
    except AssertionError as e:
        print(f"✗ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_transport_qoe_computation():
    """Test overall QoE computation"""
    print("\n" + "="*60)
    print("TEST 4: TransportQoE Overall Computation")
    print("="*60)
    
    try:
        # Setup
        db = TransportStatsDB()
        db.clear()
        
        qoe_calc = TransportQoE()
        
        # Create sample stats
        stats = TransportStats.from_iperf3_json(SAMPLE_IPERF3_JSON, "aa:bb:cc:dd:ee:ff")
        db.add(stats)
        
        # Compute QoE
        components = qoe_calc.compute_qoe(stats)
        
        print(f"✓ Computed QoE components:")
        print(f"  Latency: {components.latency:.1f}")
        print(f"  Reliability: {components.reliability:.1f}")
        print(f"  Throughput: {components.throughput:.1f}")
        print(f"  Overall: {components.overall:.3f}")
        
        # Verify overall is in valid range
        assert 0.0 <= components.overall <= 1.0, f"Overall QoE {components.overall} out of range"
        
        # Verify components are reasonable
        assert components.latency > 0, "Latency score should be positive"
        assert components.reliability > 0, "Reliability score should be positive"
        assert components.throughput > 0, "Throughput score should be positive"
        
        print("✓ All QoE values in valid range")
        
        # Test update() method
        qoe_calc.update()
        qoe_score = qoe_calc.get_qoe("aa:bb:cc:dd:ee:ff")
        assert qoe_score is not None, "QoE score not found after update"
        assert qoe_score == components.overall, "QoE score mismatch"
        
        print(f"✓ update() method works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("TRANSPORT LAYER STATS - TEST SUITE")
    print("="*60)
    
    tests = [
        test_transport_stats_parsing,
        test_transport_stats_db,
        test_transport_qoe_scoring,
        test_transport_qoe_computation
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
