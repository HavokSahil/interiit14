"""
Test script for Enhanced Event Loop.

Demonstrates:
- DFS radar event handling
- Non-WiFi interference detection
- Automatic rollback on degradation
- Audit trail generation
"""

from datetime import datetime
import time

from models.event_models import Event, EventType, Severity, SensingSource
from models.enhanced_event_loop import EnhancedEventLoop
from datatype import AccessPoint, Client, Interferer
from config_engine import ConfigEngine


def test_dfs_radar_event():
    """Test DFS radar detection and channel switch"""
    print("\n" + "="*70)
    print("TEST 1: DFS Radar Detection")
    print("="*70)
    
    # Create test APs
    aps = [
        AccessPoint(id=0, x=10, y=10, tx_power=23, channel=52, bandwidth=80),  # DFS channel
        AccessPoint(id=1, x=30, y=30, tx_power=23, channel=36, bandwidth=80),
    ]
    
    clients = []
    
    # Create config engine
    config_engine = ConfigEngine(aps)
    
    # Create event loop
    event_loop = EnhancedEventLoop(config_engine)
    
    # Create DFS radar event
    dfs_event = Event(
        event_id="dfs_001",
        event_type=EventType.DFS_RADAR,
        severity=Severity.CRITICAL,
        ap_id="ap_0",
        radio="5g",
        timestamp_utc=datetime.utcnow(),
        detection_confidence=1.0,  # Definitive radar detection
        metadata={
            'channel': 52,
            'pulse_width_us': 1.0,
            'repetition_interval_us': 1000
        },
        sensing_source=SensingSource.SERVING_RADIO
    )
    
    print(f"\nAP 0 before: Channel {aps[0].channel}")
    
    # Register event
    event_loop.register_event(dfs_event)
    
    # Execute event loop
    result = event_loop.execute(step=100, access_points=aps, clients=clients)
    
    print(f"AP 0 after: Channel {aps[0].channel}")
    
    # Check rollback token created
    stats = event_loop.get_statistics()
    print(f"\nRollback tokens created: {stats['rollback_stats']['total_tokens']}")
    print(f"Audit records: {stats['audit_stats']['total_records']}")
    
    # Print status
    event_loop.print_status()
    
    return event_loop, aps


def test_interference_burst():
    """Test non-WiFi interference burst handling"""
    print("\n" + "="*70)
    print("TEST 2: Non-WiFi Interference Burst")
    print("="*70)
    
    # Create test APs
    aps = [
        AccessPoint(id=0, x=10, y=10, tx_power=23, channel=6, bandwidth=20),
        AccessPoint(id=1, x=30, y=30, tx_power=23, channel=11, bandwidth=20),
    ]
    
    # Create interferer (microwave)
    interferers = [
        Interferer(id=0, x=12, y=12, tx_power=30, channel=6, 
                  type="Microwave", duty_cycle=0.8)
    ]
    
    clients = []
    
    config_engine = ConfigEngine(aps)
    event_loop = EnhancedEventLoop(config_engine)
    
    # Create interference burst event
    interference_event = Event(
        event_id="intf_001",
        event_type=EventType.NON_WIFI_BURST,
        severity=Severity.HIGH,
        ap_id="ap_0",
        radio="2g",
        timestamp_utc=datetime.utcnow(),
        detection_confidence=0.85,
        metadata={
            'interferer_type': 'Microwave',
            'interferer_channel': 6,
            'duty_cycle_pct': 80,
            'center_freq': 2.437  # Channel 6
        },
        sensing_source=SensingSource.ADDITIONAL_RADIO
    )
    
    print(f"\nAP 0 before: Channel {aps[0].channel}")
    print(f"Interferer on channel 6 with 80% duty cycle")
    
    event_loop.register_event(interference_event)
    result = event_loop.execute(step=200, access_points=aps, clients=clients, 
                               interferers=interferers)
    
    print(f"AP 0 after: Channel {aps[0].channel}")
    
    event_loop.print_status()
    
    return event_loop, aps


def test_automatic_rollback():
    """Test automatic rollback on degradation"""
    print("\n" + "="*70)
    print("TEST 3: Automatic Rollback on Degradation")
    print("="*70)
    
    aps = [
        AccessPoint(id=0, x=10, y=10, tx_power=23, channel=6, bandwidth=20,
                    p95_retry_rate=5.0)  # Initial retry rate 5%
    ]
    
    clients = []
    
    config_engine = ConfigEngine(aps)
    event_loop = EnhancedEventLoop(config_engine, monitoring_window_sec=10)
    
    # Create and handle an event
    event = Event(
        event_id="test_rollback",
        event_type=EventType.SPECTRUM_SAT,
        severity=Severity.HIGH,
        ap_id="ap_0",
        radio="2g",
        timestamp_utc=datetime.utcnow(),
        detection_confidence=0.9,
        metadata={'cca_busy_pct': 96}
    )
    
    print(f"\nAP 0 before: OBSS-PD = {aps[0].obss_pd_threshold} dBm")
    
    event_loop.register_event(event)
    event_loop.execute(step=300, access_points=aps, clients=clients)
    
    print(f"AP 0 after action: OBSS-PD = {aps[0].obss_pd_threshold} dBm")
    
    # Simulate degradation (retry rate spikes)
    print("\nSimulating network degradation (retry rate spike)...")
    time.sleep(2)
    aps[0].p95_retry_rate = 15.0  # Spiked to 15% (3x increase = >30% threshold)
    
    # Run monitoring check
    event_loop._check_monitoring(step=310, access_points=aps, clients=clients)
    
    print(f"AP 0 after rollback: OBSS-PD = {aps[0].obss_pd_threshold} dBm")
    
    stats = event_loop.get_statistics()
    print(f"\nRollbacks triggered: {stats['rollbacks_triggered']}")
    
    event_loop.print_status()
    
    return event_loop, aps


def test_audit_trail_export():
    """Test audit trail export"""
    print("\n" + "="*70)
    print("TEST 4: Audit Trail Export")
    print("="*70)
    
    # Run a few events
    event_loop, aps = test_dfs_radar_event()
    
    # Export audit trail
    export_path = event_loop.audit_logger.export_audit_trail(ap_id="ap_0")
    
    print(f"\nAudit trail exported to: {export_path}")
    
    # Read and verify
    with open(export_path, 'r') as f:
        lines = f.readlines()
        print(f"Total audit records exported: {len(lines)}")
    
    return event_loop


def main():
    """Run all tests"""
    print("\n" + "#"*70)
    print("# Enhanced Event Loop - Test Suite")
    print("#"*70)
    
    # Test 1: DFS radar
    test_dfs_radar_event()
    
    print("\n" + "-"*70 + "\n")
    time.sleep(1)
    
    # Test 2: Interference burst
    test_interference_burst()
    
    print("\n" + "-"*70 + "\n")
    time.sleep(1)
    
    # Test 3: Automatic rollback
    test_automatic_rollback()
    
    print("\n" + "-"*70 + "\n")
    time.sleep(1)
    
    # Test 4: Audit export
    test_audit_trail_export()
    
    print("\n" + "#"*70)
    print("# All tests completed!")
    print("#"*70)


if __name__ == "__main__":
    main()
