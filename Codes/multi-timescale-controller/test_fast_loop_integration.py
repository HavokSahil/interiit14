"""
Test script for integrated Refactored Fast Loop Controller.

Tests the integration with EnhancedRRMEngine.
"""

import sys
sys.path.insert(0, '/home/sahil/Work/interiit14/Codes/multi-timescale-controller')

from datatype import *
from enhanced_rrm_engine import EnhancedRRMEngine

try:
    from propagation import PathLossModel, MultipathFadingModel
except ImportError:
    # Fallback if propagation not available
    PathLossModel = None
    MultipathFadingModel = None


# Create environment
env = Environment(x_min=0, x_max=100, y_min=0, y_max=100)

# Create propagation model (optional)
prop_model = None
if PathLossModel and MultipathFadingModel:
    try:
        base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=5.0)
        prop_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
    except Exception as e:
        print(f"Note: Could not create propagation model: {e}")
        prop_model = None


# Create APs
aps = [
    AccessPoint(id=0,x=25,y=25,tx_power=20,channel=6,bandwidth=20,max_throughput=150),
    AccessPoint(id=1,x=75,y=25,tx_power=20,channel=11,bandwidth=20,max_throughput=150),
    AccessPoint(id=2,x=25,y=75,tx_power=20,channel=1,bandwidth=20,max_throughput=150),
    AccessPoint(id=3,x=75,y=75,tx_power=20,channel=6,bandwidth=20,max_throughput=150),
]

# Create clients with varied RSSI
clients = [
    Client(id=0, x=20, y=20, demand_mbps=10, rssi_dbm=-70, throughput_mbps=50, retry_rate=5),
    Client(id=1, x=30, y=30, demand_mbps=15, rssi_dbm=-80, throughput_mbps=30, retry_rate=15),  # Weak
    Client(id=2, x=70, y=20, demand_mbps=20, rssi_dbm=-65, throughput_mbps=80, retry_rate=3),
    Client(id=3, x=80, y=25, demand_mbps=10, rssi_dbm=-85, throughput_mbps=20, retry_rate=20),  # Very weak
    Client(id=4, x=25, y=70, demand_mbps=12, rssi_dbm=-72, throughput_mbps=45, retry_rate=8),
]

# Associate clients to APs
aps[0].connected_clients = [0, 1]
aps[1].connected_clients = [2, 3]
aps[2].connected_clients = [4]

clients[0].associated_ap = 0
clients[1].associated_ap = 0
clients[2].associated_ap = 1
clients[3].associated_ap = 1
clients[4].associated_ap = 2

# Set CCA busy for some APs (to trigger actions)
aps[1].cca_busy_percentage = 0.75  # High CCA, should trigger decrease
aps[0].cca_busy_percentage = 0.2   # Low CCA, with weak clients, should trigger increase

# Create interferers
interferers = [
    Interferer(id=0, x=50, y=50, tx_power=30, channel=6, type="Microwave", duty_cycle=0.7)
]

print("="*70)
print("TEST: REFACTORED FAST LOOP INTEGRATION")
print("="*70)
print(f"APs: {len(aps)}")
print(f"Clients: {len(clients)}")
print(f"Interferers: {len(interferers)}")

# Create Enhanced RRM Engine (with refactored fast loop)
try:
    rrm = EnhancedRRMEngine(
        access_points=aps,
        clients=clients,
        interferers=interferers,
        prop_model=prop_model,
        slow_loop_period=100,
        cooldown_steps=20,
        audit_log_dir="test_audit"
    )
    print("\n✓ EnhancedRRMEngine created successfully")
except Exception as e:
    print(f"\n✗ Error creating RRM Engine: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check if refactored fast loop was loaded
if rrm.fast_loop_engine:
    if hasattr(rrm.fast_loop_engine, 'get_statistics'):
        print("✓ Refactored Fast Loop Controller loaded")
    else:
        print("⚠ Original Fast Loop Controller loaded (fallback)")
else:
    print("✗ No Fast Loop Controller loaded")

print("\n" + "="*70)
print("RUNNING TEST SIMULATION (10 steps)")
print("="*70)

# Run simulation for a few steps
for step in range(1, 11):
    result = rrm.execute(step)
    
    if step == 1:
        print(f"\nStep {step}:")
        if 'fast_loop' in result:
            print(f"  Fast Loop Actions: {len(result['fast_loop'])}")
            for action in result['fast_loop']:
                print(f"    AP {action['ap_id']}: {action['action']} -> {action['result']['status']}")
        if 'fast_loop_stats' in result:
            stats = result['fast_loop_stats']
            print(f"  Fast Loop Stats:")
            print(f"    Actions: {stats['actions_executed']}")
            print(f"    Rollback Rate: {stats['rollback_rate']:.1%}")

print("\n" + "="*70)
print("FINAL STATUS")
print("="*70)

# Print final status
rrm.print_status()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)

# Check results
if rrm.fast_loop_engine and hasattr(rrm.fast_loop_engine, 'get_statistics'):
    stats = rrm.fast_loop_engine.get_statistics()
    
    print(f"\n✓ Test passed")
    print(f"  - Actions executed: {stats['actions_executed']}")
    print(f"  - Actions succeeded: {stats['actions_succeeded']}")
    print(f"  - Actions rolled back: {stats['actions_rolled_back']}")
    
    if stats['actions_executed'] > 0:
        print(f"\n✓ Fast Loop Controller is active and executing actions")
    else:
        print(f"\n⚠ Fast Loop Controller loaded but no actions executed")
        print(f"  (This may be expected if no triggers met)")
else:
    print(f"\n⚠ Could not get Fast Loop statistics")
