"""
Test SimpleHast Loop with 10-minute periodicity in Enhanced RRM Engine
"""

import sys
sys.path.insert(0, '/home/sahil/Work/interiit14/Codes/multi-timescale-controller')

from datatype import Environment, AccessPoint, Client, Interferer
from model import PathLossModel, MultipathFadingModel
from enhanced_rrm_engine import EnhancedRRMEngine
import random

def test_simple_fast_loop():
    """Test that simplified Fast Loop runs every 10 minutes (60 steps)"""
    
    print("\n" + "="*70)
    print("TESTING SIMPLE FAST LOOP WITH 10-MINUTE PERIODICITY")
    print("="*70)
    
    # Create environment
    env = Environment(x_min=0, x_max=100, y_min=0, y_max=100)
    
    # Create propagation model
    base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=3.5)
    prop_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
    
    # Create 3 APs
    access_points = []
    for i in range(3):
        ap = AccessPoint(
            id=i,
            x=25 + i * 25,
            y=50,
            tx_power=20.0,
            channel=[1, 6, 11][i],
            bandwidth=20
        )
        access_points.append(ap)
    
    # Create 10 clients
    clients = []
    for i in range(10):
        client = Client(
            id=i,
            x=random.uniform(0, 100),
            y=random.uniform(0, 100),
            demand_mbps=random.uniform(10, 30),
            velocity=random.uniform(0.5, 2.0),
            rssi_dbm=random.uniform(-80, -60),
            throughput_mbps=random.uniform(20, 80),
            retry_rate=random.uniform(2, 10)
        )
        clients.append(client)
    
    # Create RRM Engine with Fast Loop period = 60 steps (10 minutes)
    print("\nInitializing Enhanced RRM Engine...")
    rrm = EnhancedRRMEngine(
        access_points=access_points,
        clients=clients,
        interferers=[],
        prop_model=prop_model,
        slow_loop_period=100,
        fast_loop_period=60,  # 10 minutes
        cooldown_steps=20,
        audit_log_dir="test_simple_fast_loop_audit"
    )
    
    print(f"Fast Loop Period: {rrm.fast_loop_period} steps")
    print(f"Expected execution at steps: 60, 120, 180, ...")
    
    # Run simulation and track when Fast Loop executes
    fast_loop_executions = []
    
    print("\nRunning 200 steps...")
    for step in range(1, 201):
        result = rrm.execute(step)
        
        # Check if Fast Loop executed (steering actions present)
        if 'steering' in result and result['steering']:
            fast_loop_executions.append(step)
            print(f"\n[Step {step}] Fast Loop executed!")
            print(f"  Steering actions: {len(result['steering'])}")
            for action in result['steering']:
                print(f"    Client {action['client_id']}: AP{action['old_ap']} → AP{action['new_ap']}")
        
        # Progress indicator
        if step % 20 == 0:
            print(f"Step {step}/200 completed")
    
    # Verify periodicity
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total steps run: 200")
    print(f"Fast Loop executions: {fast_loop_executions}")
    print(f"Expected executions at: [60, 120, 180]")
    
    # Check if execution happened at correct intervals
    if len(fast_loop_executions) >= 3:
        intervals = [fast_loop_executions[i] - fast_loop_executions[i-1] 
                    for i in range(1, len(fast_loop_executions))]
        print(f"Intervals between executions: {intervals}")
        
        # All intervals should be 60
        if all(interval == 60 for interval in intervals):
            print("✅ PASS: Fast Loop runs exactly every 60 steps")
        else:
            print("❌ FAIL: Intervals are not consistent")
    else:
        print("⚠️  WARNING: Not enough executions to verify periodicity")
    
    # Print Fast Loop statistics
    if hasattr(rrm.fast_loop_engine, 'get_statistics'):
        stats = rrm.fast_loop_engine.get_statistics()
        print(f"\nFast Loop Statistics:")
        print(f"  Total steers: {stats.get('total_steers', 0)}")
        print(f"  QoE threshold: {stats.get('qoe_threshold', 0)}")
        print(f"  RSSI threshold: {stats.get('rssi_threshold', 0)} dBm")
    
    print("\n✓ Test complete!\n")


if __name__ == "__main__":
    test_simple_fast_loop()
