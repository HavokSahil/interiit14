"""
Enhanced Simulation with Event Loop Integration.

Demonstrates:
- RRM Engine with Enhanced Event Loop
- DFS event injection
- Interference event handling
- Automatic rollback
- Audit trail generation
"""

from assoc import *
from datatype import *
from metrics import *
from sim import *
from utils import *
from generate_training_data import create_grid_topology, create_random_topology
from enhanced_rrm_engine import EnhancedRRMEngine


def main():
    """Run enhanced simulation with Event Loop"""
    print("\n" + "="*70)
    print("ENHANCED SIMULATION with Event Loop Integration")
    print("="*70)
    
    # Create environment
    env = Environment(x_min=0, x_max=50, y_min=0, y_max=50)
    
    # Create propagation model
    base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=5.0)
    fading_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
    
    # Create simulation
    sim = WirelessSimulation(env, fading_model, 
                            interference_threshold_dbm=-75.0, 
                            enable_logging=True)
    
    # Create APs (some on DFS channels for testing)
    N_ap = 4
    ap_positions = create_random_topology(N_ap, env)
    ap_channels = [52, 6, 36, 11]  # Mix of DFS (52) and non-DFS channels
    
    for i, (x, y) in enumerate(ap_positions):
        channel = ap_channels[i] if i < len(ap_channels) else 6
        tx_power = random.uniform(20, 23)
        bandwidth = 80 if channel > 14 else 20  # 5 GHz gets 80 MHz
        
        ap = AccessPoint(
            id=i, x=x, y=y, 
            tx_power=tx_power, 
            channel=channel,
            bandwidth=bandwidth, 
            max_throughput=150.0
        )
        sim.add_access_point(ap)
        print(f"AP {i}: Channel {channel}, Power {tx_power:.1f} dBm, Pos ({x:.1f}, {y:.1f})")
    
    # Create clients
    N_client = 15
    client_positions = create_random_topology(N_client, env)
    for i, (x, y) in enumerate(client_positions):
        demand_mbps = random.uniform(5, 20)
        velocity = random.uniform(0.5, 2.0)
        sim.add_client(Client(id=i, x=x, y=y, demand_mbps=demand_mbps, velocity=velocity))
    
    # Add interferers
    interferer = Interferer(
        id=0, x=25, y=25, 
        tx_power=30, 
        channel=6, 
        type="Microwave",
        duty_cycle=0.7
    )
    sim.add_interferer(interferer)
    print(f"Interferer: Microwave on Channel 6 (duty cycle 70%)")
    
    # Initialize simulation
    sim.initialize()
    
    # ========== CREATE ENHANCED RRM ENGINE ==========
    rrm = EnhancedRRMEngine(
        access_points=sim.access_points,
        clients=sim.clients,
        interferers=sim.interferers,
        prop_model=fading_model,
        cooldown_steps=20,  # 20 steps cooldown
        audit_log_dir="audit_logs"
    )
    
    print("\n" + "="*70)
    print("RRM Engine initialized with Enhanced Event Loop")
    print("="*70)
    
    # Run simulation
    SIMULATION_STEPS = 100
    
    print(f"\nRunning simulation for {SIMULATION_STEPS} steps...")
    print("\nEvent injection schedule:")
    print(f"  Step 10: DFS radar on AP 0 (Channel 52)")
    print(f"  Step 30: Interference burst on AP 1 (Channel 6)")
    print(f"  Step 50: Spectrum saturation on AP 3 (Channel 11)")
    print()
    
    for step in range(1, SIMULATION_STEPS + 1):
        # Execute simulation step
        sim.step()
        
        # ========== INJECT EVENTS FOR TESTING ==========
        
        # Step 10: DFS radar detected on AP 0
        if step == 10:
            print(f"\n{'='*70}")
            print(f"STEP {step}: Injecting DFS RADAR event on AP 0")
            print(f"{'='*70}")
            rrm.inject_dfs_event(ap_id=0, channel=52)
        
        # Step 30: Interference burst on AP 1
        if step == 30:
            print(f"\n{'='*70}")
            print(f"STEP {step}: Injecting INTERFERENCE BURST event on AP 1")
            print(f"{'='*70}")
            rrm.inject_interference_event(ap_id=1, interferer_type="Microwave")
        
        # Step 50: Spectrum saturation on AP 3
        if step == 50:
            print(f"\n{'='*70}")
            print(f"STEP {step}: Injecting SPECTRUM SATURATION event on AP 3")
            print(f"{'='*70}")
            rrm.inject_spectrum_saturation_event(ap_id=3, cca_busy_pct=96)
        
        # ========== EXECUTE RRM ENGINE ==========
        rrm_result = rrm.execute(step)
        
        # Print event actions
        if 'event_action' in rrm_result:
            print(f"\n  ⚡ EVENT ACTION EXECUTED")
            print(f"  Metadata: {rrm_result['event_metadata']}")
            if 'event_loop_stats' in rrm_result:
                stats = rrm_result['event_loop_stats']
                print(f"  Event Loop Stats: {stats['events_processed']} events, "
                      f"{stats['actions_executed']} actions, "
                      f"{stats['rollbacks_triggered']} rollbacks")
        
        # Print periodic status
        if step % 20 == 0:
            print(f"\n{'='*70}")
            print(f"STEP {step} - Network Status")
            print(f"{'='*70}")
            for ap in sim.access_points:
                print(f"  AP {ap.id}: Ch={ap.channel}, "
                      f"Power={ap.tx_power:.1f}dBm, "
                      f"OBSS-PD={ap.obss_pd_threshold:.1f}dBm, "
                      f"Clients={len(ap.connected_clients)}")
    
    # Print final status
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    
    print("\n" + "="*70)
    print("FINAL NETWORK STATE")
    print("="*70)
    sim.print_status()
    
    print("\n" + "="*70)
    print("RRM ENGINE FINAL STATUS")
    print("="*70)
    rrm.print_status()
    
    # Export audit trail
    print("\n" + "="*70)
    print("AUDIT TRAIL")
    print("="*70)
    audit_path = rrm.event_loop.audit_logger.export_audit_trail()
    print(f"Audit trail exported to: {audit_path}")
    
    # Print some audit records
    audit_records = rrm.event_loop.audit_logger.recent_records
    print(f"\nTotal audit records: {len(audit_records)}")
    
    if audit_records:
        print("\nRecent audit records:")
        for i, record in enumerate(audit_records[-5:]):  # Last 5 records
            print(f"\n  Record {i+1}:")
            print(f"    Event Type: {record.event.event_type if record.event else 'N/A'}")
            print(f"    AP: {record.ap_id}")
            print(f"    Action: {record.action_type}")
            print(f"    Status: {record.execution_status}")
            print(f"    Reason: {record.reason}")
            if record.configuration_changes:
                for change in record.configuration_changes:
                    print(f"    Change: {change.param} "
                          f"{change.old_value} → {change.new_value}")
    
    # Rollback statistics
    rollback_stats = rrm.event_loop.rollback_manager.get_statistics()
    print("\n" + "="*70)
    print("ROLLBACK STATISTICS")
    print("="*70)
    print(f"Total Rollback Tokens: {rollback_stats['total_tokens']}")
    print(f"Active Tokens: {rollback_stats['active_tokens']}")
    print(f"Total Rollbacks: {rollback_stats['total_rollbacks']}")
    print(f"  - Automatic: {rollback_stats['auto_rollbacks']}")
    print(f"  - Manual: {rollback_stats['manual_rollbacks']}")
    
    print("\n" + "="*70)
    print("✓ Enhanced simulation complete with Event Loop integration!")
    print("="*70)


if __name__ == "__main__":
    main()
