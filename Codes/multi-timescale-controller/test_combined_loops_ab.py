"""
A/B Test: No RRM vs Combined Loops (Fast + Event)
With Live Visualization and Event Injection
"""

import time
import numpy as np
import pygame
from datatype import Environment, AccessPoint, Client, Interferer, EventType
from model import PathLossModel, MultipathFadingModel
from sim import WirelessSimulation
from ab_testing import ABTestFramework, ABTestConfig
from qoe_monitor import QoEMonitor
from qoe_dashboard import QoEDashboard
from slo_catalog import SLOCatalog
from policy_engine import PolicyEngine
from enhanced_rrm_engine import EnhancedRRMEngine


def create_network():
    """Create test network topology"""
    # Environment
    env = Environment(x_min=0, x_max=200, y_min=0, y_max=200)
    
    # Propagation model
    base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=3.5)
    prop_model = MultipathFadingModel(base_model, fading_margin_db=6.0)
    
    # Access Points (High power for coverage)
    aps = [
        AccessPoint(id=0, x=50, y=50, tx_power=23.0, channel=1),
        AccessPoint(id=1, x=150, y=50, tx_power=23.0, channel=6),
        AccessPoint(id=2, x=50, y=150, tx_power=23.0, channel=11),
        AccessPoint(id=3, x=150, y=150, tx_power=23.0, channel=1),
    ]
    
    # Clients (mix of roles)
    clients = []
    for i in range(25):
        x = np.random.uniform(10, 190)
        y = np.random.uniform(10, 190)
        demand_mbps = np.random.uniform(5, 15)
        clients.append(Client(id=i, x=x, y=y, demand_mbps=demand_mbps))
    
    return env, prop_model, aps, clients


def run_variant(variant_config: ABTestConfig, steps: int, 
                slo_catalog: SLOCatalog, policy_engine: PolicyEngine):
    """Run simulation for one variant with visualization"""
    print(f"\n{'='*60}")
    print(f"Running Variant: {variant_config.variant_name}")
    print(f"RRM Mode: {variant_config.rrm_mode}")
    print(f"{'='*60}\n")
    
    # Create network
    env, prop_model, aps, clients = create_network()
    
    # Create simulation
    sim = WirelessSimulation(env, prop_model, enable_logging=False)
    
    for ap in aps:
        sim.add_access_point(ap)
    for client in clients:
        sim.add_client(client)
    
    sim.initialize()
    
    # Enable visualization
    if sim.enable_visualization(width=1920, height=1080):
        print("✓ Visualization enabled")
        print("Controls: P (Pause), ESC (Quit)")
    
    # Create QoE monitor
    qoe_monitor = QoEMonitor(slo_catalog, policy_engine)
    
    # Create RRM engine if needed
    rrm = None
    if variant_config.rrm_mode != "none":
        rrm = EnhancedRRMEngine(
            access_points=aps,
            clients=clients,
            interferers=[],
            prop_model=prop_model,
            slo_catalog_path="slo_catalog.yml",
            default_role="BE",
            fast_loop_period=60,   # 10 minutes
            slow_loop_period=300   # 50 minutes
        )
        
        # Configure loops based on variant
        if not variant_config.enable_fast_loop:
            rrm.fast_loop_engine = None
        if not variant_config.enable_slow_loop:
            rrm.slow_loop_engine = None
        # Event loop is always enabled in EnhancedRRMEngine but we can ignore its output if disabled
        # For this test, we assume if rrm_mode is 'full' or 'event_loop', we want it.
        
        print(f"[RRM] Initialized with Fast Loop: {variant_config.enable_fast_loop}, Event Loop: {variant_config.enable_event_loop}")

    # Run simulation
    start_time = time.time()
    actions_taken = 0
    
    # Visualization Loop
    if sim.visualizer:
        clock = pygame.time.Clock()
        running = True
        fps = 30  # Faster FPS for smooth viewing
        
        target_step = sim.step_count + steps
        
        while running and sim.step_count < target_step:
            # Handle events
            running = sim.visualizer.handle_events()
            if not running: break
            
            if not sim.visualizer.paused:
                sim.step()
                current_step = sim.step_count
                
                # INJECT INTERFERENCE EVENT at step 500 for Variant B
                if variant_config.rrm_mode != "none" and current_step == 500:
                    print("\n!!! INJECTING INTERFERENCE BURST !!!")
                    # Add a microwave interferer near AP 0
                    interferer = Interferer(
                        id=99, x=60, y=60, tx_power=30.0, channel=1, 
                        type="Microwave", bandwidth=20.0, duty_cycle=0.8
                    )
                    sim.add_interferer(interferer)
                    # Also notify RRM about it (simulating detection)
                    if rrm and rrm.sensing_api:
                        rrm.interferers.append(interferer)
                        print("[System] Interference injected and detected")

                # RRM Execution
                if rrm:
                    result = rrm.execute(current_step)
                    
                    # Count Fast Loop actions
                    if result.get('fast_loop_actions'):
                        count = len(result['fast_loop_actions'])
                        actions_taken += count
                        print(f"[Step {current_step}] Fast Loop Action: {count} changes")
                    
                    # Count Event Loop actions
                    if result.get('event_action'):
                        actions_taken += 1
                        print(f"[Step {current_step}] EVENT LOOP TRIGGERED: {result['event_action']}")
                
                # QoE Update
                if current_step % 10 == 0:
                    qoe_monitor.update(current_step, sim.clients, sim.access_points)
                
                # Progress
                if current_step % 200 == 0:
                    stats = qoe_monitor.get_network_qoe_stats()
                    print(f"Step {current_step}/{target_step}: QoE={stats.mean:.4f}")
            
            # Draw
            sim.visualizer.screen.fill((20, 20, 30))
            sim.visualizer.draw_coverage_areas()
            sim.visualizer.draw_association_lines()
            sim.visualizer.draw_access_points()
            sim.visualizer.draw_clients()
            sim.visualizer.draw_interferers()
            sim.visualizer.draw_sidebar()
            
            # Draw overlay text for variant
            font = pygame.font.SysFont("Arial", 24)
            text = font.render(f"Variant: {variant_config.variant_name}", True, (255, 255, 255))
            sim.visualizer.screen.blit(text, (20, 20))
            
            # Update UI
            time_delta = clock.tick(fps) / 1000.0
            sim.visualizer.ui_manager.update(time_delta)
            sim.visualizer.ui_manager.draw_ui(sim.visualizer.screen)
            
            pygame.display.flip()
            
        pygame.quit()
        
    else:
        print("Error: Visualization failed to initialize")
        return None

    runtime = time.time() - start_time
    final_stats = qoe_monitor.get_network_qoe_stats()
    
    return {
        'qoe_monitor': qoe_monitor,
        'final_qoe': final_stats.mean,
        'actions': actions_taken,
        'runtime': runtime,
        'clients': sim.clients
    }


def main():
    """Run A/B Test"""
    print("\n" + "="*70)
    print("A/B TEST: No RRM vs Combined Loops (Fast + Event)")
    print("="*70)
    
    slo_catalog = SLOCatalog("slo_catalog.yml")
    policy_engine = PolicyEngine(slo_catalog, default_role="BE")
    
    # Variant A: No RRM
    variant_a = ABTestConfig(
        variant_name="No RRM (Baseline)",
        rrm_mode="none"
    )
    
    # Variant B: Full RRM (Fast + Event)
    variant_b = ABTestConfig(
        variant_name="Combined Loops (Fast + Event)",
        rrm_mode="full"  # Enables Fast, Event, and Slow loops
    )
    
    ab_framework = ABTestFramework(variant_a, variant_b)
    test_steps = 2000  # Short run for demonstration
    
    # Run Variant A
    print("\n### STARTING VARIANT A (No RRM) ###")
    print("Close the window to proceed to Variant B...")
    result_a = run_variant(variant_a, test_steps, slo_catalog, policy_engine)
    
    if result_a:
        ab_framework.set_runtime("A", result_a['runtime'])
        for snapshot in result_a['qoe_monitor'].get_time_series():
            total_tput = sum(c.throughput_mbps for c in result_a['clients'])
            avg_retry = sum(c.retry_rate for c in result_a['clients']) / len(result_a['clients'])
            ab_framework.collect_metrics("A", snapshot.network_qoe, total_tput, avg_retry, 0)

    # Run Variant B
    print("\n### STARTING VARIANT B (Combined Loops) ###")
    print("Watch for Interference Injection at Step 500!")
    result_b = run_variant(variant_b, test_steps, slo_catalog, policy_engine)
    
    if result_b:
        ab_framework.set_runtime("B", result_b['runtime'])
        for snapshot in result_b['qoe_monitor'].get_time_series():
            total_tput = sum(c.throughput_mbps for c in result_b['clients'])
            avg_retry = sum(c.retry_rate for c in result_b['clients']) / len(result_b['clients'])
            ab_framework.collect_metrics("B", snapshot.network_qoe, total_tput, avg_retry, 0)
            
    # Comparison
    print("\n" + "="*70)
    print("COMPUTING COMPARISON...")
    print("="*70)
    
    comparison = ab_framework.compute_comparison()
    comparison.print_report()
    
    # Save results
    ab_framework.export_results("combined_loops_results.json", format="json")
    print("\nTest Complete!")


if __name__ == "__main__":
    main()
