"""
Example A/B Test: No RRM vs Fast Loop

This script demonstrates how to run an A/B test comparing:
- Variant A: No RRM (baseline)
- Variant B: Fast Loop enabled

The test will show QoE improvements from RRM optimization.
"""

import time
import numpy as np
import pygame
from datatype import Environment, AccessPoint, Client, Interferer
from model import PathLossModel, MultipathFadingModel
from sim import WirelessSimulation
from ab_testing import ABTestFramework, ABTestConfig
from qoe_monitor import QoEMonitor
from qoe_dashboard import QoEDashboard
from slo_catalog import SLOCatalog
from policy_engine import PolicyEngine
from enhanced_rrm_engine import EnhancedRRMEngine

from generate_training_data import *
import random

def generate_topology(num_aps=6, num_clients=25):
    """Generate static topology configuration for consistent A/B testing"""
    env = Environment(x_min=0, x_max=50, y_min=0, y_max=50)
    
    # Generate AP configs
    ap_configs = []
    ap_positions = create_linear_topology(num_aps, env)
    for i, (x, y) in enumerate(ap_positions):
        ap_configs.append({
            'id': i, 'x': x, 'y': y,
            'channel': random.choice([1, 6, 11]),
            'tx_power': random.uniform(20, 30),
            'bandwidth': 20.0,
            'max_throughput': 150.0
        })
    
    # Generate Client configs
    client_configs = []
    client_positions = create_linear_topology(num_clients, env)
    for i, (x, y) in enumerate(client_positions):
        client_configs.append({
            'id': i, 'x': x, 'y': y,
            'demand_mbps': random.uniform(5, 30),
            'velocity': random.uniform(0.5, 2.0)
        })
        
    return env, ap_configs, client_configs


def create_network_from_config(env, ap_configs, client_configs):
    """Instantiate network objects from configuration"""
    # Propagation model
    base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=5.0)
    prop_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
    
    # Create objects
    aps = [AccessPoint(**conf) for conf in ap_configs]
    clients = [Client(**conf) for conf in client_configs]
    
    return env, prop_model, aps, clients


def run_variant(variant_config: ABTestConfig, steps: int, 
                slo_catalog: SLOCatalog, policy_engine: PolicyEngine,
                topology_data: tuple):
    """
    Run simulation for one variant using provided topology.
    """
    print(f"\n{'='*60}")
    print(f"Running Variant: {variant_config.variant_name}")
    print(f"RRM Mode: {variant_config.rrm_mode}")
    print(f"{'='*60}\n")
    
    # Unpack and instantiate fresh network
    env_config, ap_configs, client_configs = topology_data
    env, prop_model, aps, clients = create_network_from_config(env_config, ap_configs, client_configs)
    
    # Create simulation with logging enabled
    sim = WirelessSimulation(env, prop_model, enable_logging=True, log_dir=f"logs_{variant_config.variant_name.replace(' ', '_')}")
    
    for ap in aps:
        sim.add_access_point(ap)
    for client in clients:
        sim.add_client(client)
    
    sim.initialize()
    
    # Enable visualization for live view
    if sim.enable_visualization(width=1920, height=1080):
        print("✓ Visualization enabled - you can see the simulation running!")
    
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
            fast_loop_period=60,  # 10 minutes
            slow_loop_period=300  # 50 minutes
        )
        # Disable loops we don't want
        if not variant_config.enable_fast_loop:
            rrm.fast_loop_engine = None
        if not variant_config.enable_slow_loop:
            rrm.slow_loop_engine = None

    
    # Run simulation
    start_time = time.time()
    actions_taken = 0
    
    # Check if visualization is enabled
    if sim.visualizer:
        # Run with visualization (this handles the step loop internally)
        print(f"Running {steps} steps with live visualization...")
        print("Press 'P' to pause, 'ESC' to quit early")
        
        # Store original step count
        original_step = sim.step_count
        target_step = original_step + steps
        
        # Create a custom run loop that integrates RRM and QoE monitoring
        clock = pygame.time.Clock()
        running = True
        fps = 10
        
        while running and sim.step_count < target_step:
            # Process events using visualizer's handler (handles UI buttons)
            running = sim.visualizer.handle_events()
            
            if not running:
                break
            
            if not sim.visualizer.paused:
                # Simulation step
                sim.step()
                current_step = sim.step_count
                
                # INJECT INTERFERENCE EVENT at step 500 for RRM variants
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
                
                # RRM optimization (if enabled)
                if rrm:
                    result = rrm.execute(current_step)
                    if result.get('fast_loop_actions'):
                        actions_taken += len(result['fast_loop_actions'])
                    if result.get('event_action'):
                        actions_taken += 1
                        print(f"[Step {current_step}] EVENT LOOP TRIGGERED: {result['event_action']}")
                
                # Update QoE (every 10 steps)
                if current_step % 10 == 0:
                    qoe_monitor.update(current_step, sim.clients, sim.access_points)
                
                # Progress indicator (every 500 steps)
                if current_step % 500 == 0:
                    stats = qoe_monitor.get_network_qoe_stats()
                    print(f"Step {current_step}/{target_step}: QoE={stats.mean:.4f}, Actions={actions_taken}")
            
            # Update visualization
            sim.visualizer.screen.fill((20, 20, 30))
            sim.visualizer.draw_coverage_areas()
            sim.visualizer.draw_association_lines()
            sim.visualizer.draw_access_points()
            sim.visualizer.draw_clients()
            sim.visualizer.draw_interferers()
            sim.visualizer.draw_sidebar()
            
            # Update GUI
            time_delta = clock.tick(fps) / 1000.0
            sim.visualizer.ui_manager.update(time_delta)
            sim.visualizer.ui_manager.draw_ui(sim.visualizer.screen)
            
            pygame.display.flip()
        
        pygame.quit()
        
    else:
        # Run without visualization (original loop)
        for step in range(1, steps + 1):
            # Simulation step
            sim.step()
            
            # INJECT INTERFERENCE EVENT at step 500 for RRM variants
            if variant_config.rrm_mode != "none" and step == 500:
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
            
            # RRM optimization (if enabled)
            if rrm:
                result = rrm.execute(step)
                if result.get('fast_loop_actions'):
                    actions_taken += len(result['fast_loop_actions'])
                if result.get('event_action'):
                    actions_taken += 1
                    print(f"[Step {step}] EVENT LOOP TRIGGERED: {result['event_action']}")
            
            # Update QoE (every 10 steps)
            if step % 10 == 0:
                qoe_monitor.update(step, sim.clients, sim.access_points)
            
            # Progress indicator
            if step % 500 == 0:
                stats = qoe_monitor.get_network_qoe_stats()
                print(f"Step {step}/{steps}: QoE={stats.mean:.4f}, Actions={actions_taken}")

    
    runtime = time.time() - start_time
    
    # Final stats
    final_stats = qoe_monitor.get_network_qoe_stats()
    
    print(f"\nCompleted in {runtime:.2f}s")
    print(f"Final QoE: {final_stats.mean:.4f}")
    print(f"Actions Taken: {actions_taken}")
    
    return {
        'qoe_monitor': qoe_monitor,
        'final_qoe': final_stats.mean,
        'actions': actions_taken,
        'runtime': runtime,
        'clients': sim.clients
    }


def main():
    """Run A/B test"""
    print("\n" + "="*70)
    print("A/B TEST: No RRM vs Fast Loop")
    print("="*70)
    
    # Load SLO catalog
    slo_catalog = SLOCatalog("slo_catalog.yml")
    policy_engine = PolicyEngine(slo_catalog, default_role="BE")
    
    # Assign some clients to different roles for testing
    # (Would be done properly in real system)
    
    # Define variants
    variant_a = ABTestConfig(
        variant_name="No RRM (Baseline)",
        rrm_mode="none"
    )
    
    variant_b = ABTestConfig(
        variant_name="Combined Loops (Fast + Event)",
        rrm_mode="full"
    )
    
    # Create A/B framework
    ab_framework = ABTestFramework(variant_a, variant_b)
    
    # Generate consistent topology for both variants
    print("Generating network topology...")
    topology_data = generate_topology(num_aps=6, num_clients=25)
    
    test_steps = 100000  # 100,000 steps
    
    # Run Variant A
    print("\n### VARIANT A ###")
    result_a = run_variant(variant_a, test_steps, slo_catalog, policy_engine, topology_data)
    ab_framework.set_runtime("A", result_a['runtime'])
    
    # Collect metrics from Variant A
    for snapshot in result_a['qoe_monitor'].get_time_series():
        # Calculate network-wide metrics
        total_tput = sum(c.throughput_mbps for c in result_a['clients'])
        avg_retry = sum(c.retry_rate for c in result_a['clients']) / len(result_a['clients']) if result_a['clients'] else 0
        
        ab_framework.collect_metrics(
            variant="A",
            qoe=snapshot.network_qoe,
            throughput=total_tput,
            retry_rate=avg_retry,
            interference=0  # Would track from interference graph
        )
    
    # Run Variant B
    print("\n### VARIANT B ###")
    result_b = run_variant(variant_b, test_steps, slo_catalog, policy_engine, topology_data)
    ab_framework.set_runtime("B", result_b['runtime'])
    
    # Collect metrics from Variant B
    for snapshot in result_b['qoe_monitor'].get_time_series():
        total_tput = sum(c.throughput_mbps for c in result_b['clients'])
        avg_retry = sum(c.retry_rate for c in result_b['clients']) / len(result_b['clients']) if result_b['clients'] else 0
        
        ab_framework.collect_metrics(
            variant="B",
            qoe=snapshot.network_qoe,
            throughput=total_tput,
            retry_rate=avg_retry,
            interference=0
        )
    
    # Compute and display comparison
    print("\n" + "="*70)
    print("COMPUTING COMPARISON...")
    print("="*70)
    
    comparison = ab_framework.compute_comparison()
    comparison.print_report()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    dashboard = QoEDashboard()
    
    # Save plots
    dashboard.save_report("ab_test_results", result_b['qoe_monitor'], ab_framework)
    
    # Export results
    ab_framework.export_results("ab_test_results.json", format="json")
    ab_framework.export_results("ab_test_results.csv", format="csv")
    
    print("\n✓ A/B test complete!")
    print(f"\nKey findings:")
    print(f"  QoE Improvement: {comparison.qoe_improvement_pct:+.2f}%")
    print(f"  Statistically Significant: {'YES' if comparison.qoe_significant else 'NO'}")
    print(f"  Fast Loop Actions: {comparison.variant_b_actions}")
    
    # Show dashboard
    print("\nShowing dashboard... (close window to exit)")
    dashboard.show_dashboard(result_b['qoe_monitor'], ab_framework)


if __name__ == "__main__":
    main()
