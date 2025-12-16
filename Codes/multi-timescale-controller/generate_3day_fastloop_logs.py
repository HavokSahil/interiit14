"""
3-Day RRM Simulation with Refactored Fast Loop Logging

Generates comprehensive logs for a 3-day simulation period with:
- Realistic day/night patterns
- Fast Loop TX power refinement and QoE correction
- EWMA baseline evolution
- Adaptive tolerance tracking
- Automatic rollback monitoring
- Complete audit trail

Output:
- simulation_logs_fastloop/ directory with detailed CSV logs
- fastloop_audit/ directory with fast loop specific audit
- Summary statistics and analytics
"""

import random
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
import csv

from assoc import *
from datatype import *
from metrics import *
from sim import *
from utils import *
from generate_training_data import create_grid_topology, create_random_topology
from enhanced_rrm_engine import EnhancedRRMEngine


class ThreeDayFastLoopSimulation:
    """Generate 3-day simulation logs with Fast Loop tracking"""
    
    def __init__(self, output_dir="simulation_logs_fastloop", audit_dir="fastloop_audit"):
        self.output_dir = Path(output_dir)
        self.audit_dir = Path(audit_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.audit_dir.mkdir(exist_ok=True)
        
        # Simulation parameters
        self.steps_per_hour = 360  # 10 seconds per step = 360 steps/hour
        self.hours_per_day = 24
        self.num_days = 3
        self.total_steps = self.steps_per_hour * self.hours_per_day * self.num_days
        
        # Statistics tracking
        self.stats = {
            'fast_loop': {
                'total_actions': 0,
                'successful_actions': 0,
                'rolled_back_actions': 0,
                'tx_power_actions': 0,
                'qoe_corrections': 0,
                'actions_by_hour': [0] * 24,
                'rollbacks_by_hour': [0] * 24,
                'actions_by_ap': {}
            },
            'event_loop': {
                'events_by_type': {},
                'events_by_hour': [0] * 24
            },
            'network': {
                'peak_clients': 0,
                'total_roams': 0,
                'config_changes': 0
            },
            'ewma_evolution': [],  # Track EWMA baseline evolution
            'rollback_details': []  # Detailed rollback information
        }
        
        # Fast Loop metrics log
        self.fastloop_log_file = self.output_dir / f"fastloop_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.fastloop_log = None
        
        print(f"\n{'='*70}")
        print("3-DAY FAST LOOP RRM SIMULATION")
        print(f"{'='*70}")
        print(f"Total steps: {self.total_steps:,}")
        print(f"Steps per hour: {self.steps_per_hour}")
        print(f"Duration: {self.num_days} days")
        print(f"Output: {self.output_dir}/")
        print(f"Audit: {self.audit_dir}/")
        print(f"{'='*70}\n")
    
    def get_hour_of_day(self, step):
        """Get hour of day (0-23) from step number"""
        total_hours = step / self.steps_per_hour
        return int(total_hours % 24)
    
    def get_day_number(self, step):
        """Get day number (0-2) from step number"""
        total_hours = step / self.steps_per_hour
        return int(total_hours / 24)
    
    def is_peak_hour(self, hour):
        """Determine if hour is peak (9am-5pm)"""
        return 9 <= hour < 17
    
    def is_night_hour(self, hour):
        """Determine if hour is night (10pm-6am)"""
        return hour >= 22 or hour < 6
    
    def get_client_count(self, step):
        """Get realistic client count based on time of day"""
        hour = self.get_hour_of_day(step)
        day = self.get_day_number(step)
        
        base_count = 20
        
        if self.is_peak_hour(hour):
            multiplier = 2.0
        elif self.is_night_hour(hour):
            multiplier = 0.3
        else:
            multiplier = 1.0
        
        if day == 2:  # Weekend
            multiplier *= 0.5
        
        multiplier *= random.uniform(0.8, 1.2)
        
        return int(base_count * multiplier)
    
    def inject_network_stress(self, step, sim):
        """Inject realistic network stress conditions"""
        hour = self.get_hour_of_day(step)
        
        # Simulate traffic bursts during peak hours
        if self.is_peak_hour(hour) and random.random() < 0.1:
            for ap in sim.access_points:
                # Increase CCA busy
                ap.cca_busy_percentage = min(0.95, ap.cca_busy_percentage + random.uniform(0.1, 0.3))
                
                # Degrade some client metrics
                for cid in ap.connected_clients:
                    client = next((c for c in sim.clients if c.id == cid), None)
                    if client and random.random() < 0.3:
                        # Reduce RSSI
                        client.rssi_dbm = max(-90, client.rssi_dbm - random.uniform(3, 8))
                        # Increase retry rate
                        client.retry_rate = min(35, client.retry_rate + random.uniform(5, 15))
    
    def adjust_client_count(self, sim, target_count):
        """Dynamically add or remove clients to match target"""
        current_count = len(sim.clients)
        
        if target_count > current_count:
            for _ in range(target_count - current_count):
                x = random.uniform(sim.env.x_min, sim.env.x_max)
                y = random.uniform(sim.env.y_min, sim.env.y_max)
                demand = random.uniform(5, 30)
                velocity = random.uniform(0.5, 2.0)
                
                client = Client(
                    id=len(sim.clients),
                    x=x, y=y,
                    demand_mbps=demand,
                    velocity=velocity,
                    rssi_dbm=random.uniform(-85, -60),
                    throughput_mbps=random.uniform(10, 80),
                    retry_rate=random.uniform(2, 12)
                )
                sim.add_client(client)
        
        elif target_count < current_count:
            clients_to_remove = current_count - target_count
            for _ in range(clients_to_remove):
                if sim.clients:
                    removed = sim.clients.pop()
                    if removed.associated_ap is not None:
                        ap = next((a for a in sim.access_points if a.id == removed.associated_ap), None)
                        if ap and removed.id in ap.connected_clients:
                            ap.connected_clients.remove(removed.id)
    
    def log_fast_loop_metrics(self, step, rrm, result):
        """Log Fast Loop specific metrics"""
        if self.fastloop_log is None:
            return
        
        # Get Fast Loop statistics
        if hasattr(rrm.fast_loop_engine, 'get_statistics'):
            stats = rrm.fast_loop_engine.get_statistics()
            
            # Get EWMA baselines for each AP
            for ap_id in rrm.aps.keys():
                state = rrm.fast_loop_engine.state_store
                
                row = {
                    'step': step,
                    'hour': self.get_hour_of_day(step),
                    'day': self.get_day_number(step),
                    'ap_id': ap_id,
                    'actions_executed': stats['actions_executed'],
                    'actions_succeeded': stats['actions_succeeded'],
                    'actions_rolled_back': stats['actions_rolled_back'],
                    'rollback_rate': stats['rollback_rate'],
                    'active_penalties': stats['active_penalties'],
                    'ewma_throughput': state.get(f'ewma_mean_throughput_mean_{ap_id}'),
                    'ewma_rssi': state.get(f'ewma_mean_median_rssi_{ap_id}'),
                    'ewma_retry': state.get(f'ewma_mean_retry_rate_{ap_id}'),
                    'ewma_cca': state.get(f'ewma_mean_cca_busy_percentage_{ap_id}'),
                    'var_throughput': state.get(f'ewma_var_throughput_mean_{ap_id}'),
                    'var_rssi': state.get(f'ewma_var_median_rssi_{ap_id}')
                }
                
                self.fastloop_log.writerow(row)
    
    def track_fast_loop_action(self, step, result):
        """Track Fast Loop actions from result"""
        hour = self.get_hour_of_day(step)
        
        if 'fast_loop' in result:
            for action_result in result['fast_loop']:
                ap_id = action_result.get('ap_id')
                action_type = action_result.get('action')
                status = action_result.get('result', {}).get('status')
                
                self.stats['fast_loop']['total_actions'] += 1
                self.stats['fast_loop']['actions_by_hour'][hour] += 1
                
                if ap_id not in self.stats['fast_loop']['actions_by_ap']:
                    self.stats['fast_loop']['actions_by_ap'][ap_id] = 0
                self.stats['fast_loop']['actions_by_ap'][ap_id] += 1
                
                # Print action details for visibility
                print(f"\n[Fast Loop] AP {ap_id} Action: {action_type} -> {status}")
                if status == 'acted_success':
                    self.stats['fast_loop']['successful_actions'] += 1
                    if action_type == 'tx_power_step':
                        self.stats['fast_loop']['tx_power_actions'] += 1
                        print(f"  Details: {action_result.get('result', {}).get('from_tx', '?')} -> {action_result.get('result', {}).get('to_tx', '?')} dBm")
                    elif action_type == 'qoe_correction':
                        self.stats['fast_loop']['qoe_corrections'] += 1
                
                elif status == 'attempted_tx_power':
                     # Count as QoE correction action
                     self.stats['fast_loop']['qoe_corrections'] += 1
                     self.stats['fast_loop']['successful_actions'] += 1
                     print(f"  Details: QoE Drop {action_result.get('result', {}).get('qoe_drop', 0):.1%}")

                elif status == 'rolled_back':
                    self.stats['fast_loop']['rolled_back_actions'] += 1
                    self.stats['fast_loop']['rollbacks_by_hour'][hour] += 1
                    print(f"  ROLLED BACK")
                    
                    # Track rollback details
                    self.stats['rollback_details'].append({
                        'step': step,
                        'hour': hour,
                        'ap_id': ap_id,
                        'action': action_type,
                        'result': action_result.get('result')
                    })
    
    def run(self):
        """Run 3-day simulation with Fast Loop tracking"""
        # Create environment
        env = Environment(x_min=0, x_max=100, y_min=0, y_max=100)
        
        # Create propagation model
        try:
            from model import PathLossModel, MultipathFadingModel
            base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=5.0)
            fading_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
        except ImportError:
            print("Warning: Could not import propagation models from 'model'. Trying 'propagation'...")
            try:
                from propagation import PathLossModel, MultipathFadingModel
                base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=5.0)
                fading_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
            except ImportError:
                print("Error: Could not import propagation models. Simulation will fail.")
                fading_model = None

        
        # Create simulation with logging
        sim = WirelessSimulation(
            env, fading_model,
            interference_threshold_dbm=-75.0,
            enable_logging=True,
            log_dir=str(self.output_dir)
        )
        
        # Create 6 APs with mixed channels
        print("Creating network topology...")
        N_ap = 6
        ap_positions = create_grid_topology(N_ap, env)
        ap_channels = [52, 6, 36, 11, 40, 1]
        
        for i, (x, y) in enumerate(ap_positions):
            channel = ap_channels[i]
            tx_power = random.uniform(18, 22)  # Varied power for Fast Loop to optimize
            bandwidth = 80 if channel > 14 else 20
            
            ap = AccessPoint(
                id=i, x=x, y=y,
                tx_power=tx_power,
                channel=channel,
                bandwidth=bandwidth,
                max_throughput=150.0,
                cca_busy_percentage=random.uniform(0.1, 0.4)  # Initial CCA
            )
            sim.add_access_point(ap)
        
        # Start with initial clients
        initial_clients = self.get_client_count(0)
        client_positions = create_random_topology(initial_clients, env)
        for i, (x, y) in enumerate(client_positions):
            demand = random.uniform(5, 30)
            velocity = random.uniform(0.5, 2.0)
            sim.add_client(Client(
                id=i, x=x, y=y,
                demand_mbps=demand,
                velocity=velocity,
                rssi_dbm=random.uniform(-85, -60),
                throughput_mbps=random.uniform(10, 80),
                retry_rate=random.uniform(2, 12)
            ))
        
        # Add interferers
        for i in range(2):
            x = random.uniform(env.x_min, env.x_max)
            y = random.uniform(env.y_min, env.y_max)
            interferer = Interferer(
                id=i, x=x, y=y,
                tx_power=random.uniform(25, 35),
                channel=random.choice([6, 11]),
                type=random.choice(["Microwave", "Bluetooth"]),
                duty_cycle=random.uniform(0.5, 0.8)
            )
            sim.add_interferer(interferer)
        
        # Initialize simulation
        sim.initialize()
        
        # Create Enhanced RRM Engine with Refactored Fast Loop
        print("Initializing Enhanced RRM Engine with Refactored Fast Loop...")
        rrm = EnhancedRRMEngine(
            access_points=sim.access_points,
            clients=sim.clients,
            interferers=sim.interferers,
            prop_model=fading_model,
            cooldown_steps=20,
            slow_loop_period=100,
            audit_log_dir=str(self.audit_dir)
        )
        
        # Initialize Fast Loop metrics CSV
        csv_file = open(self.fastloop_log_file, 'w', newline='')
        self.fastloop_log = csv.DictWriter(csv_file, fieldnames=[
            'step', 'hour', 'day', 'ap_id',
            'actions_executed', 'actions_succeeded', 'actions_rolled_back',
            'rollback_rate', 'active_penalties',
            'ewma_throughput', 'ewma_rssi', 'ewma_retry', 'ewma_cca',
            'var_throughput', 'var_rssi'
        ])
        self.fastloop_log.writeheader()
        
        print(f"Starting 3-day simulation ({self.total_steps:,} steps)...")
        print("This will take several minutes...\n")
        
        # Progress tracking
        last_report_step = 0
        report_interval = self.steps_per_hour
        
        start_time = time.time()
        
        # Main simulation loop
        for step in range(1, self.total_steps + 1):
            hour = self.get_hour_of_day(step)
            day = self.get_day_number(step)
            
            # Adjust client count based on time of day
            if step % (self.steps_per_hour // 6) == 0:  # Every 10 minutes
                target_clients = self.get_client_count(step)
                self.adjust_client_count(sim, target_clients)
                self.stats['network']['peak_clients'] = max(
                    self.stats['network']['peak_clients'], len(sim.clients)
                )
            
            # Inject network stress
            if step % (self.steps_per_hour // 2) == 0:  # Every 30 minutes
                self.inject_network_stress(step, sim)
            
            # Execute simulation step
            sim.step()
            
            # Execute RRM engine (includes Fast Loop)
            rrm_result = rrm.execute(step)
            
            # Track Fast Loop actions
            self.track_fast_loop_action(step, rrm_result)
            
            # Log Fast Loop metrics every hour
            if step % self.steps_per_hour == 0:
                self.log_fast_loop_metrics(step, rrm, rrm_result)
            
            # Track general statistics
            if 'event_action' in rrm_result:
                event_type = rrm_result.get('event_metadata', {}).get('event_type', 'unknown')
                self.stats['event_loop']['events_by_type'][event_type] = \
                    self.stats['event_loop']['events_by_type'].get(event_type, 0) + 1
                self.stats['event_loop']['events_by_hour'][hour] += 1
            
            if rrm_result.get('steering'):
                self.stats['network']['total_roams'] += len(rrm_result['steering'])
            
            # Progress reporting
            if step - last_report_step >= report_interval:
                elapsed = time.time() - start_time
                progress = (step / self.total_steps) * 100
                eta = (elapsed / step) * (self.total_steps - step)
                
                print(f"\r[Day {day+1}, Hour {hour:02d}:00] "
                      f"Step {step:,}/{self.total_steps:,} ({progress:.1f}%) | "
                      f"Clients: {len(sim.clients)} | "
                      f"Fast Loop Actions: {self.stats['fast_loop']['total_actions']} | "
                      f"Rollbacks: {self.stats['fast_loop']['rolled_back_actions']} | "
                      f"ETA: {eta/60:.1f}min", end='')
                
                last_report_step = step
        
        csv_file.close()
        
        print("\n\nSimulation complete!")
        
        # Generate summary report
        self.generate_summary_report(sim, rrm, start_time)
    
    def generate_summary_report(self, sim, rrm, start_time):
        """Generate comprehensive summary statistics"""
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("3-DAY FAST LOOP SIMULATION SUMMARY")
        print(f"{'='*70}")
        
        print(f"\nDuration: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
        print(f"Steps executed: {self.total_steps:,}")
        print(f"Steps per second: {self.total_steps/elapsed:.1f}")
        
        print(f"\nNetwork Statistics:")
        print(f"  Access Points: {len(sim.access_points)}")
        print(f"  Peak Clients: {self.stats['network']['peak_clients']}")
        print(f"  Total Client Roams: {self.stats['network']['total_roams']:,}")
        
        print(f"\nFast Loop Statistics:")
        print(f"  Total Actions: {self.stats['fast_loop']['total_actions']}")
        print(f"  Successful Actions: {self.stats['fast_loop']['successful_actions']}")
        print(f"  Rolled Back Actions: {self.stats['fast_loop']['rolled_back_actions']}")
        if self.stats['fast_loop']['total_actions'] > 0:
            rollback_rate = self.stats['fast_loop']['rolled_back_actions'] / self.stats['fast_loop']['total_actions']
            print(f"  Rollback Rate: {rollback_rate:.1%}")
        print(f"  TX Power Actions: {self.stats['fast_loop']['tx_power_actions']}")
        print(f"  QoE Corrections: {self.stats['fast_loop']['qoe_corrections']}")
        
        print(f"\nActions by AP:")
        for ap_id, count in sorted(self.stats['fast_loop']['actions_by_ap'].items()):
            print(f"  AP {ap_id}: {count} actions")
        
        print(f"\nEvent Loop Statistics:")
        total_events = sum(self.stats['event_loop']['events_by_type'].values())
        print(f"  Total Events: {total_events}")
        for event_type, count in sorted(self.stats['event_loop']['events_by_type'].items()):
            print(f"    {event_type}: {count}")
        
        print(f"\nFast Loop Actions by Hour:")
        for hour in range(24):
            actions = self.stats['fast_loop']['actions_by_hour'][hour]
            rollbacks = self.stats['fast_loop']['rollbacks_by_hour'][hour]
            bar = '█' * (actions // 5)
            print(f"  {hour:02d}:00 - {hour:02d}:59  Actions: {actions:4d}  Rollbacks: {rollbacks:3d}  {bar}")
        
        # Get final Fast Loop statistics
        if hasattr(rrm.fast_loop_engine, 'get_statistics'):
            final_stats = rrm.fast_loop_engine.get_statistics()
            print(f"\nFinal Fast Loop Engine State:")
            print(f"  Total Actions Executed: {final_stats['actions_executed']}")
            print(f"  Total Actions Succeeded: {final_stats['actions_succeeded']}")
            print(f"  Total Rollbacks: {final_stats['actions_rolled_back']}")
            print(f"  Final Rollback Rate: {final_stats['rollback_rate']:.1%}")
            print(f"  Active Penalties: {final_stats['active_penalties']}")
        
        # Generate JSON summary
        summary = {
            'simulation_duration_sec': elapsed,
            'total_steps': self.total_steps,
            'num_days': self.num_days,
            'network': self.stats['network'],
            'fast_loop': self.stats['fast_loop'],
            'event_loop': self.stats['event_loop'],
            'rollback_details': self.stats['rollback_details'][:100],  # First 100
            'generated_at': datetime.now().isoformat()
        }
        
        summary_file = self.output_dir / "fastloop_simulation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Logs saved to: {self.output_dir}/")
        print(f"Audit trail: {self.audit_dir}/")
        print(f"Summary: {summary_file}")
        print(f"Fast Loop Metrics: {self.fastloop_log_file}")
        print(f"{'='*70}\n")
        
        # Export audit trail
        if hasattr(rrm.event_loop, 'audit_logger'):
            audit_export = rrm.event_loop.audit_logger.export_audit_trail()
            print(f"Audit trail exported: {audit_export}")
        
        print("\n✓ 3-day Fast Loop RRM simulation complete!")


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("3-DAY FAST LOOP RRM SIMULATION")
    print("="*70)
    print("\nThis will generate comprehensive logs for a 3-day simulation.")
    print("Featuring:")
    print("  - Refactored Fast Loop Controller")
    print("  - EWMA baseline tracking")
    print("  - Adaptive tolerance evolution")
    print("  - Automatic rollback monitoring")
    print("  - TX power refinement and QoE correction")
    print("  - Complete audit trail")
    print("\nEstimated time: 12-18 minutes")
    print("Estimated disk space: ~100-150 MB")
    
    response = input("\nProceed? (y/n): ")
    
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    sim = ThreeDayFastLoopSimulation(
        output_dir="simulation_logs_fastloop",
        audit_dir="fastloop_audit"
    )
    
    sim.run()


if __name__ == "__main__":
    main()
