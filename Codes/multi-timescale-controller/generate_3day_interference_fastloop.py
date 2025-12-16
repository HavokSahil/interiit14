"""
3-Day RRM Simulation with Interference-Based Fast Loop Logging

Generates comprehensive logs for a 3-day simulation period with:
- Realistic day/night patterns
- Fast Loop interference-based optimization
- Channel changes, bandwidth adjustments, OBSS-PD tuning
- Interference graph analysis
- Complete audit trail

Output:
- simulation_logs_interference_fastloop/ directory with detailed CSV logs
- interference_fastloop_audit/ directory with fast loop specific audit
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


class ThreeDayInterferenceFastLoopSimulation:
    """Generate 3-day simulation logs with Interference-Based Fast Loop tracking"""
    
    def __init__(self, output_dir="simulation_logs_interference_fastloop", 
                 audit_dir="interference_fastloop_audit"):
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
                'channel_changes': 0,
                'bandwidth_increases': 0,
                'bandwidth_decreases': 0,
                'obss_pd_increases': 0,
                'obss_pd_decreases': 0,
                'actions_by_hour': [0] * 24,
                'actions_by_ap': {},
                'action_reasons': {}
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
            'interference_evolution': [],  # Track interference over time
            'action_details': []  # Detailed action information
        }
        
        # Fast Loop metrics log
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.fastloop_log_file = self.output_dir / f"interference_fastloop_metrics_{timestamp}.csv"
        self.actions_log_file = self.output_dir / f"interference_fastloop_actions_{timestamp}.csv"
        self.fastloop_log = None
        self.actions_log = None
        
        print(f"\n{'='*70}")
        print("3-DAY INTERFERENCE-BASED FAST LOOP RRM SIMULATION")
        print(f"{'='*70}")
        print(f"Total steps: {self.total_steps:,}")
        print(f"Steps per hour: {self.steps_per_hour}")
        print(f"Duration: {self.num_days} days")
        print(f"Fast Loop Period: 60 steps (10 minutes)")
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
            multiplier = 2.5  # More clients during peak
        elif self.is_night_hour(hour):
            multiplier = 0.3
        else:
            multiplier = 1.0
        
        if day == 2:  # Weekend
            multiplier *= 0.6
        
        multiplier *= random.uniform(0.8, 1.2)
        
        return int(base_count * multiplier)
    
    def inject_network_stress(self, step, sim):
        """Inject realistic network stress conditions to trigger Fast Loop"""
        hour = self.get_hour_of_day(step)
        
        # Simulate interference bursts during peak hours
        if self.is_peak_hour(hour) and random.random() < 0.15:
            # Increase interference and CCA busy
            for ap in sim.access_points:
                ap.cca_busy_percentage = min(0.95, ap.cca_busy_percentage + random.uniform(0.15, 0.35))
                
                # Degrade some client metrics
                for cid in ap.connected_clients:
                    client = next((c for c in sim.clients if c.id == cid), None)
                    if client and random.random() < 0.4:
                        client.rssi_dbm = max(-90, client.rssi_dbm - random.uniform(3, 10))
                        client.retry_rate = min(35, client.retry_rate + random.uniform(5, 20))
    
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
    
    def calculate_average_interference(self, graph):
        """Calculate average interference in the network"""
        if graph.number_of_edges() == 0:
            return 0.0
        
        total_weight = sum(data.get('weight', 0) for _, _, data in graph.edges(data=True))
        return total_weight / graph.number_of_edges()
    
    def log_fast_loop_metrics(self, step, rrm, interference_graph):
        """Log Fast Loop specific metrics"""
        if self.fastloop_log is None:
            return
        
        hour = self.get_hour_of_day(step)
        day = self.get_day_number(step)
        
        # Calculate network-wide interference
        avg_interference = self.calculate_average_interference(interference_graph)
        
        # Log metrics for each AP
        for ap_id, ap in rrm.aps.items():
            # Get AP's interference from graph
            interferers = list(interference_graph.predecessors(ap_id)) if ap_id in interference_graph else []
            total_interference = sum(
                interference_graph[i][ap_id].get('weight', 0.0)
                for i in interferers
            ) if ap_id in interference_graph else 0.0
            
            row = {
                'step': step,
                'hour': hour,
                'day': day,
                'ap_id': ap_id,
                'channel': ap.channel,
                'bandwidth': ap.bandwidth,
                'obss_pd_threshold': ap.obss_pd_threshold,
                'cca_busy_pct': ap.cca_busy_percentage * 100,
                'retry_rate': getattr(ap, 'p95_retry_rate', 0.0),
                'num_clients': len(ap.connected_clients),
                'total_interference': total_interference,
                'num_interferers': len(interferers),
                'avg_network_interference': avg_interference
            }
            
            self.fastloop_log.writerow(row)
    
    def track_fast_loop_action(self, step, result):
        """Track Fast Loop actions from result"""
        hour = self.get_hour_of_day(step)
        
        if 'fast_loop_actions' in result:
            for action_result in result['fast_loop_actions']:
                if not action_result.get('success'):
                    continue
                
                ap_id = action_result.get('ap_id')
                action_type = action_result.get('type')
                reason = action_result.get('reason', '')
                action = action_result.get('action', {})
                
                self.stats['fast_loop']['total_actions'] += 1
                self.stats['fast_loop']['actions_by_hour'][hour] += 1
                
                if ap_id not in self.stats['fast_loop']['actions_by_ap']:
                    self.stats['fast_loop']['actions_by_ap'][ap_id] = 0
                self.stats['fast_loop']['actions_by_ap'][ap_id] += 1
                
                # Track by reason
                if reason not in self.stats['fast_loop']['action_reasons']:
                    self.stats['fast_loop']['action_reasons'][reason] = 0
                self.stats['fast_loop']['action_reasons'][reason] += 1
                
                # Track by action type
                if action_type == 'channel_change':
                    self.stats['fast_loop']['channel_changes'] += 1
                    detail = f"Channel: {action.get('new_channel')}"
                elif action_type == 'bandwidth_increase':
                    self.stats['fast_loop']['bandwidth_increases'] += 1
                    detail = f"Bandwidth: {action.get('new_bandwidth')} MHz"
                elif action_type == 'bandwidth_reduce':
                    self.stats['fast_loop']['bandwidth_decreases'] += 1
                    detail = f"Bandwidth: {action.get('new_bandwidth')} MHz"
                elif action_type == 'obss_pd_increase':
                    self.stats['fast_loop']['obss_pd_increases'] += 1
                    detail = f"OBSS-PD: {action.get('new_obss_pd')} dBm"
                elif action_type == 'obss_pd_decrease':
                    self.stats['fast_loop']['obss_pd_decreases'] += 1
                    detail = f"OBSS-PD: {action.get('new_obss_pd')} dBm"
                else:
                    detail = str(action)
                
                # Print action details for visibility
                print(f"\n[Fast Loop] AP {ap_id} | {action_type} | {reason} | {detail}")
                
                # Log to actions CSV
                if self.actions_log:
                    self.actions_log.writerow({
                        'step': step,
                        'hour': hour,
                        'day': self.get_day_number(step),
                        'ap_id': ap_id,
                        'action_type': action_type,
                        'reason': reason,
                        'details': detail
                    })
                
                # Track detailed action
                self.stats['action_details'].append({
                    'step': step,
                    'hour': hour,
                    'ap_id': ap_id,
                    'action_type': action_type,
                    'reason': reason,
                    'action': action
                })
    
    def run(self):
        """Run 3-day simulation with Fast Loop tracking"""
        # Create environment
        env = Environment(x_min=0, x_max=100, y_min=0, y_max=100)
        
        # Create propagation model
        try:
            from model import PathLossModel, MultipathFadingModel
            base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=3.5)
            fading_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
        except ImportError:
            try:
                from propagation import PathLossModel, MultipathFadingModel
                base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=3.5)
                fading_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
            except ImportError:
                print("Error: Could not import propagation models.")
                return
        
        # Create simulation with logging
        sim = WirelessSimulation(
            env, fading_model,
            interference_threshold_dbm=-75.0,
            enable_logging=True,
            log_dir=str(self.output_dir)
        )
        
        # Create 6 APs with mixed channels (intentionally create interference)
        print("Creating network topology...")
        N_ap = 6
        ap_positions = create_grid_topology(N_ap, env)
        
        # Mix of 2.4 GHz and 5 GHz with overlapping channels
        ap_configs = [
            {'channel': 1, 'bandwidth': 20, 'tx_power': 20.0},   # 2.4 GHz
            {'channel': 6, 'bandwidth': 20, 'tx_power': 20.0},   # 2.4 GHz (overlaps with 1)
            {'channel': 36, 'bandwidth': 40, 'tx_power': 20.0},  # 5 GHz
            {'channel': 40, 'bandwidth': 80, 'tx_power': 20.0},  # 5 GHz (overlaps with 36)
            {'channel': 149, 'bandwidth': 20, 'tx_power': 20.0}, # 5 GHz
            {'channel': 11, 'bandwidth': 20, 'tx_power': 20.0}   # 2.4 GHz
        ]
        
        for i, (x, y) in enumerate(ap_positions):
            config = ap_configs[i]
            ap = AccessPoint(
                id=i, x=x, y=y,
                tx_power=config['tx_power'],
                channel=config['channel'],
                bandwidth=config['bandwidth'],
                max_throughput=150.0,
                cca_busy_percentage=random.uniform(0.2, 0.5),  # Start with moderate CCA
                obss_pd_threshold=-82  # Start conservative
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
        
        # Add interferers to create dynamic interference
        for i in range(3):
            x = random.uniform(env.x_min, env.x_max)
            y = random.uniform(env.y_min, env.y_max)
            interferer = Interferer(
                id=i, x=x, y=y,
                tx_power=random.uniform(25, 35),
                channel=random.choice([1, 6, 11, 36, 40]),
                type=random.choice(["Microwave", "Bluetooth", "ZigBee"]),
                duty_cycle=random.uniform(0.4, 0.8)
            )
            sim.add_interferer(interferer)
        
        # Initialize simulation
        sim.initialize()
        
        # Create Enhanced RRM Engine with Interference-Based Fast Loop
        print("Initializing Enhanced RRM Engine with Interference-Based Fast Loop...")
        rrm = EnhancedRRMEngine(
            access_points=sim.access_points,
            clients=sim.clients,
            interferers=sim.interferers,
            prop_model=fading_model,
            cooldown_steps=20,
            slow_loop_period=100,
            fast_loop_period=60,  # 10 minutes
            audit_log_dir=str(self.audit_dir)
        )
        
        # Initialize Fast Loop metrics CSV
        csv_file = open(self.fastloop_log_file, 'w', newline='')
        self.fastloop_log = csv.DictWriter(csv_file, fieldnames=[
            'step', 'hour', 'day', 'ap_id',
            'channel', 'bandwidth', 'obss_pd_threshold',
            'cca_busy_pct', 'retry_rate', 'num_clients',
            'total_interference', 'num_interferers', 'avg_network_interference'
        ])
        self.fastloop_log.writeheader()
        
        # Initialize Actions log CSV
        actions_file = open(self.actions_log_file, 'w', newline='')
        self.actions_log = csv.DictWriter(actions_file, fieldnames=[
            'step', 'hour', 'day', 'ap_id', 'action_type', 'reason', 'details'
        ])
        self.actions_log.writeheader()
        
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
            
            # Execute RRM engine (includes Fast Loop every 60 steps)
            rrm_result = rrm.execute(step)
            
            # Track Fast Loop actions
            self.track_fast_loop_action(step, rrm_result)
            
            # Log Fast Loop metrics every hour
            if step % self.steps_per_hour == 0:
                interference_graph = rrm.graph_builder.build_graph(list(rrm.aps.values()))
                self.log_fast_loop_metrics(step, rrm, interference_graph)
            
            # Track general statistics
            if 'event_action' in rrm_result:
                event_type = rrm_result.get('event_metadata', {}).get('event_type', 'unknown')
                self.stats['event_loop']['events_by_type'][event_type] = \
                    self.stats['event_loop']['events_by_type'].get(event_type, 0) + 1
                self.stats['event_loop']['events_by_hour'][hour] += 1
            
            # Progress reporting
            if step - last_report_step >= report_interval:
                elapsed = time.time() - start_time
                progress = (step / self.total_steps) * 100
                eta = (elapsed / step) * (self.total_steps - step)
                
                print(f"\r[Day {day+1}, Hour {hour:02d}:00] "
                      f"Step {step:,}/{self.total_steps:,} ({progress:.1f}%) | "
                      f"Clients: {len(sim.clients)} | "
                      f"Fast Loop Actions: {self.stats['fast_loop']['total_actions']} | "
                      f"ETA: {eta/60:.1f}min", end='')
                
                last_report_step = step
        
        csv_file.close()
        actions_file.close()
        
        print("\n\nSimulation complete!")
        
        # Generate summary report
        self.generate_summary_report(sim, rrm, start_time)
    
    def generate_summary_report(self, sim, rrm, start_time):
        """Generate comprehensive summary statistics"""
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("3-DAY INTERFERENCE-BASED FAST LOOP SIMULATION SUMMARY")
        print(f"{'='*70}")
        
        print(f"\nDuration: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
        print(f"Steps executed: {self.total_steps:,}")
        print(f"Steps per second: {self.total_steps/elapsed:.1f}")
        
        print(f"\nNetwork Statistics:")
        print(f"  Access Points: {len(sim.access_points)}")
        print(f"  Peak Clients: {self.stats['network']['peak_clients']}")
        
        print(f"\nFast Loop Statistics:")
        print(f"  Total Actions: {self.stats['fast_loop']['total_actions']}")
        print(f"  Channel Changes: {self.stats['fast_loop']['channel_changes']}")
        print(f"  Bandwidth Increases: {self.stats['fast_loop']['bandwidth_increases']}")
        print(f"  Bandwidth Decreases: {self.stats['fast_loop']['bandwidth_decreases']}")
        print(f"  OBSS-PD Increases: {self.stats['fast_loop']['obss_pd_increases']}")
        print(f"  OBSS-PD Decreases: {self.stats['fast_loop']['obss_pd_decreases']}")
        
        print(f"\nActions by AP:")
        for ap_id, count in sorted(self.stats['fast_loop']['actions_by_ap'].items()):
            print(f"  AP {ap_id}: {count} actions")
        
        print(f"\nActions by Reason:")
        for reason, count in sorted(self.stats['fast_loop']['action_reasons'].items()):
            print(f"  {reason}: {count}")
        
        print(f"\nFast Loop Actions by Hour:")
        for hour in range(24):
            actions = self.stats['fast_loop']['actions_by_hour'][hour]
            bar = '█' * (actions // 2)
            print(f"  {hour:02d}:00 - {hour:02d}:59  Actions: {actions:4d}  {bar}")
        
        # Get final Fast Loop statistics
        if hasattr(rrm.fast_loop_engine, 'get_statistics'):
            final_stats = rrm.fast_loop_engine.get_statistics()
            print(f"\nFinal Fast Loop Engine State:")
            print(f"  Channel Changes: {final_stats['channel_changes']}")
            print(f"  Bandwidth Changes: {final_stats['bandwidth_changes']}")
            print(f"  OBSS-PD Changes: {final_stats['obss_pd_changes']}")
            print(f"  Total Actions: {final_stats['total_actions']}")
        
        # Generate JSON summary
        summary = {
            'simulation_duration_sec': elapsed,
            'total_steps': self.total_steps,
            'num_days': self.num_days,
            'network': self.stats['network'],
            'fast_loop': self.stats['fast_loop'],
            'event_loop': self.stats['event_loop'],
            'action_details': self.stats['action_details'][:100],  # First 100
            'generated_at': datetime.now().isoformat()
        }
        
        summary_file = self.output_dir / "interference_fastloop_simulation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Logs saved to: {self.output_dir}/")
        print(f"Audit trail: {self.audit_dir}/")
        print(f"Summary: {summary_file}")
        print(f"Metrics: {self.fastloop_log_file}")
        print(f"Actions: {self.actions_log_file}")
        print(f"{'='*70}\n")
        
        print("\n✓ 3-day Interference-Based Fast Loop RRM simulation complete!")


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("3-DAY INTERFERENCE-BASED FAST LOOP RRM SIMULATION")
    print("="*70)
    print("\nThis will generate comprehensive logs for a 3-day simulation.")
    print("Featuring:")
    print("  - Interference-Based Fast Loop Controller")
    print("  - Channel optimization")
    print("  - Bandwidth adjustment (gradual)")
    print("  - OBSS-PD tuning")
    print("  - Interference graph analysis")
    print("  - Complete audit trail")
    print("\nEstimated time: 10-15 minutes")
    print("Estimated disk space: ~100 MB")
    
    response = input("\nProceed? (y/n): ")
    
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    sim = ThreeDayInterferenceFastLoopSimulation(
        output_dir="simulation_logs_interference_fastloop",
        audit_dir="interference_fastloop_audit"
    )
    
    sim.run()


if __name__ == "__main__":
    main()
