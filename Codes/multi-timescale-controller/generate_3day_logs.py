"""
3-Day Simulation Log Generator

Generates comprehensive simulation logs spanning 3 days with:
- Realistic day/night patterns
- Peak/off-peak hour traffic
- Random event injection (DFS, interference, failures)
- Client mobility and roaming
- Network optimization cycles

Output:
- simulation_logs_3day/ directory with CSV logs
- audit_logs_3day/ directory with audit trail
- Summary statistics and analytics
"""

import random
import time
from datetime import datetime, timedelta
from pathlib import Path
import json

from assoc import *
from datatype import *
from metrics import *
from sim import *
from utils import *
from generate_training_data import create_grid_topology, create_random_topology
from enhanced_rrm_engine import EnhancedRRMEngine


class ThreeDaySimulation:
    """Generate 3-day realistic simulation logs"""
    
    def __init__(self, output_dir="simulation_logs_3day", audit_dir="audit_logs_3day"):
        self.output_dir = Path(output_dir)
        self.audit_dir = Path(audit_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.audit_dir.mkdir(exist_ok=True)
        
        # Simulation parameters
        self.steps_per_hour = 360  # 10 seconds per step = 360 steps/hour
        self.hours_per_day = 24
        self.num_days = 200
        self.total_steps = self.steps_per_hour * self.hours_per_day * self.num_days
        
        # Statistics tracking
        self.stats = {
            'events_by_type': {},
            'events_by_hour': [0] * 24,
            'actions_executed': 0,
            'rollbacks_triggered': 0,
            'peak_clients': 0,
            'total_roams': 0
        }
        
        print(f"\n{'='*70}")
        print("3-DAY SIMULATION LOG GENERATOR")
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
        
        # Base client count
        base_count = 20
        
        # Peak hours (9am-5pm): more clients
        if self.is_peak_hour(hour):
            multiplier = 2.0
        # Night hours (10pm-6am): fewer clients
        elif self.is_night_hour(hour):
            multiplier = 0.3
        # Normal hours
        else:
            multiplier = 1.0
        
        # Weekend effect (day 2 = Saturday)
        if day == 2:  # Weekend
            multiplier *= 0.5
        
        # Add some randomness
        multiplier *= random.uniform(0.8, 1.2)
        
        return int(base_count * multiplier)
    
    def inject_random_events(self, step, rrm, sim):
        """Inject random events based on probability"""
        hour = self.get_hour_of_day(step)
        
        # DFS radar events (very rare, 0.1% per hour)
        # Only inject if there are 5 GHz APs (channel > 14)
        if random.random() < 0.001:
            five_ghz_aps = [ap.id for ap in sim.access_points if ap.channel > 14]
            if five_ghz_aps:  # Check if list is not empty
                ap_id = random.choice(five_ghz_aps)
                rrm.inject_dfs_event(ap_id, sim.access_points[ap_id].channel)
                self.stats['events_by_type']['dfs_radar'] = \
                    self.stats['events_by_type'].get('dfs_radar', 0) + 1
                self.stats['events_by_hour'][hour] += 1
        
        # Interference bursts (more common during peak hours)
        interference_prob = 0.02 if self.is_peak_hour(hour) else 0.005
        if random.random() < interference_prob:
            if sim.access_points:  # Check if APs exist
                ap_id = random.choice([ap.id for ap in sim.access_points])
                rrm.inject_interference_event(ap_id, random.choice(["Microwave", "Bluetooth", "Zigbee"]))
                self.stats['events_by_type']['interference'] = \
                    self.stats['events_by_type'].get('interference', 0) + 1
                self.stats['events_by_hour'][hour] += 1
        
        # Spectrum saturation (during peak hours)
        if self.is_peak_hour(hour) and random.random() < 0.01:
            if sim.access_points:  # Check if APs exist
                ap_id = random.choice([ap.id for ap in sim.access_points])
                cca_busy = random.uniform(92, 98)
                rrm.inject_spectrum_saturation_event(ap_id, cca_busy)
                self.stats['events_by_type']['spectrum_sat'] = \
                    self.stats['events_by_type'].get('spectrum_sat', 0) + 1
                self.stats['events_by_hour'][hour] += 1
    
    def adjust_client_count(self, sim, target_count):
        """Dynamically add or remove clients to match target"""
        current_count = len(sim.clients)
        
        if target_count > current_count:
            # Add clients
            for _ in range(target_count - current_count):
                x = random.uniform(sim.env.x_min, sim.env.x_max)
                y = random.uniform(sim.env.y_min, sim.env.y_max)
                demand = random.uniform(5, 30)
                velocity = random.uniform(0.5, 2.0)
                
                client = Client(
                    id=len(sim.clients),
                    x=x, y=y,
                    demand_mbps=demand,
                    velocity=velocity
                )
                sim.add_client(client)
        
        elif target_count < current_count:
            # Remove clients (remove from end)
            clients_to_remove = current_count - target_count
            for _ in range(clients_to_remove):
                if sim.clients:
                    removed = sim.clients.pop()
                    if removed.associated_ap is not None:
                        ap = next((a for a in sim.access_points if a.id == removed.associated_ap), None)
                        if ap and removed.id in ap.connected_clients:
                            ap.connected_clients.remove(removed.id)
    
    def run(self):
        """Run 3-day simulation"""
        # Create environment
        env = Environment(x_min=0, x_max=100, y_min=0, y_max=100)
        
        # Create propagation model
        base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=5.0)
        fading_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
        
        # Create simulation with logging
        log_filename = f"3day_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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
        ap_channels = [52, 6, 36, 11, 40, 1]  # Mix of 2.4G and 5G
        
        for i, (x, y) in enumerate(ap_positions):
            channel = ap_channels[i]
            tx_power = random.uniform(20, 23)
            bandwidth = 80 if channel > 14 else 20
            
            ap = AccessPoint(
                id=i, x=x, y=y,
                tx_power=tx_power,
                channel=channel,
                bandwidth=bandwidth,
                max_throughput=150.0
            )
            sim.add_access_point(ap)
        
        # Start with initial clients
        initial_clients = self.get_client_count(0)
        client_positions = create_random_topology(initial_clients, env)
        for i, (x, y) in enumerate(client_positions):
            demand = random.uniform(5, 30)
            velocity = random.uniform(0.5, 2.0)
            sim.add_client(Client(id=i, x=x, y=y, demand_mbps=demand, velocity=velocity))
        
        # Add interferers
        for i in range(2):
            x = random.uniform(env.x_min, env.x_max)
            y = random.uniform(env.y_min, env.y_max)
            interferer = Interferer(
                id=i,
                x=x, y=y,
                tx_power=random.uniform(25, 35),
                channel=random.choice([6, 11]),
                type=random.choice(["Microwave", "Bluetooth"]),
                duty_cycle=random.uniform(0.5, 0.8)
            )
            sim.add_interferer(interferer)
        
        # Initialize simulation
        sim.initialize()
        
        # Create Enhanced RRM Engine
        rrm = EnhancedRRMEngine(
            access_points=sim.access_points,
            clients=sim.clients,
            interferers=sim.interferers,
            prop_model=fading_model,
            cooldown_steps=20,
            audit_log_dir=str(self.audit_dir)
        )
        
        print(f"Starting 3-day simulation ({self.total_steps:,} steps)...")
        print("This will take several minutes...\n")
        
        # Progress tracking
        last_report_step = 0
        report_interval = self.steps_per_hour  # Report every hour
        
        start_time = time.time()
        
        # Main simulation loop
        for step in range(1, self.total_steps + 1):
            # Get current time context
            hour = self.get_hour_of_day(step)
            day = self.get_day_number(step)
            
            # Adjust client count based on time of day
            if step % (self.steps_per_hour // 6) == 0:  # Every 10 minutes
                target_clients = self.get_client_count(step)
                self.adjust_client_count(sim, target_clients)
                self.stats['peak_clients'] = max(self.stats['peak_clients'], len(sim.clients))
            
            # Execute simulation step
            sim.step()
            
            # Inject random events
            self.inject_random_events(step, rrm, sim)
            
            # Execute RRM engine
            rrm_result = rrm.execute(step)
            
            # Track statistics
            if 'event_action' in rrm_result:
                self.stats['actions_executed'] += 1
            
            if rrm_result.get('steering'):
                self.stats['total_roams'] += len(rrm_result['steering'])
            
            # Progress reporting
            if step - last_report_step >= report_interval:
                elapsed = time.time() - start_time
                progress = (step / self.total_steps) * 100
                eta = (elapsed / step) * (self.total_steps - step)
                
                print(f"\r[Day {day+1}, Hour {hour:02d}:00] "
                      f"Step {step:,}/{self.total_steps:,} ({progress:.1f}%) | "
                      f"Clients: {len(sim.clients)} | "
                      f"Events: {sum(self.stats['events_by_type'].values())} | "
                      f"Actions: {self.stats['actions_executed']} | "
                      f"ETA: {eta/60:.1f}min", end='')
                
                last_report_step = step
        
        print("\n\nSimulation complete!")
        
        # Update final statistics
        rrm_stats = rrm.event_loop.get_statistics()
        self.stats['rollbacks_triggered'] = rrm_stats['rollbacks_triggered']
        
        # Generate summary report
        self.generate_summary_report(sim, rrm, start_time)
    
    def generate_summary_report(self, sim, rrm, start_time):
        """Generate summary statistics and report"""
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("3-DAY SIMULATION SUMMARY")
        print(f"{'='*70}")
        
        print(f"\nDuration: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
        print(f"Steps executed: {self.total_steps:,}")
        print(f"Steps per second: {self.total_steps/elapsed:.1f}")
        
        print(f"\nNetwork Statistics:")
        print(f"  Access Points: {len(sim.access_points)}")
        print(f"  Peak Clients: {self.stats['peak_clients']}")
        print(f"  Total Client Roams: {self.stats['total_roams']:,}")
        
        print(f"\nEvent Statistics:")
        total_events = sum(self.stats['events_by_type'].values())
        print(f"  Total Events: {total_events}")
        for event_type, count in sorted(self.stats['events_by_type'].items()):
            print(f"    {event_type}: {count}")
        
        print(f"\nRRM Statistics:")
        print(f"  Actions Executed: {self.stats['actions_executed']}")
        print(f"  Rollbacks Triggered: {self.stats['rollbacks_triggered']}")
        
        rrm_stats = rrm.event_loop.get_statistics()
        print(f"  Event Loop Stats:")
        print(f"    Events Processed: {rrm_stats['events_processed']}")
        print(f"    Active Monitoring: {rrm_stats['active_monitoring']}")
        
        print(f"\nEvents by Hour of Day:")
        for hour in range(24):
            bar = '█' * (self.stats['events_by_hour'][hour] // 5)
            print(f"  {hour:02d}:00 - {hour:02d}:59  {self.stats['events_by_hour'][hour]:4d}  {bar}")
        
        # Generate JSON summary
        summary = {
            'simulation_duration_sec': elapsed,
            'total_steps': self.total_steps,
            'num_days': self.num_days,
            'network': {
                'access_points': len(sim.access_points),
                'peak_clients': self.stats['peak_clients'],
                'total_roams': self.stats['total_roams']
            },
            'events': {
                'total': total_events,
                'by_type': self.stats['events_by_type'],
                'by_hour': self.stats['events_by_hour']
            },
            'rrm': {
                'actions_executed': self.stats['actions_executed'],
                'rollbacks_triggered': self.stats['rollbacks_triggered'],
                'events_processed': rrm_stats['events_processed']
            },
            'generated_at': datetime.now().isoformat()
        }
        
        summary_file = self.output_dir / "simulation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Logs saved to: {self.output_dir}/")
        print(f"Audit trail: {self.audit_dir}/")
        print(f"Summary: {summary_file}")
        print(f"{'='*70}\n")
        
        # Export audit trail
        audit_export = rrm.event_loop.audit_logger.export_audit_trail()
        print(f"Audit trail exported: {audit_export}")
        
        print("\n✓ 3-day simulation log generation complete!")


import sys
def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("3-DAY SIMULATION LOG GENERATOR")
    print("="*70)
    print("\nThis will generate comprehensive logs for a 3-day simulation period.")
    print("Including:")
    print("  - Realistic day/night client patterns")
    print("  - Peak/off-peak hour variations")
    print("  - Random event injection (DFS, interference, etc.)")
    print("  - Complete audit trail")
    print("\nEstimated time: 10-15 minutes")
    print("Estimated disk space: ~50-100 MB")
    
    response = input("\nProceed? (y/n): ")
    
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    sim = ThreeDaySimulation(
        output_dir="simulation_logs_3day",
        audit_dir="audit_logs_3day"
    )
    
    sim.run()


if __name__ == "__main__":
    main()
