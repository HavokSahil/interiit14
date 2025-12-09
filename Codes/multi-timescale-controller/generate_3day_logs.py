"""
3-Day Simulation Log Generator

Generates comprehensive simulation logs spanning 3 days with:
- Realistic day/night patterns
- Peak/off-peak hour traffic
- Random event injection (DFS, interference, failures)
- Client mobility and roaming
- Network optimization cycles

Output:
- 3_day_logs/state_logs/ - CSV logs for APs, Clients, etc.
- 3_day_logs/audit/ - Audit trail for all RRM actions
- 3_day_logs/analysis/ - Plots and summary statistics
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
from generate_training_data import create_grid_topology, create_random_topology, create_linear_topology
from enhanced_rrm_engine import EnhancedRRMEngine


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class ThreeDaySimulation:
    """Generate 3-day realistic simulation logs"""
    
    def __init__(self, root_dir="3_day_logs", enable_rrm=True, environment="education"):
        self.root_dir = Path(root_dir)
        self.state_dir = self.root_dir / "state_logs"
        self.audit_dir = self.root_dir / "audit"
        self.analysis_dir = self.root_dir / "analysis"
        self.environment = environment
        
        # Create directories
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Simulation parameters
        self.steps_per_hour = 60
        self.hours_per_day = 24
        self.num_days = 3
        self.total_steps = self.steps_per_hour * self.hours_per_day * self.num_days
        
        # Simulated time: start from a random time 4 days ago
        self.sim_start_time = datetime.now() - timedelta(
            days=4,
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )

        self.seconds_per_step = 60
        
        # RRM control switch
        self.enable_rrm = enable_rrm
        
        # Statistics tracking
        self.stats = {
            'events_by_type': {},
            'events_by_hour': [0] * 24,
            'actions_executed': 0,
            'rollbacks_triggered': 0,
            'peak_clients': 0,
            'total_roams': 0,
            'client_counts': [],
            'qoe_values': [],
            # Track all loop actions
            'fast_loop_actions': 0,
            'event_loop_actions': 0,
            'slow_loop_actions': 0
        }
        
        # Comprehensive audit log
        self.audit_log = []
        
        # Active interferers with expiration tracking
        # {interferer_id: {'interferer': Interferer, 'expires_at_step': int}}
        self.active_interferers = {}
        self.next_interferer_id = 1000  # Start ID for dynamic interferers
        
        print(f"\n{'='*70}")
        print("3-DAY SIMULATION LOG GENERATOR")
        print(f"{'='*70}")
        print(f"Total steps: {self.total_steps:,}")
        print(f"Duration: {self.num_days} days")
        print(f"Output Directory: {self.root_dir}/")
        print(f"  ├─ State Logs: {self.state_dir}/")
        print(f"  ├─ Audit Logs: {self.audit_dir}/")
        print(f"  └─ Analysis:   {self.analysis_dir}/")
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
        data = ""
        if self.environment == "office":
            data = '''
| 00:00 | 5       |
| 01:00 | 4       |
| 02:00 | 4       |
| 03:00 | 3       |
| 04:00 | 3       |
| 05:00 | 8       |
| 06:00 | 30      |
| 07:00 | 120     |
| 08:00 | 350     |
| 09:00 | 680     |
| 10:00 | 900     |
| 11:00 | 1050    |  
| 12:00 | 1100    |  
| 13:00 | 1150    |
| 14:00 | 1200    |
| 15:00 | 1180    |
| 16:00 | 1000    |
| 17:00 | 750     |
| 18:00 | 400     |
| 19:00 | 150     |
| 20:00 | 40      |
| 21:00 | 15      |
| 22:00 | 10      |
| 23:00 | 7       |
'''
        elif self.environment == "education":
            data = '''
| 00:00 | 40      |
| 01:00 | 35      |
| 02:00 | 30      |
| 03:00 | 25      |
| 04:00 | 20      |
| 05:00 | 25      |
| 06:00 | 80      |
| 07:00 | 200     |
| 08:00 | 600     |
| 09:00 | 900     |
| 10:00 | 1200    |
| 11:00 | 1500    |
| 12:00 | 1700    |
| 13:00 | 1750    |
| 14:00 | 1800    |
| 15:00 | 1650    |
| 16:00 | 1400    |
| 17:00 | 1000    |
| 18:00 | 700     |
| 19:00 | 500     |
| 20:00 | 300     |
| 21:00 | 180     |
| 22:00 | 120     |
| 23:00 | 80      |
'''
        
        # Base client count
        #base_count = 20
        base_count = int(data.split("|\n")[hour].split("|")[-1].strip())
        
        # Peak hours (9am-5pm): more clients
        #if self.is_peak_hour(hour):
        #    multiplier = 2.0
        # Night hours (10pm-6am): fewer clients
        #elif self.is_night_hour(hour):
        #    multiplier = 0.3
        # Normal hours
        #else:
        #    multiplier = 1.0
        multiplier = 1.0
        
        # Weekend effect (day 2 = Saturday)
        if day == 2:  # Weekend
            multiplier *= 0.5
        
        # Add some randomness
        multiplier *= random.uniform(0.8, 1.2)
        
        return int(base_count * multiplier)
    
    def inject_random_events(self, step, rrm, sim):
        """Inject random events based on probability - creates REAL interferers"""
        hour = self.get_hour_of_day(step)
        
        # First, remove expired interferers
        self._cleanup_expired_interferers(step, sim)
        
        # NOTE: DFS radar events removed - only using 2.4GHz channels (1, 6, 11)
        # DFS only applies to 5GHz channels

        
        # Interference bursts: ~5-10 per day during peak, ~2-3 otherwise
        # Peak: 0.003 per step ≈ 0.18/hour ≈ 1.4/peak period
        # Off-peak: 0.0005 per step
        interference_prob = 0.002 if self.is_peak_hour(hour) else 0.0003
        if random.random() < interference_prob:
            if sim.access_points:
                ap_id = random.choice([ap.id for ap in sim.access_points])
                ap = next((a for a in sim.access_points if a.id == ap_id), None)
                if ap:
                    interferer_type = random.choice(["Microwave", "Bluetooth", "Zigbee"])
                    
                    # Create REAL interferer near the AP
                    interferer = self._create_real_interferer(ap, interferer_type, step, sim)
                    
                    # Also notify RRM engine (if enabled, it will react)
                    rrm.inject_interference_event(ap_id, interferer_type)
                    
                    self.stats['events_by_type']['interference'] = \
                        self.stats['events_by_type'].get('interference', 0) + 1
                    self.stats['events_by_hour'][hour] += 1
                    self._log_audit_entry(step, 'EVENT_INJECTION', 'interference', ap_id, {
                        'type': interferer_type,
                        'interferer_id': interferer.id,
                        'position': (interferer.x, interferer.y),
                        'channel': interferer.channel,
                        'tx_power': interferer.tx_power
                    })
        
        # Spectrum saturation: ~3-5 per day during peak only
        # 0.001 per step ≈ 0.06/hour ≈ 0.5/peak period
        if self.is_peak_hour(hour) and random.random() < 0.001:
            if sim.access_points:
                ap_id = random.choice([ap.id for ap in sim.access_points])
                cca_busy = random.uniform(92, 98)
                rrm.inject_spectrum_saturation_event(ap_id, cca_busy)
                self.stats['events_by_type']['spectrum_sat'] = \
                    self.stats['events_by_type'].get('spectrum_sat', 0) + 1
                self.stats['events_by_hour'][hour] += 1
                self._log_audit_entry(step, 'EVENT_INJECTION', 'spectrum_saturation', ap_id, {'cca_busy': cca_busy})

    def _create_real_interferer(self, ap, interferer_type: str, step: int, sim) -> 'Interferer':
        """
        Create a real Interferer object that impacts network physics.
        
        The interferer is placed near the affected AP and lasts for 5-15 minutes.
        """
        # Interferer characteristics based on type
        interferer_configs = {
            'Microwave': {'tx_power': 25.0, 'duty_cycle': 0.5, 'bandwidth': 60.0},
            'Bluetooth': {'tx_power': 10.0, 'duty_cycle': 0.3, 'bandwidth': 2.0},
            'Zigbee': {'tx_power': 5.0, 'duty_cycle': 0.1, 'bandwidth': 2.0}  # 0 dBm = 1mW
        }
        config = interferer_configs.get(interferer_type, {'tx_power': 10.0, 'duty_cycle': 0.3, 'bandwidth': 10.0})
        
        # Place interferer near AP (within 5-15 meters)
        offset_x = random.uniform(-15, 15)
        offset_y = random.uniform(-15, 15)
        x = max(sim.env.x_min, min(sim.env.x_max, ap.x + offset_x))
        y = max(sim.env.y_min, min(sim.env.y_max, ap.y + offset_y))
        
        # Create interferer on same channel as AP
        interferer = Interferer(
            id=self.next_interferer_id,
            x=x,
            y=y,
            tx_power=config['tx_power'],
            channel=ap.channel,
            type=interferer_type,
            bandwidth=config['bandwidth'],
            duty_cycle=config['duty_cycle']
        )
        self.next_interferer_id += 1
        
        # Duration: 5-15 minutes (5-15 steps at 1 min/step)
        duration_steps = random.randint(5, 15)
        expires_at = step + duration_steps
        
        # Track and add to simulation
        self.active_interferers[interferer.id] = {
            'interferer': interferer,
            'expires_at_step': expires_at,
            'ap_id': ap.id,
            'type': interferer_type
        }
        sim.interferers.append(interferer)
        
        print(f"[Interference] Created {interferer_type} near AP {ap.id} (ID:{interferer.id}, expires step {expires_at})")
        return interferer

    def _cleanup_expired_interferers(self, step: int, sim):
        """Remove expired interferers from the simulation"""
        expired_ids = []
        for int_id, data in self.active_interferers.items():
            if step >= data['expires_at_step']:
                expired_ids.append(int_id)
        
        for int_id in expired_ids:
            data = self.active_interferers.pop(int_id)
            interferer = data['interferer']
            if interferer in sim.interferers:
                sim.interferers.remove(interferer)
                print(f"[Interference] {data['type']} near AP {data['ap_id']} expired (ID:{int_id})")

    def _get_simulated_time(self, step):
        """Get simulated timestamp for a given step"""
        return self.sim_start_time + timedelta(seconds=step * self.seconds_per_step)

    def _log_audit_entry(self, step, action_type, action_name, ap_id=None, details=None):
        """Log an audit entry for any RRM action"""
        sim_time = self._get_simulated_time(step)
        entry = {
            'timestamp': sim_time.isoformat(),
            'step': step,
            'day': self.get_day_number(step) + 1,
            'hour': self.get_hour_of_day(step),
            'action_type': action_type,
            'action_name': action_name,
            'ap_id': ap_id,
            'details': details or {}
        }
        self.audit_log.append(entry)

    def _process_rrm_result(self, step, rrm_result):
        """Process RRM result and log all actions to audit"""
        
        # Event Loop actions
        if 'event_action' in rrm_result:
            self.stats['event_loop_actions'] += 1
            self.stats['actions_executed'] += 1
            event_data = rrm_result.get('event_action', {})
            ap_id = event_data.get('ap_id')
            self._log_audit_entry(step, 'EVENT_LOOP', 'event_response', ap_id, event_data)
        
        # Fast Loop actions
        if 'fast_loop_actions' in rrm_result:
            actions = rrm_result['fast_loop_actions']
            for action in actions:
                if action.get('success'):
                    self.stats['fast_loop_actions'] += 1
                    ap_id = action.get('ap_id')
                    self._log_audit_entry(step, 'FAST_LOOP', action.get('action', 'unknown'), ap_id, action)
        
        # Slow Loop actions (SafeRL-based)
        if 'slow_loop' in rrm_result:
            self.stats['slow_loop_actions'] += 1
            slow_data = rrm_result.get('slow_loop', {})
            self._log_audit_entry(step, 'SLOW_LOOP', 'saferl_optimization', None, slow_data)
        
        # Steering/roaming
        if rrm_result.get('steering'):
            self.stats['total_roams'] += len(rrm_result['steering'])
            for steer in rrm_result['steering']:
                self._log_audit_entry(step, 'STEERING', 'client_roam', steer.get('to_ap'), steer)

    def adjust_client_count(self, sim, target_count):
        """Dynamically add or remove clients to match target"""
        current_count = len(sim.clients)
        if target_count > current_count:
            for _ in range(target_count - current_count):
                x = random.uniform(sim.env.x_min, sim.env.x_max)
                y = random.uniform(sim.env.y_min, sim.env.y_max)
                sim.add_client(Client(id=len(sim.clients), x=x, y=y, demand_mbps=random.uniform(5, 30), velocity=random.uniform(0.01, 0.4)))
        elif target_count < current_count:
            for _ in range(current_count - target_count):
                if sim.clients:
                    removed = sim.clients.pop()
                    if removed.associated_ap is not None:
                        ap = next((a for a in sim.access_points if a.id == removed.associated_ap), None)
                        if ap and removed.id in ap.connected_clients:
                            ap.connected_clients.remove(removed.id)

    def run(self):
        """Run 3-day simulation"""
        env = Environment(x_min=0, x_max=50, y_min=0, y_max=50)
        base_model = PathLossModel(frequency_mhz=2400, path_loss_exp=5.0)
        fading_model = MultipathFadingModel(base_model, fading_margin_db=8.0)
        
        sim = WirelessSimulation(
            env, fading_model,
            interference_threshold_dbm=-75.0,
            enable_logging=True,
            log_dir=str(self.state_dir)
        )
        
        # Create network
        print("Creating network topology...")
        N_ap = 8
        ap_positions = create_linear_topology(N_ap, env)
        ap_channels = [1, 1, 6, 6, 11, 11, 1, 6]  # 8 channels for 8 APs (2.4GHz only)
        
        for i, (x, y) in enumerate(ap_positions):
            channel = ap_channels[i]
            bandwidth = 20
            sim.add_access_point(AccessPoint(id=i, x=x, y=y, tx_power=random.uniform(23, 30), channel=channel, bandwidth=bandwidth, max_throughput=150.0))
        
        # Initial clients
        initial_clients = self.get_client_count(0)
        client_positions = create_random_topology(initial_clients, env)
        for i, (x, y) in enumerate(client_positions):
            sim.add_client(Client(id=i, x=x, y=y, demand_mbps=random.uniform(10, 50), velocity=random.uniform(0.1, 0.5)))
        
        sim.initialize()
        
        # RRM Engine - All loops enabled (Fast, Event, Slow)
        rrm = EnhancedRRMEngine(
            access_points=sim.access_points,
            clients=sim.clients,
            interferers=sim.interferers,
            prop_model=fading_model,
            cooldown_steps=8,
            fast_loop_period=10,  # Every 10 steps = 10 minutes
            slow_loop_period=1440,  # Every 1440 steps (once per day)
            audit_log_dir=str(self.audit_dir)
        )
        # Slow loop is now enabled (uses SafeRL inference)
        print(f"[RRM] Slow Loop enabled: {rrm.slow_loop_engine is not None}")
        
        print(f"Starting 3-day simulation ({self.total_steps:,} steps)...")
        print(f"RRM Enabled: {self.enable_rrm}")
        start_time = time.time()
        
        # Main loop
        for step in range(1, self.total_steps + 1):
            # Adjust client count every 10 minutes (10 steps)
            if step % 10 == 0:
                target_clients = self.get_client_count(step)
                self.adjust_client_count(sim, target_clients)
                self.stats['peak_clients'] = max(self.stats['peak_clients'], len(sim.clients))
            
            safe_rl_data = sim.step()
            
            # Always inject events (for realistic conditions)
            self.inject_random_events(step, rrm, sim)
            
            # RRM execution (if enabled) - reacts to injected events
            if self.enable_rrm:
                rrm_result = rrm.execute(step, safe_rl_data)
                # Process and log all RRM actions
                self._process_rrm_result(step, rrm_result)
            
            # Time series data every 30 mins (30 steps)
            if step % 30 == 0:
                self.stats['client_counts'].append(len(sim.clients))
            
            # Progress every hour
            if step % self.steps_per_hour == 0:
                day = self.get_day_number(step)
                hour = self.get_hour_of_day(step)
                progress = (step / self.total_steps) * 100
                total_events = sum(self.stats['events_by_type'].values())
                print(f"\r[Day {day+1}, Hour {hour:02d}:00] {progress:.1f}% | Clients: {len(sim.clients)} | Events: {total_events} | FL: {self.stats['fast_loop_actions']} | EL: {self.stats['event_loop_actions']} | SL: {self.stats['slow_loop_actions']}", end='')
        
        print("\n\nSimulation complete!")
        
        # Final stats from RRM
        try:
            rrm_stats = rrm.event_loop.get_statistics()
            self.stats['rollbacks_triggered'] = rrm_stats.get('rollbacks_triggered', 0)
        except:
            pass
        
        # Export event loop audit trail
        try:
            rrm.event_loop.audit_logger.export_audit_trail()
        except:
            pass
        
        self.generate_summary_report(sim, rrm, start_time)
        self.save_audit_log()
        self.generate_plots()

    def save_audit_log(self):
        """Save comprehensive audit log to file"""
        audit_file = self.audit_dir / "comprehensive_audit.json"
        
        # Group by action type for summary
        summary = {
            'total_entries': len(self.audit_log),
            'by_action_type': {},
            'entries': self.audit_log
        }
        
        for entry in self.audit_log:
            action_type = entry['action_type']
            summary['by_action_type'][action_type] = summary['by_action_type'].get(action_type, 0) + 1
        
        with open(audit_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nComprehensive audit log saved to {audit_file}")
        print(f"  Total entries: {len(self.audit_log)}")
        for action_type, count in summary['by_action_type'].items():
            print(f"  - {action_type}: {count}")

    def generate_plots(self):
        """Generate analysis plots"""
        print("\nGenerating analysis plots...")
        
        # Set seaborn style
        sns.set_theme(style="whitegrid")
        
        # 1. Events by Hour
        plt.figure(figsize=(12, 6))
        hours = list(range(24))
        data = pd.DataFrame({'Hour': hours, 'Events': self.stats['events_by_hour']})
        
        sns.barplot(data=data, x='Hour', y='Events', color='skyblue', edgecolor='black')
        plt.title('Events Distribution by Hour of Day', fontsize=14)
        plt.xlabel('Hour of Day (0-23)', fontsize=12)
        plt.ylabel('Number of Events', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'events_by_hour.png')
        plt.close()
        
        # 2. Client Count Over Time
        plt.figure(figsize=(14, 7))
        plt.plot(self.stats['client_counts'], label='Active Clients', color='#2ecc71', linewidth=2)
        plt.title('Client Load Trend Over 3 Days', fontsize=14)
        plt.xlabel('Time (30-min intervals)', fontsize=12)
        plt.ylabel('Number of Active Clients', fontsize=12)
        plt.fill_between(range(len(self.stats['client_counts'])), self.stats['client_counts'], alpha=0.3, color='#2ecc71')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'client_load_trend.png')
        plt.close()
        
        # 3. Actions by Loop Type (Pie Chart)
        plt.figure(figsize=(8, 8))
        loop_actions = {
            'Fast Loop': self.stats['fast_loop_actions'],
            'Event Loop': self.stats['event_loop_actions'],
            'Slow Loop': self.stats['slow_loop_actions']
        }
        # Filter out zeros
        loop_actions = {k: v for k, v in loop_actions.items() if v > 0}
        if loop_actions:
            plt.pie(loop_actions.values(), labels=loop_actions.keys(), autopct='%1.1f%%', 
                   colors=['#3498db', '#e74c3c', '#2ecc71'])
            plt.title('RRM Actions by Loop Type', fontsize=14)
            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'actions_by_loop.png')
        plt.close()
        
        print(f"Plots saved to {self.analysis_dir}/")

    def generate_summary_report(self, sim, rrm, start_time):
        """Generate summary statistics and report"""
        elapsed = time.time() - start_time
        
        summary = {
            'duration_sec': elapsed,
            'total_steps': self.total_steps,
            'network': {
                'access_points': len(sim.access_points),
                'peak_clients': self.stats['peak_clients'],
                'total_roams': self.stats['total_roams']
            },
            'events': {
                'total': sum(self.stats['events_by_type'].values()),
                'by_type': self.stats['events_by_type']
            },
            'rrm_actions': {
                'fast_loop': self.stats['fast_loop_actions'],
                'event_loop': self.stats['event_loop_actions'],
                'slow_loop': self.stats['slow_loop_actions'],
                'total': self.stats['fast_loop_actions'] + self.stats['event_loop_actions'] + self.stats['slow_loop_actions']
            },
            'rollbacks': self.stats['rollbacks_triggered']
        }
        
        with open(self.analysis_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\nSummary saved to {self.analysis_dir}/summary.json")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='3-Day Simulation Log Generator')
    parser.add_argument('--no-rrm', action='store_true', help='Disable RRM actions')
    parser.add_argument('--output', '-o', default='3_day_logs', help='Output directory')
    parser.add_argument('--environment', default='education', help='Simulation environment: office or education')
    args = parser.parse_args()
    
    sim = ThreeDaySimulation(root_dir=args.output, enable_rrm=not args.no_rrm, environment=args.environment)
    sim.run()

if __name__ == "__main__":
    main()
