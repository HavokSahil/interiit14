"""
3-Day Log Analysis Script

Analyzes logs from 3_day_logs directory and computes:
- Client QoE metrics and plots
- Summary statistics (avg throughput, SINR, retry rates, etc.)
- Similar to A/B testing but from pre-recorded logs

Usage:
    python analyze_3day_logs.py [log_directory]
    python analyze_3day_logs.py 3_day_logs
"""

import os
import sys
import glob
import json
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ClientQoEMetrics:
    """QoE metrics for a single client at a point in time"""
    client_id: int
    step: int
    rssi_dbm: float
    sinr_db: float
    throughput_mbps: float
    retry_rate: float
    airtime_fraction: float
    qoe_score: float = 0.0


@dataclass
class SummaryStats:
    """Summary statistics for the simulation"""
    total_steps: int = 0
    total_clients_seen: int = 0
    
    # Throughput stats
    avg_throughput_mbps: float = 0.0
    p50_throughput_mbps: float = 0.0
    p95_throughput_mbps: float = 0.0
    max_throughput_mbps: float = 0.0
    
    # SINR stats
    avg_sinr_db: float = 0.0
    p50_sinr_db: float = 0.0
    p95_sinr_db: float = 0.0
    
    # RSSI stats
    avg_rssi_dbm: float = 0.0
    p50_rssi_dbm: float = 0.0
    
    # Retry rate stats
    avg_retry_rate: float = 0.0
    p95_retry_rate: float = 0.0
    
    # QoE stats
    avg_qoe: float = 0.0
    p50_qoe: float = 0.0
    p95_qoe: float = 0.0
    min_qoe: float = 0.0
    
    # Roaming stats
    total_roams: int = 0
    
    # AP stats
    num_aps: int = 0
    avg_clients_per_ap: float = 0.0


class LogAnalyzer:
    """Analyzes 3-day simulation logs and computes QoE metrics"""
    
    def __init__(self, log_dir: str = "3_day_logs"):
        self.log_dir = Path(log_dir)
        self.state_dir = self.log_dir / "state_logs"
        self.audit_dir = self.log_dir / "audit"
        self.analysis_dir = self.log_dir / "analysis"
        
        # Create analysis directory if needed
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # DataFrames for each log type
        self.client_df: Optional[pd.DataFrame] = None
        self.ap_df: Optional[pd.DataFrame] = None
        self.roam_df: Optional[pd.DataFrame] = None
        self.audit_data: Optional[Dict] = None
        
        # Computed metrics
        self.qoe_time_series: List[float] = []
        self.throughput_time_series: List[float] = []
        self.summary: Optional[SummaryStats] = None
        
        print(f"\n{'='*70}")
        print("3-DAY LOG ANALYZER")
        print(f"{'='*70}")
        print(f"Log Directory: {self.log_dir}/")
    
    def load_logs(self) -> bool:
        """Load all log files from the directory"""
        print("\nLoading log files...")
        
        # Find client log file
        client_files = list(self.state_dir.glob("*_client_*.csv"))
        if client_files:
            self.client_df = pd.read_csv(client_files[0])
            print(f"  ✓ Client log: {client_files[0].name} ({len(self.client_df)} rows)")
        else:
            print("  ✗ No client log found")
            return False
        
        # Find AP log file
        ap_files = list(self.state_dir.glob("*_ap_*.csv"))
        if ap_files:
            self.ap_df = pd.read_csv(ap_files[0])
            print(f"  ✓ AP log: {ap_files[0].name} ({len(self.ap_df)} rows)")
        else:
            print("  ✗ No AP log found")
        
        # Find roaming log file
        roam_files = list(self.state_dir.glob("*_roam_*.csv"))
        if roam_files:
            self.roam_df = pd.read_csv(roam_files[0])
            print(f"  ✓ Roaming log: {roam_files[0].name} ({len(self.roam_df)} rows)")
        else:
            print("  ✗ No roaming log found")
        
        # Load audit data
        audit_file = self.audit_dir / "comprehensive_audit.json"
        if audit_file.exists():
            with open(audit_file) as f:
                self.audit_data = json.load(f)
            print(f"  ✓ Audit log: {audit_file.name} ({self.audit_data.get('total_entries', 0)} entries)")
        else:
            print("  ✗ No audit log found")
        
        return True
    
    def compute_client_qoe(self, row: pd.Series) -> float:
        """
        Compute QoE score for a client based on metrics.
        
        QoE = weighted combination of:
        - Signal quality (RSSI normalized)
        - Throughput (normalized to demand)
        - Reliability (1 - retry_rate/100)
        - Airtime efficiency
        """
        # Weights
        w_signal = 0.2
        w_throughput = 0.4
        w_reliability = 0.3
        w_airtime = 0.1
        
        # Signal quality (RSSI: -90 to -40 dBm -> 0 to 1)
        rssi = row.get('rssi_dbm', -80)
        if isinstance(rssi, str):
            rssi = float(rssi) if rssi != 'N/A' else -80
        signal_score = max(0, min(1, (rssi + 90) / 50))
        
        # Throughput satisfaction (actual / demand)
        throughput = row.get('throughput_mbps', 0)
        demand = row.get('demand_mbps', 10)
        if isinstance(throughput, str):
            throughput = float(throughput) if throughput != 'N/A' else 0
        if isinstance(demand, str):
            demand = float(demand) if demand != 'N/A' else 10
        throughput_score = min(1.0, throughput / max(demand, 1))
        
        # Reliability (inverse of retry rate)
        retry_rate = row.get('retry_rate', 0)
        if isinstance(retry_rate, str):
            retry_rate = float(retry_rate) if retry_rate != 'N/A' else 0
        reliability_score = max(0, 1 - retry_rate / 100)
        
        # Airtime efficiency
        airtime = row.get('airtime_fraction', 0)
        if isinstance(airtime, str):
            airtime = float(airtime) if airtime != 'N/A' else 0
        # Penalize very low airtime (starved) or very high (hogging)
        airtime_score = 1 - abs(airtime - 0.3) / 0.7  # Optimal around 0.3
        airtime_score = max(0, min(1, airtime_score))
        
        # Weighted QoE
        qoe = (w_signal * signal_score +
               w_throughput * throughput_score +
               w_reliability * reliability_score +
               w_airtime * airtime_score)
        
        return qoe
    
    def compute_statistics(self):
        """Compute summary statistics from logs"""
        print("\nComputing statistics...")
        
        if self.client_df is None:
            print("  ✗ No client data to analyze")
            return
        
        # Convert numeric columns
        numeric_cols = ['throughput_mbps', 'rssi_dbm', 'sinr_db', 'retry_rate', 'airtime_fraction']
        for col in numeric_cols:
            if col in self.client_df.columns:
                self.client_df[col] = pd.to_numeric(self.client_df[col], errors='coerce')
        
        # Compute QoE for each row
        self.client_df['qoe'] = self.client_df.apply(self.compute_client_qoe, axis=1)
        
        # Get valid data
        valid_throughput = self.client_df['throughput_mbps'].dropna()
        valid_sinr = self.client_df['sinr_db'].dropna()
        valid_rssi = self.client_df['rssi_dbm'].dropna()
        valid_retry = self.client_df['retry_rate'].dropna()
        valid_qoe = self.client_df['qoe'].dropna()
        
        # Build summary
        self.summary = SummaryStats()
        self.summary.total_steps = self.client_df['step'].nunique()
        self.summary.total_clients_seen = self.client_df['client_id'].nunique()
        
        # Throughput stats
        if len(valid_throughput) > 0:
            self.summary.avg_throughput_mbps = valid_throughput.mean()
            self.summary.p50_throughput_mbps = valid_throughput.quantile(0.5)
            self.summary.p95_throughput_mbps = valid_throughput.quantile(0.95)
            self.summary.max_throughput_mbps = valid_throughput.max()
        
        # SINR stats
        if len(valid_sinr) > 0:
            self.summary.avg_sinr_db = valid_sinr.mean()
            self.summary.p50_sinr_db = valid_sinr.quantile(0.5)
            self.summary.p95_sinr_db = valid_sinr.quantile(0.95)
        
        # RSSI stats
        if len(valid_rssi) > 0:
            self.summary.avg_rssi_dbm = valid_rssi.mean()
            self.summary.p50_rssi_dbm = valid_rssi.quantile(0.5)
        
        # Retry rate stats
        if len(valid_retry) > 0:
            self.summary.avg_retry_rate = valid_retry.mean()
            self.summary.p95_retry_rate = valid_retry.quantile(0.95)
        
        # QoE stats
        if len(valid_qoe) > 0:
            self.summary.avg_qoe = valid_qoe.mean()
            self.summary.p50_qoe = valid_qoe.quantile(0.5)
            self.summary.p95_qoe = valid_qoe.quantile(0.95)
            self.summary.min_qoe = valid_qoe.min()
        
        # Roaming stats
        if self.roam_df is not None and len(self.roam_df) > 0:
            self.summary.total_roams = len(self.roam_df)
        
        # AP stats
        if self.ap_df is not None:
            self.summary.num_aps = self.ap_df['ap_id'].nunique()
            if 'num_clients' in self.ap_df.columns:
                self.summary.avg_clients_per_ap = pd.to_numeric(
                    self.ap_df['num_clients'], errors='coerce'
                ).mean()
        
        # Time series for plots
        step_groups = self.client_df.groupby('step')
        self.qoe_time_series = step_groups['qoe'].mean().tolist()
        self.throughput_time_series = step_groups['throughput_mbps'].mean().tolist()
        
        print("  ✓ Statistics computed")
    
    def generate_plots(self):
        """Generate analysis plots"""
        print("\nGenerating plots...")
        
        sns.set_theme(style="whitegrid")
        
        # 1. QoE Time Series
        if self.qoe_time_series:
            plt.figure(figsize=(14, 6))
            plt.plot(self.qoe_time_series, color='#3498db', linewidth=1.5, alpha=0.8)
            plt.fill_between(range(len(self.qoe_time_series)), self.qoe_time_series, 
                           alpha=0.3, color='#3498db')
            plt.axhline(y=self.summary.avg_qoe, color='red', linestyle='--', 
                       label=f'Mean QoE: {self.summary.avg_qoe:.3f}')
            plt.title('Client QoE Over Time', fontsize=14)
            plt.xlabel('Simulation Step', fontsize=12)
            plt.ylabel('Average QoE Score', fontsize=12)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'qoe_time_series.png', dpi=150)
            plt.close()
            print(f"  ✓ QoE time series plot saved")
        
        # 2. Throughput Time Series
        if self.throughput_time_series:
            plt.figure(figsize=(14, 6))
            plt.plot(self.throughput_time_series, color='#2ecc71', linewidth=1.5, alpha=0.8)
            plt.fill_between(range(len(self.throughput_time_series)), self.throughput_time_series,
                           alpha=0.3, color='#2ecc71')
            plt.axhline(y=self.summary.avg_throughput_mbps, color='red', linestyle='--',
                       label=f'Mean: {self.summary.avg_throughput_mbps:.2f} Mbps')
            plt.title('Average Throughput Over Time', fontsize=14)
            plt.xlabel('Simulation Step', fontsize=12)
            plt.ylabel('Throughput (Mbps)', fontsize=12)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'throughput_time_series.png', dpi=150)
            plt.close()
            print(f"  ✓ Throughput time series plot saved")
        
        # 3. QoE Distribution
        if self.client_df is not None and 'qoe' in self.client_df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.client_df['qoe'].dropna(), bins=50, kde=True, color='#9b59b6')
            plt.axvline(x=self.summary.avg_qoe, color='red', linestyle='--',
                       label=f'Mean: {self.summary.avg_qoe:.3f}')
            plt.axvline(x=self.summary.p95_qoe, color='orange', linestyle='--',
                       label=f'P95: {self.summary.p95_qoe:.3f}')
            plt.title('QoE Distribution', fontsize=14)
            plt.xlabel('QoE Score', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'qoe_distribution.png', dpi=150)
            plt.close()
            print(f"  ✓ QoE distribution plot saved")
        
        # 4. Throughput Distribution
        if self.client_df is not None:
            plt.figure(figsize=(10, 6))
            throughput_data = self.client_df['throughput_mbps'].dropna()
            sns.histplot(throughput_data, bins=50, kde=True, color='#e74c3c')
            plt.axvline(x=self.summary.avg_throughput_mbps, color='blue', linestyle='--',
                       label=f'Mean: {self.summary.avg_throughput_mbps:.2f}')
            plt.axvline(x=self.summary.p95_throughput_mbps, color='green', linestyle='--',
                       label=f'P95: {self.summary.p95_throughput_mbps:.2f}')
            plt.title('Throughput Distribution', fontsize=14)
            plt.xlabel('Throughput (Mbps)', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'throughput_distribution.png', dpi=150)
            plt.close()
            print(f"  ✓ Throughput distribution plot saved")
        
        # 5. Summary Bar Chart
        plt.figure(figsize=(12, 6))
        metrics = ['Avg QoE', 'P95 QoE', 'Avg Throughput\n(normalized)', 'P95 Retry Rate\n(inverted)']
        values = [
            self.summary.avg_qoe,
            self.summary.p95_qoe,
            min(1.0, self.summary.avg_throughput_mbps / 50),  # Normalize to 0-1
            max(0, 1 - self.summary.p95_retry_rate / 100)  # Invert retry rate
        ]
        colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c']
        
        bars = plt.bar(metrics, values, color=colors, edgecolor='black')
        plt.ylim(0, 1.1)
        plt.title('Performance Summary', fontsize=14)
        plt.ylabel('Score (0-1)', fontsize=12)
        
        # Add value labels
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'performance_summary.png', dpi=150)
        plt.close()
        print(f"  ✓ Performance summary plot saved")
        
        # 6. RRM Actions Breakdown (from audit)
        if self.audit_data and 'by_action_type' in self.audit_data:
            plt.figure(figsize=(8, 8))
            action_types = self.audit_data['by_action_type']
            if action_types:
                labels = list(action_types.keys())
                sizes = list(action_types.values())
                colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'][:len(labels)]
                
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors,
                       explode=[0.02]*len(labels))
                plt.title('RRM Actions by Type', fontsize=14)
                plt.tight_layout()
                plt.savefig(self.analysis_dir / 'rrm_actions_breakdown.png', dpi=150)
            plt.close()
            print(f"  ✓ RRM actions breakdown plot saved")
    
    def print_summary(self):
        """Print summary statistics to console"""
        if not self.summary:
            return
        
        print(f"\n{'='*70}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*70}")
        
        print(f"\n📊 SIMULATION OVERVIEW")
        print(f"   Total Steps:        {self.summary.total_steps:,}")
        print(f"   Unique Clients:     {self.summary.total_clients_seen}")
        print(f"   Access Points:      {self.summary.num_aps}")
        print(f"   Total Roams:        {self.summary.total_roams}")
        
        print(f"\n📶 THROUGHPUT METRICS")
        print(f"   Average:            {self.summary.avg_throughput_mbps:.2f} Mbps")
        print(f"   Median (P50):       {self.summary.p50_throughput_mbps:.2f} Mbps")
        print(f"   95th Percentile:    {self.summary.p95_throughput_mbps:.2f} Mbps")
        print(f"   Maximum:            {self.summary.max_throughput_mbps:.2f} Mbps")
        
        print(f"\n📡 SIGNAL QUALITY")
        print(f"   Avg RSSI:           {self.summary.avg_rssi_dbm:.1f} dBm")
        print(f"   Avg SINR:           {self.summary.avg_sinr_db:.1f} dB")
        print(f"   P95 SINR:           {self.summary.p95_sinr_db:.1f} dB")
        
        print(f"\n🔄 RELIABILITY")
        print(f"   Avg Retry Rate:     {self.summary.avg_retry_rate:.2f}%")
        print(f"   P95 Retry Rate:     {self.summary.p95_retry_rate:.2f}%")
        
        print(f"\n⭐ QoE SCORES")
        print(f"   Average QoE:        {self.summary.avg_qoe:.4f}")
        print(f"   Median (P50):       {self.summary.p50_qoe:.4f}")
        print(f"   95th Percentile:    {self.summary.p95_qoe:.4f}")
        print(f"   Minimum:            {self.summary.min_qoe:.4f}")
        
        if self.audit_data:
            print(f"\n🔧 RRM ACTIONS")
            for action_type, count in self.audit_data.get('by_action_type', {}).items():
                print(f"   {action_type}: {count}")
        
        print(f"\n{'='*70}")
    
    def save_summary(self):
        """Save summary statistics to JSON file"""
        if not self.summary:
            return
        
        summary_dict = {
            'simulation_overview': {
                'total_steps': self.summary.total_steps,
                'unique_clients': self.summary.total_clients_seen,
                'access_points': self.summary.num_aps,
                'total_roams': self.summary.total_roams,
                'avg_clients_per_ap': round(self.summary.avg_clients_per_ap, 2)
            },
            'throughput': {
                'avg_mbps': round(self.summary.avg_throughput_mbps, 2),
                'p50_mbps': round(self.summary.p50_throughput_mbps, 2),
                'p95_mbps': round(self.summary.p95_throughput_mbps, 2),
                'max_mbps': round(self.summary.max_throughput_mbps, 2)
            },
            'signal_quality': {
                'avg_rssi_dbm': round(self.summary.avg_rssi_dbm, 1),
                'p50_rssi_dbm': round(self.summary.p50_rssi_dbm, 1),
                'avg_sinr_db': round(self.summary.avg_sinr_db, 1),
                'p95_sinr_db': round(self.summary.p95_sinr_db, 1)
            },
            'reliability': {
                'avg_retry_rate': round(self.summary.avg_retry_rate, 2),
                'p95_retry_rate': round(self.summary.p95_retry_rate, 2)
            },
            'qoe': {
                'avg': round(self.summary.avg_qoe, 4),
                'p50': round(self.summary.p50_qoe, 4),
                'p95': round(self.summary.p95_qoe, 4),
                'min': round(self.summary.min_qoe, 4)
            },
            'rrm_actions': self.audit_data.get('by_action_type', {}) if self.audit_data else {}
        }
        
        output_file = self.analysis_dir / 'analysis_summary.json'
        with open(output_file, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        
        print(f"\n✓ Summary saved to: {output_file}")
    
    def run(self):
        """Run complete analysis pipeline"""
        if not self.load_logs():
            print("\n✗ Failed to load logs. Exiting.")
            return False
        
        self.compute_statistics()
        self.generate_plots()
        self.print_summary()
        self.save_summary()
        
        print(f"\n✓ Analysis complete! Results saved to: {self.analysis_dir}/")
        return True


def main():
    """Main entry point"""
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "3_day_logs"
    
    analyzer = LogAnalyzer(log_dir)
    analyzer.run()


if __name__ == "__main__":
    main()
