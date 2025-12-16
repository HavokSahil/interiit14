"""
QoE Dashboard - Visualization for QoE monitoring and A/B testing.

Provides real-time plots and comparison visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Optional
import numpy as np

from qoe_monitor import QoEMonitor, QoESnapshot
from ab_testing import ABTestFramework, ComparisonReport


class QoEDashboard:
    """Interactive QoE visualization dashboard"""
    
    def __init__(self, update_interval: int = 10):
        """
        Initialize dashboard.
        
        Args:
            update_interval: Steps between plot updates
        """
        self.update_interval = update_interval
        self.fig = None
        self.axes = None
    
    def create_realtime_plot(self, qoe_monitor: QoEMonitor, window: int = 500):
        """
        Create real-time QoE line plot.
        
        Args:
            qoe_monitor: QoE monitor instance
            window: Number of steps to show
        """
        history = qoe_monitor.get_time_series(window=window)
        
        if not history:
            print("No QoE data to plot")
            return
        
        steps = [s.step for s in history]
        network_qoe = [s.network_qoe for s in history]
        
        # Get per-role data
        roles = set()
        for snapshot in history:
            roles.update(snapshot.per_role_qoe.keys())
        
        plt.figure(figsize=(12, 6))
        
        # Plot network QoE
        plt.plot(steps, network_qoe, 'b-', linewidth=2, label='Network Average')
        
        # Plot per-role QoE
        colors = ['r', 'g', 'orange', 'purple']
        for i, role in enumerate(sorted(roles)):
            role_qoe = [s.per_role_qoe.get(role, 0) for s in history]
            plt.plot(steps, role_qoe, color=colors[i % len(colors)], 
                    linestyle='--', label=f'Role: {role}', alpha=0.7)
        
        plt.xlabel('Simulation Step')
        plt.ylabel('QoE Score')
        plt.title('Real-Time QoE Monitoring')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.0])
        
        plt.tight_layout()
        return plt.gcf()
    
    def create_comparison_plot(self, ab_framework: ABTestFramework):
        """
        Create A/B comparison visualization.
        
        Args:
            ab_framework: A/B test framework with results
        """
        if not ab_framework.comparison_report:
            ab_framework.compute_comparison()
        
        report = ab_framework.comparison_report
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # QoE comparison
        ax = axes[0, 0]
        variants = [report.variant_a_name, report.variant_b_name]
        qoe_means = [report.variant_a_qoe_mean, report.variant_b_qoe_mean]
        colors = ['#FF6B6B', '#4ECDC4']
        bars = ax.bar(variants, qoe_means, color=colors, alpha=0.8)
        ax.set_ylabel('Mean QoE')
        ax.set_title(f'QoE Comparison\n({report.qoe_improvement_pct:+.2f}% improvement)')
        ax.set_ylim([0, 1.0])
        
        # Add value labels on bars
        for bar, val in zip(bars, qoe_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom')
        
        # Throughput comparison
        ax = axes[0, 1]
        tput_means = [report.variant_a_throughput_mean, report.variant_b_throughput_mean]
        bars = ax.bar(variants, tput_means, color=colors, alpha=0.8)
        ax.set_ylabel('Throughput (Mbps)')
        ax.set_title(f'Throughput Comparison\n({report.throughput_improvement_pct:+.2f}% improvement)')
        
        for bar, val in zip(bars, tput_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}', ha='center', va='bottom')
        
        # Retry rate comparison
        ax = axes[1, 0]
        retry_means = [report.variant_a_retry_mean, report.variant_b_retry_mean]
        bars = ax.bar(variants, retry_means, color=colors, alpha=0.8)
        ax.set_ylabel('Retry Rate (%)')
        ax.set_title(f'Retry Rate Comparison\n({report.retry_improvement_pct:+.2f}% improvement)')
        
        for bar, val in zip(bars, retry_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f'{val:.2f}', ha='center', va='bottom')
        
        # Actions taken
        ax = axes[1, 1]
        actions = [report.variant_a_actions, report.variant_b_actions]
        bars = ax.bar(variants, actions, color=colors, alpha=0.8)
        ax.set_ylabel('Actions Taken')
        ax.set_title('RRM Actions Comparison')
        
        for bar, val in zip(bars, actions):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def create_distribution_plot(self, ab_framework: ABTestFramework):
        """
        Create QoE distribution histograms.
        
        Args:
            ab_framework: A/B test framework
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Variant A histogram
        ax1.hist(ab_framework.metrics_a.qoe_values, bins=50, 
                color='#FF6B6B', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('QoE Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{ab_framework.variant_a.variant_name} QoE Distribution')
        ax1.axvline(ab_framework.metrics_a.get_qoe_mean(), 
                   color='red', linestyle='--', linewidth=2, label='Mean')
        ax1.legend()
        ax1.set_xlim([0, 1.0])
        
        # Variant B histogram
        ax2.hist(ab_framework.metrics_b.qoe_values, bins=50,
                color='#4ECDC4', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('QoE Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{ab_framework.variant_b.variant_name} QoE Distribution')
        ax2.axvline(ab_framework.metrics_b.get_qoe_mean(),
                   color='blue', linestyle='--', linewidth=2, label='Mean')
        ax2.legend()
        ax2.set_xlim([0, 1.0])
        
        plt.tight_layout()
        return fig
    
    def show_dashboard(self, qoe_monitor: QoEMonitor, 
                       ab_framework: Optional[ABTestFramework] = None):
        """
        Show complete dashboard with all visualizations.
        
        Args:
            qoe_monitor: QoE monitor instance
            ab_framework: Optional A/B test framework for comparison
        """
        if ab_framework:
            # Show A/B comparison dashboard
            fig1 = self.create_comparison_plot(ab_framework)
            fig2 = self.create_distribution_plot(ab_framework)
            plt.show()
        else:
            # Show real-time monitoring dashboard
            fig = self.create_realtime_plot(qoe_monitor)
            plt.show()
    
    def save_report(self, filepath: str, qoe_monitor: QoEMonitor,
                   ab_framework: Optional[ABTestFramework] = None):
        """
        Save dashboard plots to file.
        
        Args:
            filepath: Output file path (without extension)
            qoe_monitor: QoE monitor instance
            ab_framework: Optional A/B test framework
        """
        if ab_framework:
            # Save comparison plots
            fig1 = self.create_comparison_plot(ab_framework)
            fig1.savefig(f"{filepath}_comparison.png", dpi=300, bbox_inches='tight')
            
            fig2 = self.create_distribution_plot(ab_framework)
            fig2.savefig(f"{filepath}_distribution.png", dpi=300, bbox_inches='tight')
            
            print(f"Comparison plots saved to {filepath}_comparison.png and {filepath}_distribution.png")
        else:
            # Save real-time plot
            fig = self.create_realtime_plot(qoe_monitor)
            fig.savefig(f"{filepath}_realtime.png", dpi=300, bbox_inches='tight')
            
            print(f"Real-time plot saved to {filepath}_realtime.png")
