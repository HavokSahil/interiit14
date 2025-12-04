"""
A/B Testing Framework for RRM simulation.

Allows comparison of different RRM strategies including:
- No RRM (baseline)
- Fast Loop only
- Event Loop only  
- Full RRM (all loops enabled)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import statistics


@dataclass
class ABTestConfig:
    """Configuration for an A/B test variant"""
    variant_name: str
    
    # RRM mode: "none", "fast_loop", "event_loop", "slow_loop", "full"
    rrm_mode: str = "none"
    
    # Individual loop toggles
    enable_fast_loop: bool = False
    enable_event_loop: bool = False
    enable_slow_loop: bool = False
    
    # Config overrides (optional)
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set individual toggles based on rrm_mode"""
        if self.rrm_mode == "none":
            self.enable_fast_loop = False
            self.enable_event_loop = False
            self.enable_slow_loop = False
        elif self.rrm_mode == "fast_loop":
            self.enable_fast_loop = True
            self.enable_event_loop = False
            self.enable_slow_loop = False
        elif self.rrm_mode == "event_loop":
            self.enable_fast_loop = False
            self.enable_event_loop = True
            self.enable_slow_loop = False
        elif self.rrm_mode == "slow_loop":
            self.enable_fast_loop = False
            self.enable_event_loop = False
            self.enable_slow_loop = True
        elif self.rrm_mode == "full":
            self.enable_fast_loop = True
            self.enable_event_loop = True
            self.enable_slow_loop = True


@dataclass
class VariantMetrics:
    """Metrics collected for one variant"""
    variant_name: str
    qoe_values: List[float] = field(default_factory=list)
    throughput_values: List[float] = field(default_factory=list)
    retry_rate_values: List[float] = field(default_factory=list)
    interference_values: List[float] = field(default_factory=list)
    actions_taken: int = 0
    runtime_seconds: float = 0.0
    
    def get_qoe_mean(self) -> float:
        """Get mean QoE"""
        return statistics.mean(self.qoe_values) if self.qoe_values else 0.0
    
    def get_qoe_p95(self) -> float:
        """Get P95 QoE"""
        if not self.qoe_values:
            return 0.0
        sorted_vals = sorted(self.qoe_values)
        idx = int(len(sorted_vals) * 0.95)
        return sorted_vals[idx]


@dataclass
class ComparisonReport:
    """Comparison report between two variants"""
    variant_a_name: str
    variant_b_name: str
    
    # QoE comparison
    variant_a_qoe_mean: float
    variant_b_qoe_mean: float
    qoe_improvement_pct: float
    qoe_p_value: float
    qoe_significant: bool
    
    # Throughput comparison
    variant_a_throughput_mean: float
    variant_b_throughput_mean: float
    throughput_improvement_pct: float
    
    # Retry rate comparison
    variant_a_retry_mean: float
    variant_b_retry_mean: float
    retry_improvement_pct: float
    
    # Actions taken
    variant_a_actions: int
    variant_b_actions: int
    
    # Runtime
    variant_a_runtime: float
    variant_b_runtime: float
    
    def print_report(self):
        """Print formatted comparison report"""
        print("\n" + "="*70)
        print("A/B TEST COMPARISON REPORT")
        print("="*70)
        print(f"Variant A: {self.variant_a_name}")
        print(f"Variant B: {self.variant_b_name}")
        print()
        
        print("QoE Metrics:")
        print(f"  Variant A Mean: {self.variant_a_qoe_mean:.4f}")
        print(f"  Variant B Mean: {self.variant_b_qoe_mean:.4f}")
        print(f"  Improvement:    {self.qoe_improvement_pct:+.2f}%")
        print(f"  P-value:        {self.qoe_p_value:.4f}")
        print(f"  Significant:    {'✓ YES' if self.qoe_significant else '✗ NO'}")
        print()
        
        print("Throughput:")
        print(f"  Variant A: {self.variant_a_throughput_mean:.2f} Mbps")
        print(f"  Variant B: {self.variant_b_throughput_mean:.2f} Mbps")
        print(f"  Improvement: {self.throughput_improvement_pct:+.2f}%")
        print()
        
        print("Retry Rate:")
        print(f"  Variant A: {self.variant_a_retry_mean:.2f}%")
        print(f"  Variant B: {self.variant_b_retry_mean:.2f}%")
        print(f"  Improvement: {self.retry_improvement_pct:+.2f}%")
        print()
        
        print("Actions Taken:")
        print(f"  Variant A: {self.variant_a_actions}")
        print(f"  Variant B: {self.variant_b_actions}")
        print()
        
        print("Runtime:")
        print(f"  Variant A: {self.variant_a_runtime:.2f}s")
        print(f"  Variant B: {self.variant_b_runtime:.2f}s")
        print("="*70)


class ABTestFramework:
    """
    Framework for A/B testing different RRM configurations.
    
    Supports comparing:
    - No RRM vs RRM (to show pure optimization gains)
    - Different RRM strategies (Fast Loop vs Event Loop, etc.)
    """
    
    def __init__(self, variant_a: ABTestConfig, variant_b: ABTestConfig):
        """
        Initialize A/B test framework.
        
        Args:
            variant_a: Configuration for variant A
            variant_b: Configuration for variant B
        """
        self.variant_a = variant_a
        self.variant_b = variant_b
        
        self.metrics_a = VariantMetrics(variant_name=variant_a.variant_name)
        self.metrics_b = VariantMetrics(variant_name=variant_b.variant_name)
        
        self.comparison_report: Optional[ComparisonReport] = None
    
    def collect_metrics(self, variant: str, qoe: float, throughput: float, 
                       retry_rate: float, interference: float):
        """
        Collect metrics for a variant.
        
        Args:
            variant: "A" or "B"
            qoe: QoE value
            throughput: Network throughput
            retry_rate: Average retry rate
            interference: Average interference
        """
        metrics = self.metrics_a if variant == "A" else self.metrics_b
        
        metrics.qoe_values.append(qoe)
        metrics.throughput_values.append(throughput)
        metrics.retry_rate_values.append(retry_rate)
        metrics.interference_values.append(interference)
    
    def record_action(self, variant: str):
        """Record that an action was taken for a variant"""
        if variant == "A":
            self.metrics_a.actions_taken += 1
        else:
            self.metrics_b.actions_taken += 1
    
    def set_runtime(self, variant: str, runtime_seconds: float):
        """Set runtime for a variant"""
        if variant == "A":
            self.metrics_a.runtime_seconds = runtime_seconds
        else:
            self.metrics_b.runtime_seconds = runtime_seconds
    
    def compute_comparison(self) -> ComparisonReport:
        """
        Compute statistical comparison between variants.
        
        Returns:
            ComparisonReport with comparison statistics
        """
        # QoE comparison
        qoe_a_mean = self.metrics_a.get_qoe_mean()
        qoe_b_mean = self.metrics_b.get_qoe_mean()
        
        if qoe_a_mean > 0:
            qoe_improvement = ((qoe_b_mean - qoe_a_mean) / qoe_a_mean) * 100
        else:
            qoe_improvement = 0.0
        
        # T-test for statistical significance
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(self.metrics_a.qoe_values, 
                                               self.metrics_b.qoe_values)
            qoe_p_value = p_value
            qoe_significant = p_value < 0.05
        except (ImportError, Exception):
            # Fallback if scipy not available
            qoe_p_value = 1.0
            qoe_significant = False
        
        # Throughput comparison
        tput_a_mean = statistics.mean(self.metrics_a.throughput_values) if self.metrics_a.throughput_values else 0
        tput_b_mean = statistics.mean(self.metrics_b.throughput_values) if self.metrics_b.throughput_values else 0
        
        if tput_a_mean > 0:
            tput_improvement = ((tput_b_mean - tput_a_mean) / tput_a_mean) * 100
        else:
            tput_improvement = 0.0
        
        # Retry rate comparison
        retry_a_mean = statistics.mean(self.metrics_a.retry_rate_values) if self.metrics_a.retry_rate_values else 0
        retry_b_mean = statistics.mean(self.metrics_b.retry_rate_values) if self.metrics_b.retry_rate_values else 0
        
        if retry_a_mean > 0:
            # Lower is better, so improvement is negative delta
            retry_improvement = -((retry_b_mean - retry_a_mean) / retry_a_mean) * 100
        else:
            retry_improvement = 0.0
        
        self.comparison_report = ComparisonReport(
            variant_a_name=self.variant_a.variant_name,
            variant_b_name=self.variant_b.variant_name,
            variant_a_qoe_mean=qoe_a_mean,
            variant_b_qoe_mean=qoe_b_mean,
            qoe_improvement_pct=qoe_improvement,
            qoe_p_value=qoe_p_value,
            qoe_significant=qoe_significant,
            variant_a_throughput_mean=tput_a_mean,
            variant_b_throughput_mean=tput_b_mean,
            throughput_improvement_pct=tput_improvement,
            variant_a_retry_mean=retry_a_mean,
            variant_b_retry_mean=retry_b_mean,
            retry_improvement_pct=retry_improvement,
            variant_a_actions=self.metrics_a.actions_taken,
            variant_b_actions=self.metrics_b.actions_taken,
            variant_a_runtime=self.metrics_a.runtime_seconds,
            variant_b_runtime=self.metrics_b.runtime_seconds
        )
        
        return self.comparison_report
    
    def export_results(self, filepath: str, format: str = "json"):
        """
        Export A/B test results to file.
        
        Args:
            filepath: Output file path
            format: "json" or "csv"
        """
        if not self.comparison_report:
            self.compute_comparison()
        
        if format == "json":
            # Export as JSON
            result = {
                'timestamp': datetime.now().isoformat(),
                'variant_a': {
                    'name': self.variant_a.variant_name,
                    'mode': self.variant_a.rrm_mode,
                    'qoe_mean': self.comparison_report.variant_a_qoe_mean,
                    'throughput_mean': self.comparison_report.variant_a_throughput_mean,
                    'retry_mean': self.comparison_report.variant_a_retry_mean,
                    'actions': self.comparison_report.variant_a_actions,
                    'runtime': self.comparison_report.variant_a_runtime
                },
                'variant_b': {
                    'name': self.variant_b.variant_name, 
                    'mode': self.variant_b.rrm_mode,
                    'qoe_mean': self.comparison_report.variant_b_qoe_mean,
                    'throughput_mean': self.comparison_report.variant_b_throughput_mean,
                    'retry_mean': self.comparison_report.variant_b_retry_mean,
                    'actions': self.comparison_report.variant_b_actions,
                    'runtime': self.comparison_report.variant_b_runtime
                },
                'comparison': {
                    'qoe_improvement_pct': self.comparison_report.qoe_improvement_pct,
                    'qoe_p_value': self.comparison_report.qoe_p_value,
                    'qoe_significant': int(self.comparison_report.qoe_significant),
                    'throughput_improvement_pct': self.comparison_report.throughput_improvement_pct,
                    'retry_improvement_pct': self.comparison_report.retry_improvement_pct
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"Results exported to {filepath}")
        
        elif format == "csv":
            # Export as CSV
            import csv
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow(['Metric', 'Variant A', 'Variant B', 'Improvement %'])
                
                # Data
                writer.writerow(['QoE Mean', 
                               f"{self.comparison_report.variant_a_qoe_mean:.4f}",
                               f"{self.comparison_report.variant_b_qoe_mean:.4f}",
                               f"{self.comparison_report.qoe_improvement_pct:+.2f}"])
                
                writer.writerow(['Throughput (Mbps)',
                               f"{self.comparison_report.variant_a_throughput_mean:.2f}",
                               f"{self.comparison_report.variant_b_throughput_mean:.2f}",
                               f"{self.comparison_report.throughput_improvement_pct:+.2f}"])
                
                writer.writerow(['Retry Rate (%)',
                               f"{self.comparison_report.variant_a_retry_mean:.2f}",
                               f"{self.comparison_report.variant_b_retry_mean:.2f}",
                               f"{self.comparison_report.retry_improvement_pct:+.2f}"])
                
                writer.writerow(['Actions Taken',
                               self.comparison_report.variant_a_actions,
                               self.comparison_report.variant_b_actions,
                               ''])
            
            print(f"Results exported to {filepath}")
