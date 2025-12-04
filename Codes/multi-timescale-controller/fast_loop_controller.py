"""
Fast Loop Controller - Interference-Based Optimization

Uses real-time interference graph to optimize:
- Channel assignment
- Bandwidth (20/40/80 MHz)
- OBSS-PD threshold (spatial reuse)

Runs every 10 minutes (60 steps) with configuration from YAML.
"""

from typing import Dict, List, Optional, Any, Tuple
import networkx as nx
import yaml
from pathlib import Path
from datatype import AccessPoint
from config_engine import ConfigEngine
from policy_engine import PolicyEngine


class FastLoopController:
    """
    Fast Loop Controller for interference-based optimization.
    
    Uses the interference graph to make reactive decisions about:
    - Channel changes (to avoid congested channels)
    - Bandwidth adjustments (reduce if crowded, increase if clear)
    - OBSS-PD tuning (spatial reuse aggressiveness)
    """
    
    def __init__(self,
                 config_engine: ConfigEngine,
                 policy_engine: PolicyEngine,
                 access_points: List[AccessPoint],
                 config_path: str = "fast_loop_config.yml"):
        """
        Initialize Fast Loop Controller.
        
        Args:
            config_engine: Configuration engine for applying changes
            policy_engine: Policy engine for compliance checks
            access_points: List of access points
            config_path: Path to YAML configuration file
        """
        self.config_engine = config_engine
        self.policy_engine = policy_engine
        self.aps = {ap.id: ap for ap in access_points}
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # State tracking
        self.last_actions = {}  # ap_id -> {action, step, config_before}
        self.action_history = []  # List of all actions taken
        self.current_step = 0  # Track current simulation step
        
        # Statistics
        self.stats = {
            'channel_changes': 0,
            'bandwidth_changes': 0,
            'obss_pd_changes': 0,
            'total_actions': 0,
            'rollbacks': 0
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"[Fast Loop] Config not found: {config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration if YAML not found"""
        return {
            'thresholds': {
                'interference': {'low': 0.2, 'moderate': 0.5, 'high': 0.7},
                'cca_busy': {'low': 0.3, 'moderate': 0.6, 'high': 0.8},
                'retry_rate': {'low': 5.0, 'moderate': 10.0, 'high': 20.0}
            },
            'channels': {
                'band_2ghz': {'available': [1, 6, 11]},
                'band_5ghz': {'available': [36, 40, 44, 48, 149, 153, 157, 161]}
            },
            'bandwidth': {
                'max_increase_step': 1,
                'max_decrease_step': 1
            },
            'obss_pd': {
                'min_threshold': -82,
                'max_threshold': -62,
                'step_size': 3
            }
        }
    
    def execute(self, interference_graph: nx.DiGraph, current_step: int = 0) -> List[Dict[str, Any]]:
        """
        Execute Fast Loop optimization.
        
        Args:
            interference_graph: Real-time interference graph with:
                - Nodes: AP info (channel, load, position)
                - Edges: Interference weights (0-1)
            current_step: Current simulation step number
        
        Returns:
            List of actions taken: [{'ap_id': 0, 'type': 'channel_change', ...}, ...]
        """
        self.current_step = current_step
        actions = []
        
        # Enforce safety limit on concurrent actions
        max_actions = self.config.get('safety', {}).get('max_actions_per_loop', 3)
        
        for ap_id in self.aps.keys():
            if len(actions) >= max_actions:
                break
            
            # Check cooldown for this AP
            if not self._check_cooldown(ap_id):
                continue
            
            # Analyze interference from graph
            analysis = self._analyze_interference(ap_id, interference_graph)
            
            # Get current AP metrics
            ap = self.aps[ap_id]
            metrics = {
                'cca_busy': getattr(ap, 'cca_busy_percentage', 0.0),
                'retry_rate': getattr(ap, 'p95_retry_rate', 0.0),
                'channel': ap.channel,
                'bandwidth': ap.bandwidth,
                'obss_pd': ap.obss_pd_threshold
            }
            
            # Decide action (priority-based)
            action = self._decide_action(ap_id, analysis, metrics)
            
            if action:
                # Apply action
                result = self._apply_action(ap_id, action, metrics)
                if result['success']:
                    actions.append(result)
                    self.stats['total_actions'] += 1
        
        return actions
    
    def _analyze_interference(self, ap_id: int, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Analyze interference for a single AP.
        
        Args:
            ap_id: Access Point ID
            graph: Interference graph
            
        Returns:
            Analysis dict with interference metrics
        """
        if ap_id not in graph.nodes:
            return {'total_interference': 0.0, 'num_interferers': 0, 'channel_interference': {}}
        
        # Get interferers (incoming edges)
        interferers = list(graph.predecessors(ap_id))
        
        # Calculate total interference
        total_interference = sum(
            graph[i][ap_id].get('weight', 0.0)
            for i in interferers
        )
        
        # Group interference by channel
        channel_interference = {}
        for interferer_id in interferers:
            interferer_channel = graph.nodes[interferer_id].get('channel', 1)
            weight = graph[interferer_id][ap_id].get('weight', 0.0)
            
            if interferer_channel not in channel_interference:
                channel_interference[interferer_channel] = 0.0
            channel_interference[interferer_channel] += weight
        
        # Find current channel interference
        current_channel = self.aps[ap_id].channel
        current_channel_interference = channel_interference.get(current_channel, 0.0)
        
        return {
            'total_interference': total_interference,
            'num_interferers': len(interferers),
            'channel_interference': channel_interference,
            'current_channel_interference': current_channel_interference,
            'interferer_ids': interferers
        }
    
    def _decide_action(self, ap_id: int, analysis: Dict, metrics: Dict) -> Optional[Dict]:
        """
        Decide what action to take based on analysis and metrics.
        
        Priority order with intermediate rules to avoid dead zones:
        1. Channel change (if severe interference)
        2. Bandwidth reduction (if moderate interference + medium retry)
        3. OBSS-PD increase (if high CCA but low retry)
        4. Bandwidth increase (if low interference)
        5. OBSS-PD decrease (if high retry)
        6. MODERATE OBSS-PD increase (fill dead zone)
        7. MODERATE OBSS-PD decrease (fill dead zone)
        8. MODERATE channel change (fallback)
        9. Small Bandwidth Reduction (fallback for congestion)
        """
        thresholds = self.config['thresholds']
        interference = analysis['total_interference']
        cca_busy = metrics['cca_busy']
        retry_rate = metrics['retry_rate']
        
        # Priority 1: Channel Change (severe interference)
        if (interference > thresholds['interference']['high'] and 
            retry_rate > thresholds['retry_rate']['high']):
            new_channel = self._find_best_channel(ap_id, analysis, metrics)
            if new_channel and new_channel != metrics['channel']:
                return {
                    'type': 'channel_change',
                    'new_channel': new_channel,
                    'reason': 'severe_interference'
                }
        
        # Priority 2: Bandwidth Reduction (moderate interference + medium retry)
        if (interference > thresholds['interference']['moderate'] and
            retry_rate > thresholds['retry_rate']['moderate'] and
            metrics['bandwidth'] > 20):
            new_bw = self._get_reduced_bandwidth(metrics['bandwidth'])
            if new_bw:
                return {
                    'type': 'bandwidth_reduce',
                    'new_bandwidth': new_bw,
                    'reason': 'high_interference'
                }
        
        # Priority 3: OBSS-PD Increase (high CCA, low retry)
        if (cca_busy > thresholds['cca_busy']['high'] and
            retry_rate < thresholds['retry_rate']['moderate'] and
            metrics['obss_pd'] < self.config['obss_pd']['max_threshold']):
            new_obss_pd = min(
                metrics['obss_pd'] + self.config['obss_pd']['step_size'],
                self.config['obss_pd']['max_threshold']
            )
            return {
                'type': 'obss_pd_increase',
                'new_obss_pd': new_obss_pd,
                'reason': 'high_cca_low_retry'
            }
        
        # Priority 4: Bandwidth Increase (clean spectrum)
        if (interference < thresholds['interference']['low'] and
            cca_busy < thresholds['cca_busy']['low'] and
            retry_rate < thresholds['retry_rate']['low']):
            new_bw = self._get_increased_bandwidth(metrics['bandwidth'], metrics['channel'])
            if new_bw:
                return {
                    'type': 'bandwidth_increase',
                    'new_bandwidth': new_bw,
                    'reason': 'clean_spectrum'
                }
        
        # Priority 5: OBSS-PD Decrease (high retry)
        if (retry_rate > thresholds['retry_rate']['high'] and
            metrics['obss_pd'] > self.config['obss_pd']['min_threshold']):
            new_obss_pd = max(
                metrics['obss_pd'] - self.config['obss_pd']['step_size'],
                self.config['obss_pd']['min_threshold']
            )
            return {
                'type': 'obss_pd_decrease',
                'new_obss_pd': new_obss_pd,
                'reason': 'high_retry_rate'
            }
        
        # === NEW: INTERMEDIATE RULES TO FILL DEAD ZONES ===
        
        # Priority 6: MODERATE OBSS-PD Increase (moderate CCA, low-moderate retry)
        # Fills dead zone: CCA=50-75%, retry=4-8%
        if (cca_busy > thresholds['cca_busy']['moderate'] and
            retry_rate < thresholds['retry_rate']['moderate'] and
            metrics['obss_pd'] < self.config['obss_pd']['max_threshold']):
            new_obss_pd = min(
                metrics['obss_pd'] + self.config['obss_pd']['step_size'],
                self.config['obss_pd']['max_threshold']
            )
            return {
                'type': 'obss_pd_increase',
                'new_obss_pd': new_obss_pd,
                'reason': 'moderate_cca_optimization'
            }
        
        # Priority 7: MODERATE OBSS-PD Decrease (moderate retry)
        # Fills dead zone: retry=8-18%
        if (retry_rate > thresholds['retry_rate']['moderate'] and
            metrics['obss_pd'] > self.config['obss_pd']['min_threshold']):
            new_obss_pd = max(
                metrics['obss_pd'] - self.config['obss_pd']['step_size'],
                self.config['obss_pd']['min_threshold']
            )
            return {
                'type': 'obss_pd_decrease',
                'new_obss_pd': new_obss_pd,
                'reason': 'moderate_retry_mitigation'
            }
        
        # Priority 8: Channel Change Fallback (moderate interference)
        # Even if retry isn't super high, change channel if interference is bad
        if (interference > thresholds['interference']['moderate'] and
            analysis['num_interferers'] >= 2):
            new_channel = self._find_best_channel(ap_id, analysis, metrics)
            if new_channel and new_channel != metrics['channel']:
                return {
                    'type': 'channel_change',
                    'new_channel': new_channel,
                    'reason': 'moderate_interference'
                }
        
        # Priority 9: Small Bandwidth Reduction (fallback for congestion)
        # If we're on high bandwidth and have any signs of trouble
        if (metrics['bandwidth'] > 40 and
            (cca_busy > thresholds['cca_busy']['moderate'] or 
             interference > thresholds['interference']['moderate'])):
            new_bw = self._get_reduced_bandwidth(metrics['bandwidth'])
            if new_bw:
                return {
                    'type': 'bandwidth_reduce',
                    'new_bandwidth': new_bw,
                    'reason': 'congestion_mitigation'
                }
        
        # Priority 10: Proactive Optimization (prevent complete stagnation)
        # If we've reached here, network is "stable" but we can still optimize
        # Take small actions to explore better configurations
        if self.current_step % 360 == 0:  # Once per hour
            # Try small OBSS-PD adjustments even in stable conditions
            if cca_busy > 0.4:  # CCA above 40%
                if metrics['obss_pd'] < -65:  # Room to increase
                    return {
                        'type': 'obss_pd_increase',
                        'new_obss_pd': min(metrics['obss_pd'] + 3, -62),
                        'reason': 'proactive_optimization'
                    }
            elif cca_busy < 0.3 and retry_rate > 5.0:  # Low CCA but non-zero retry
                if metrics['obss_pd'] > -80:  # Not too conservative already
                    return {
                        'type': 'obss_pd_decrease',
                        'new_obss_pd': max(metrics['obss_pd'] - 3, -82),
                        'reason': 'proactive_stabilization'
                    }
        
        return None
    
    def _find_best_channel(self, ap_id: int, analysis: Dict, metrics: Dict) -> Optional[int]:
        """Find channel with minimum predicted interference"""
        current_channel = metrics['channel']
        
        # Determine available channels based on current band
        if current_channel <= 14:
            # 2.4 GHz
            available = self.config['channels']['band_2ghz']['available']
        else:
            # 5 GHz
            available = self.config['channels']['band_5ghz']['available']
        
        # Find channel with minimum interference
        channel_scores = {}
        for channel in available:
            if channel == current_channel:
                channel_scores[channel] = analysis['current_channel_interference']
            else:
                # Estimate interference on this channel
                channel_scores[channel] = analysis['channel_interference'].get(channel, 0.0)
        
        # Find best channel
        best_channel = min(channel_scores, key=channel_scores.get)
        
        # Check if improvement is significant enough
        min_improvement = self.config['thresholds']['min_improvement']['channel_change']
        current_interference = channel_scores[current_channel]
        best_interference = channel_scores[best_channel]
        
        if current_interference > 0 and best_interference < current_interference * (1 - min_improvement):
            return best_channel
        
        return None
    
    def _get_reduced_bandwidth(self, current_bw: int) -> Optional[int]:
        """Get next smaller bandwidth (gradual reduction)"""
        bw_options = [20, 40, 80, 160]
        try:
            current_idx = bw_options.index(current_bw)
            if current_idx > 0:
                return bw_options[current_idx - 1]
        except ValueError:
            pass
        return None
    
    def _get_increased_bandwidth(self, current_bw: int, channel: int) -> Optional[int]:
        """Get next larger bandwidth (gradual increase)"""
        # Check band
        if channel <= 14:
            max_bw = 20  # 2.4 GHz limited to 20 MHz
        else:
            max_bw = 80  # 5 GHz can go up to 80 MHz
        
        bw_options = [20, 40, 80, 160]
        try:
            current_idx = bw_options.index(current_bw)
            if current_idx < len(bw_options) - 1:
                next_bw = bw_options[current_idx + 1]
                if next_bw <= max_bw:
                    return next_bw
        except ValueError:
            pass
        return None
    
    def _apply_action(self, ap_id: int, action: Dict, metrics_before: Dict) -> Dict[str, Any]:
        """
        Apply configuration action to AP.
        
        Returns:
            Result dictionary with success status
        """
        ap = self.aps[ap_id]
        action_type = action['type']
        
        try:
            if action_type == 'channel_change':
                ap.channel = action['new_channel']
                config = self.config_engine.build_channel_config(ap_id, action['new_channel'])
                self.config_engine.apply_config(config)
                self.stats['channel_changes'] += 1
                
            elif action_type in ['bandwidth_reduce', 'bandwidth_increase']:
                ap.bandwidth = action['new_bandwidth']
                # Note: ConfigEngine may not have build_bandwidth_config yet
                # For now, just update AP attribute
                self.stats['bandwidth_changes'] += 1
                
            elif action_type in ['obss_pd_increase', 'obss_pd_decrease']:
                ap.obss_pd_threshold = action['new_obss_pd']
                self.stats['obss_pd_changes'] += 1
            
            # Record action
            self.last_actions[ap_id] = {
                'step': self.current_step,  # Use self.current_step
                'action': action,
                'metrics_before': metrics_before
            }
            
            return {
                'success': True,
                'ap_id': ap_id,
                'type': action_type,
                'action': action,
                'reason': action.get('reason', '')
            }
            
        except Exception as e:
            print(f"[Fast Loop] Failed to apply action for AP {ap_id}: {e}")
            return {
                'success': False,
                'ap_id': ap_id,
                'type': action_type,
                'error': str(e)
            }
    
    def _check_cooldown(self, ap_id: int) -> bool:
        """Check if AP is in cooldown period"""
        if ap_id not in self.last_actions:
            return True
        
        last_action = self.last_actions[ap_id]
        steps_since = self.current_step - last_action['step']  # Use actual steps
        
        # Get cooldown from config
        min_steps = self.config.get('safety', {}).get('min_time_between_actions_same_ap', 60)
        
        return steps_since >= min_steps
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get controller statistics"""
        return {
            'channel_changes': self.stats['channel_changes'],
            'bandwidth_changes': self.stats['bandwidth_changes'],
            'obss_pd_changes': self.stats['obss_pd_changes'],
            'total_actions': self.stats['total_actions'],
            'rollbacks': self.stats['rollbacks']
        }
    
    def print_status(self):
        """Print controller status"""
        print("\n" + "="*60)
        print("FAST LOOP CONTROLLER STATUS")
        print("="*60)
        print(f"Total Actions: {self.stats['total_actions']}")
        print(f"  Channel Changes: {self.stats['channel_changes']}")
        print(f"  Bandwidth Changes: {self.stats['bandwidth_changes']}")
        print(f"  OBSS-PD Changes: {self.stats['obss_pd_changes']}")
        print(f"Rollbacks: {self.stats['rollbacks']}")
        print()
