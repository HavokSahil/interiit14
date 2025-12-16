"""
Action Logger with Timestamps for Safe RL Agents.

Provides timestamped logging of all action predictions with:
- Timestamp (ISO format)
- Selected action (+2 dBm, -2 dBm, etc.)
- Reason codes
- Confidence scores
- State summary
- Q-values (if available)

Outputs can be:
- Console (real-time)
- CSV file (for analysis)
- JSON file (structured data)
"""

import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np


@dataclass
class ActionLogEntry:
    """Single action log entry with timestamp."""
    timestamp: str
    action_id: int
    action_name: str
    reason_code: str
    confidence: float
    q_values: Optional[Dict[str, float]]
    state_summary: Dict[str, Any]
    safety_status: str
    agent_type: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_csv_row(self) -> List:
        """Convert to CSV row."""
        return [
            self.timestamp,
            self.action_name,
            self.reason_code,
            f"{self.confidence:.1%}",
            self.safety_status,
            self.agent_type
        ]


class ActionLogger:
    """
    Timestamped action logger for RL agents.
    
    Logs every action prediction with:
    - Precise timestamp
    - Action details (+2 dBm power, -4 dB OBSS-PD, etc.)
    - Reason codes from explainability module
    - Confidence scores
    - State summary
    """
    
    ACTION_NAMES = {
        0: "+2 dBm Tx Power",
        1: "-2 dBm Tx Power",
        2: "+4 dB OBSS-PD",
        3: "-4 dB OBSS-PD",
        4: "No-op"
    }
    
    ACTION_DESCRIPTIONS = {
        0: "Increase transmit power by 2 dBm to improve coverage",
        1: "Decrease transmit power by 2 dBm to reduce interference",
        2: "Increase OBSS-PD threshold by 4 dB for more spatial reuse",
        3: "Decrease OBSS-PD threshold by 4 dB for more isolation",
        4: "Maintain current configuration (no change)"
    }
    
    # Feature indices
    IDX_CLIENT_COUNT = 0
    IDX_MEDIAN_RSSI = 1
    IDX_P95_RETRY = 2
    IDX_P95_PER = 3
    IDX_CHANNEL_UTIL = 4
    IDX_AVG_THROUGHPUT = 5
    IDX_EDGE_THROUGHPUT = 6
    IDX_TX_POWER = 9
    IDX_OBSS_PD = 8
    
    def __init__(
        self,
        log_dir: str = "logs/actions",
        console_output: bool = True,
        csv_output: bool = True,
        json_output: bool = True
    ):
        """
        Initialize action logger.
        
        Args:
            log_dir: Directory for log files
            console_output: Print to console
            csv_output: Write to CSV file
            json_output: Write to JSON file
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.console_output = console_output
        self.csv_output = csv_output
        self.json_output = json_output
        
        self.entries: List[ActionLogEntry] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize CSV file
        if self.csv_output:
            self.csv_path = self.log_dir / f"actions_{self.session_id}.csv"
            self._init_csv()
        
        # Initialize JSON file
        if self.json_output:
            self.json_path = self.log_dir / f"actions_{self.session_id}.json"
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        headers = [
            "TIMESTAMP", "ACTION", "REASON", 
            "CONF", "STATUS", "AGENT"
        ]
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def _get_state_summary(self, state: np.ndarray) -> Dict[str, Any]:
        """Extract key metrics from state for logging."""
        return {
            "clients": int(state[self.IDX_CLIENT_COUNT]),
            "rssi_dBm": round(float(state[self.IDX_MEDIAN_RSSI]), 1),
            "retry_pct": round(float(state[self.IDX_P95_RETRY]) * 100, 1),
            "per_pct": round(float(state[self.IDX_P95_PER]) * 100, 1),
            "util_pct": round(float(state[self.IDX_CHANNEL_UTIL]) * 100, 1),
            "edge_tput_mbps": round(float(state[self.IDX_EDGE_THROUGHPUT]), 1),
            "avg_tput_mbps": round(float(state[self.IDX_AVG_THROUGHPUT]), 1),
            "tx_power_dBm": round(float(state[self.IDX_TX_POWER]), 1),
            "obss_pd_dBm": round(float(state[self.IDX_OBSS_PD]), 1)
        }
    
    def _format_q_values(self, q_values: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
        """Format Q-values as dictionary."""
        if q_values is None:
            return None
        return {
            self.ACTION_NAMES[i]: round(float(q_values[i]), 4)
            for i in range(len(q_values))
        }
    
    def log_action(
        self,
        state: np.ndarray,
        action: int,
        reason_code: str = "UNKNOWN",
        confidence: float = 0.5,
        q_values: Optional[np.ndarray] = None,
        is_safe: bool = True,
        agent_type: str = "DQN"
    ) -> ActionLogEntry:
        """
        Log a single action prediction.
        
        Args:
            state: Current network state (15 features)
            action: Selected action (0-4)
            reason_code: Reason for action selection
            confidence: Confidence score (0-1)
            q_values: Q-values for all actions (optional)
            is_safe: Whether action passed safety check
            agent_type: Type of agent (CQL, DQN, PPO, etc.)
            
        Returns:
            ActionLogEntry object
        """
        timestamp = datetime.now().isoformat(timespec='milliseconds')
        
        entry = ActionLogEntry(
            timestamp=timestamp,
            action_id=action,
            action_name=self.ACTION_NAMES[action],
            reason_code=reason_code,
            confidence=confidence,
            q_values=self._format_q_values(q_values),
            state_summary=self._get_state_summary(state),
            safety_status="SAFE" if is_safe else "BLOCKED",
            agent_type=agent_type
        )
        
        self.entries.append(entry)
        
        # Output to console
        if self.console_output:
            self._print_entry(entry)
        
        # Append to CSV
        if self.csv_output:
            self._append_csv(entry)
        
        # Update JSON (periodically, not every entry for performance)
        if self.json_output and len(self.entries) % 10 == 0:
            self._save_json()
        
        return entry
    
    def _print_entry(self, entry: ActionLogEntry):
        """Print entry to console in formatted table."""
        print(f"{entry.timestamp} | {entry.action_name:20s} | {entry.reason_code:25s} | {entry.confidence:6.1%} | {entry.safety_status}")
    
    def _append_csv(self, entry: ActionLogEntry):
        """Append entry to CSV file."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(entry.to_csv_row())
    
    def _save_json(self):
        """Save all entries to JSON file."""
        def convert_numpy(obj):
            """Convert numpy types to Python native types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        entries_serializable = [convert_numpy(e.to_dict()) for e in self.entries]
        with open(self.json_path, 'w') as f:
            json.dump(entries_serializable, f, indent=2)
    
    def print_header(self):
        """Print header for console output."""
        print("=" * 100)
        print("SAFE RL ACTION LOG")
        print("=" * 100)
        print(f"{'TIMESTAMP':<28} | {'ACTION':<20} | {'REASON':<25} | {'CONF':>6} | STATUS")
        print("-" * 100)
    
    def print_summary(self):
        """Print summary statistics."""
        if not self.entries:
            print("No actions logged.")
            return
        
        action_counts = {}
        reason_counts = {}
        safe_count = 0
        
        for entry in self.entries:
            action_counts[entry.action_name] = action_counts.get(entry.action_name, 0) + 1
            reason_counts[entry.reason_code] = reason_counts.get(entry.reason_code, 0) + 1
            if entry.safety_status == "SAFE":
                safe_count += 1
        
        print("\n" + "=" * 60)
        print("ACTION LOG SUMMARY")
        print("=" * 60)
        print(f"Total Actions: {len(self.entries)}")
        print(f"Safety Rate: {safe_count / len(self.entries):.1%}")
        print()
        print("Action Distribution:")
        for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            pct = count / len(self.entries) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {action:20s} [{bar}] {pct:5.1f}% ({count})")
        print()
        print("Top Reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  {reason}: {count} ({count / len(self.entries):.1%})")
        print("=" * 60)
    
    def get_action_table(self, last_n: int = 10) -> str:
        """Get formatted table of recent actions."""
        entries = self.entries[-last_n:] if len(self.entries) > last_n else self.entries
        
        lines = [
            "┌" + "─" * 28 + "┬" + "─" * 22 + "┬" + "─" * 27 + "┬" + "─" * 8 + "┬" + "─" * 8 + "┐",
            f"│ {'TIMESTAMP':<26} │ {'ACTION':<20} │ {'REASON':<25} │ {'CONF':>6} │ {'STATUS':<6} │",
            "├" + "─" * 28 + "┼" + "─" * 22 + "┼" + "─" * 27 + "┼" + "─" * 8 + "┼" + "─" * 8 + "┤"
        ]
        
        for entry in entries:
            lines.append(
                f"│ {entry.timestamp:<26} │ {entry.action_name:<20} │ "
                f"{entry.reason_code:<25} │ {entry.confidence:>5.1%} │ {entry.safety_status:<6} │"
            )
        
        lines.append("└" + "─" * 28 + "┴" + "─" * 22 + "┴" + "─" * 27 + "┴" + "─" * 8 + "┴" + "─" * 8 + "┘")
        
        return "\n".join(lines)
    
    def close(self):
        """Finalize logging and save all data."""
        if self.json_output:
            self._save_json()
        
        # Save summary
        summary_path = self.log_dir / f"summary_{self.session_id}.txt"
        with open(summary_path, 'w') as f:
            f.write(self.get_action_table(last_n=100))
            f.write("\n\n")
            # Capture summary to file
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            self.print_summary()
            f.write(sys.stdout.getvalue())
            sys.stdout = old_stdout


def demo():
    """Demo the action logger."""
    print("Action Logger Demo")
    print("=" * 60)
    
    # Create logger
    logger = ActionLogger(console_output=True, csv_output=False, json_output=False)
    logger.print_header()
    
    # Simulate some actions
    import numpy as np
    
    states = [
        np.array([15, -72, 0.09, 0.03, 0.65, 80, 18, -65, -75, 15, -92, 40, 0.5, 0.4, 0.02]),
        np.array([20, -65, 0.04, 0.02, 0.75, 120, 35, -70, -75, 16, -92, 40, 0.6, 0.5, 0.03]),
        np.array([8, -68, 0.06, 0.04, 0.45, 95, 28, -72, -78, 14, -93, 20, 0.4, 0.35, 0.01]),
        np.array([25, -70, 0.08, 0.05, 0.80, 70, 15, -58, -72, 18, -91, 80, 0.7, 0.6, 0.04]),
        np.array([12, -62, 0.03, 0.02, 0.50, 150, 45, -75, -76, 15, -92, 40, 0.45, 0.4, 0.02]),
    ]
    
    actions = [0, 4, 2, 3, 4]
    reasons = ["LOW_RSSI_CLIENTS", "NETWORK_STABLE", "HIGH_CHANNEL_UTIL", "HIGH_INTERFERENCE", "NETWORK_STABLE"]
    confidences = [0.72, 0.85, 0.58, 0.63, 0.91]
    agents = ["DQN", "CQL", "DQN", "RCPO", "CQL"]
    
    for i in range(len(states)):
        logger.log_action(
            state=states[i],
            action=actions[i],
            reason_code=reasons[i],
            confidence=confidences[i],
            agent_type=agents[i]
        )
    
    print()
    print(logger.get_action_table())
    logger.print_summary()


if __name__ == "__main__":
    demo()

