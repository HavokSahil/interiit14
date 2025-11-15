from typing import Optional, IO, Dict, List
from db.qoe_db import QoEDB
from metrics.qoe import QoE
import sys
import socket
from datetime import datetime


def write_stream(pipe_or_sock, text: str):
    """Write text to either a pipe (file) or a socket transparently."""
    if not pipe_or_sock:
        return
    # Socket case
    if isinstance(pipe_or_sock, socket.socket):
        try:
            pipe_or_sock.sendall((text + "\n").encode())
        except Exception:
            pass
        return
    # FIFO/file/stdout
    try:
        pipe_or_sock.write(text + "\n")
        pipe_or_sock.flush()
    except Exception:
        pass


class QoEDashboard:
    """
    Comprehensive QoE dashboard with component breakdown, trends, and alerts.
    
    Features:
    - Overall QoE scores with color-coded status
    - Component breakdown (connectivity, signal, throughput, reliability, latency, activity)
    - Historical trends (improving/declining/stable)
    - Volatility indicators
    - Statistics summary
    - Problem station alerts
    """
    
    # ANSI color codes for terminal output
    COLORS = {
        "red": "\033[91m",
        "yellow": "\033[93m",
        "green": "\033[92m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "magenta": "\033[95m",
        "white": "\033[97m",
        "gray": "\033[90m",
        "bold": "\033[1m",
        "reset": "\033[0m",
    }
    
    # Box drawing characters
    BOX = {
        "tl": "╔", "tr": "╗", "bl": "╚", "br": "╝",
        "h": "═", "v": "║",
        "t": "╤", "b": "╧", "l": "╟", "r": "╢",
        "cross": "┼",
        "vl": "├", "vr": "┤", "ht": "┬", "hb": "┴",
    }
    
    def __init__(self, qoedb: QoEDB, qoe_engine: Optional[QoE] = None):
        """
        Args:
            qoe_db: QoE database for scores
            qoe_engine: Optional QoE engine for component details and history
        """
        self.db = qoedb
        self.engine = qoe_engine
        self.use_color = True  # Can be disabled for plain text output
    
    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.use_color:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"
    
    def _qoe_color(self, qoe: float) -> str:
        """Get color based on QoE score."""
        if qoe >= 0.8:
            return "green"
        elif qoe >= 0.6:
            return "cyan"
        elif qoe >= 0.4:
            return "yellow"
        else:
            return "red"
    
    def _qoe_status(self, qoe: float) -> str:
        """Get status emoji/symbol based on QoE score."""
        if qoe >= 0.8:
            return "✓"  # Excellent
        elif qoe >= 0.6:
            return "○"  # Good
        elif qoe >= 0.4:
            return "△"  # Fair
        else:
            return "✗"  # Poor
    
    def _trend_symbol(self, trend: str) -> str:
        """Get symbol for trend direction."""
        if trend == "improving":
            return self._colorize("↑", "green")
        elif trend == "declining":
            return self._colorize("↓", "red")
        elif trend == "stable":
            return self._colorize("→", "cyan")
        else:
            return self._colorize("?", "gray")
    
    def _bar_chart(self, value: float, width: int = 20, filled_char: str = "█", empty_char: str = "░") -> str:
        """Create a simple bar chart for a 0-1 value."""
        filled = int(value * width)
        empty = width - filled
        bar = filled_char * filled + empty_char * empty
        
        # Color based on value
        if value >= 0.8:
            return self._colorize(bar, "green")
        elif value >= 0.6:
            return self._colorize(bar, "cyan")
        elif value >= 0.4:
            return self._colorize(bar, "yellow")
        else:
            return self._colorize(bar, "red")
    
    # ================================================================
    # Table Rendering
    # ================================================================
    
    def _render_header(self, title: str, width: int) -> str:
        """Render a box header."""
        inner_width = width - 4
        title_padded = title.center(inner_width)
        return (
            f"{self.BOX['tl']}{self.BOX['h'] * (width - 2)}{self.BOX['tr']}\n"
            f"{self.BOX['v']} {self._colorize(title_padded, 'bold')} {self.BOX['v']}\n"
            f"{self.BOX['bl']}{self.BOX['h'] * (width - 2)}{self.BOX['br']}"
        )
    
    def _render_table_row(self, columns: List[str], widths: List[int], separator: str = "│") -> str:
        """Render a single table row."""
        cells = [col.ljust(w) for col, w in zip(columns, widths)]
        return f" {separator} ".join(cells)
    
    # ================================================================
    # Main Tables
    # ================================================================

    def as_overview_table(self, sort_by_qoe: bool = True, limit: Optional[int] = None) -> str:
        """
        Compact overview table with current QoE scores only.
        History, trends, and volatility are removed.
        """
        rows = [{"mac": sta, "qoe": qoe} for sta, qoe in self.db.all().items()]

        if not rows:
            return self._render_header("Station QoE Dashboard", 60) + "\n\nNo QoE entries."

        # Sort by QoE descending
        if sort_by_qoe:
            rows.sort(key=lambda r: r["qoe"], reverse=True)
        if limit:
            rows = rows[:limit]

        # Build table
        lines = []
        header_width = 60
        lines.append(self._render_header("Station QoE Overview", header_width))
        lines.append("")

        # Headers
        headers = ["Station MAC", "Status", "QoE"]
        widths = [20, 10, 10]  # adjusted widths for proper alignment
        lines.append(self._render_table_row(headers, widths))
        lines.append("─" * header_width)

        for row in rows:
            mac = row["mac"]
            qoe = row["qoe"]
            # Use a simple status symbol without non-printable characters
            if qoe >= 0.8:
                status = "Excellent"
            elif qoe >= 0.6:
                status = "Good"
            elif qoe >= 0.4:
                status = "Fair"
            else:
                status = "Poor"

            qoe_str = f"{qoe:.3f}"

            cols = [mac, status, qoe_str]
            lines.append(self._render_table_row(cols, widths))

        return "\n".join(lines)

    
    def as_detailed_table(self, sort_by_qoe: bool = True, limit: Optional[int] = 10) -> str:
        """
        Detailed table with component breakdown for each station.
        """
        if not self.engine:
            return "Detailed view requires QoE engine with component tracking."
        
        rows = []
        for mac in self.db.all().keys():
            components = self.engine.get_components(mac)
            if components:
                rows.append({
                    "mac": mac,
                    "overall": components.overall,
                    "conn": components.connectivity,
                    "signal": components.signal_quality,
                    "tput": components.throughput,
                    "rel": components.reliability,
                    "lat": components.latency,
                    "act": components.activity,
                })
        
        if not rows:
            return "No component data available."
        
        # Sort
        if sort_by_qoe:
            rows.sort(key=lambda r: r["overall"], reverse=True)
        if limit:
            rows = rows[:limit]
        
        lines = []
        header_width = 120
        lines.append(self._render_header("Station QoE Component Breakdown", header_width))
        lines.append("")
        
        # Headers
        headers = ["Station MAC", "Overall", "Connect", "Signal", "Throughput", "Reliability", "Latency", "Activity"]
        widths = [17, 7, 7, 7, 10, 11, 7, 8]
        lines.append(self._render_table_row(headers, widths))
        lines.append("─" * header_width)
        
        # Rows
        for row in rows:
            mac = row["mac"]
            overall = self._colorize(f"{row['overall']:.3f}", self._qoe_color(row['overall']))
            conn = f"{row['conn']:.3f}"
            signal = f"{row['signal']:.3f}"
            tput = f"{row['tput']:.3f}"
            rel = f"{row['rel']:.3f}"
            lat = f"{row['lat']:.3f}"
            act = f"{row['act']:.3f}"
            
            cols = [mac, overall, conn, signal, tput, rel, lat, act]
            lines.append(self._render_table_row(cols, widths))
        
        return "\n".join(lines)
    
    def as_visual_breakdown(self, mac: str) -> str:
        """
        Visual bar chart breakdown for a single station.
        """
        if not self.engine:
            return "Visual breakdown requires QoE engine with component tracking."
        
        components = self.engine.get_components(mac)
        if not components:
            return f"No component data for {mac}"
        
        lines = []
        header_width = 70
        lines.append(self._render_header(f"QoE Breakdown: {mac}", header_width))
        lines.append("")
        
        # Overall
        overall_bar = self._bar_chart(components.overall, width=30)
        lines.append(f"Overall QoE:    {overall_bar}  {components.overall:.3f}")
        lines.append("")
        
        # Components
        component_data = [
            ("Connectivity", components.connectivity),
            ("Signal Quality", components.signal_quality),
            ("Throughput", components.throughput),
            ("Reliability", components.reliability),
            ("Latency", components.latency),
            ("Activity", components.activity),
        ]
        
        for name, value in component_data:
            bar = self._bar_chart(value, width=30)
            lines.append(f"{name:15} {bar}  {value:.3f}")
        
        # Trend if available
        history = self.engine.get_history(mac)
        if history:
            lines.append("")
            lines.append(f"Trend:          {self._trend_symbol(history.trend)} {history.trend}")
            lines.append(f"Average (60s):  {history.average:.3f}")
            lines.append(f"Volatility:     {history.volatility:.3f}")
        
        return "\n".join(lines)
    
    def as_statistics_summary(self) -> str:
        """
        Verbose statistics summary based only on current QoE values.
        No history or trend info.
        """
        qoe_values = list(self.db.all().values())
        total_stations = len(qoe_values)

        lines = []
        header_width = 60
        lines.append(self._render_header("Current QoE Statistics", header_width))
        lines.append("")

        lines.append(f"Total Stations: {total_stations}")

        if total_stations == 0:
            lines.append("No QoE entries available.")
            return "\n".join(lines)

        avg_qoe = sum(qoe_values) / total_stations
        min_qoe = min(qoe_values)
        max_qoe = max(qoe_values)
        std_dev = (sum((v - avg_qoe) ** 2 for v in qoe_values) / total_stations) ** 0.5

        lines.append(f"Average QoE: {self._colorize(f'{avg_qoe:.3f}', self._qoe_color(avg_qoe))}")
        lines.append(f"Minimum QoE: {self._colorize(f'{min_qoe:.3f}', self._qoe_color(min_qoe))}")
        lines.append(f"Maximum QoE: {self._colorize(f'{max_qoe:.3f}', self._qoe_color(max_qoe))}")
        lines.append(f"Std Deviation: {std_dev:.3f}")

        # Distribution
        excellent = sum(1 for v in qoe_values if v >= 0.8)
        good = sum(1 for v in qoe_values if 0.6 <= v < 0.8)
        fair = sum(1 for v in qoe_values if 0.4 <= v < 0.6)
        poor = sum(1 for v in qoe_values if v < 0.4)

        lines.append("")
        lines.append("QoE Distribution:")
        lines.append(f"  {self._colorize('Excellent (≥0.8)', 'green')}: {excellent} ({excellent/total_stations*100:.1f}%)")
        lines.append(f"  {self._colorize('Good (0.6-0.8)', 'cyan')}:      {good} ({good/total_stations*100:.1f}%)")
        lines.append(f"  {self._colorize('Fair (0.4-0.6)', 'yellow')}:      {fair} ({fair/total_stations*100:.1f}%)")
        lines.append(f"  {self._colorize('Poor (<0.4)', 'red')}:         {poor} ({poor/total_stations*100:.1f}%)")

        return "\n".join(lines)

    
    def as_alerts(self, threshold: float = 0.4, limit: int = 5) -> str:
        """
        Show stations with QoE below threshold (problem stations).
        """
        problem_stations = [
            (mac, qoe) for mac, qoe in self.db.all().items()
            if qoe < threshold
        ]
        
        if not problem_stations:
            return self._colorize("✓ No stations below alert threshold", "green")
        
        problem_stations.sort(key=lambda x: x[1])  # Worst first
        problem_stations = problem_stations[:limit]
        
        lines = []
        header_width = 70
        lines.append(self._render_header(f"QoE Alerts (< {threshold:.2f})", header_width))
        lines.append("")
        
        for mac, qoe in problem_stations:
            status = self._colorize(self._qoe_status(qoe), "red")
            qoe_str = self._colorize(f"{qoe:.3f}", "red")
            
            # Show component breakdown if available
            if self.engine:
                components = self.engine.get_components(mac)
                if components:
                    # Find weakest component
                    comp_values = [
                        ("conn", components.connectivity),
                        ("signal", components.signal_quality),
                        ("tput", components.throughput),
                        ("rel", components.reliability),
                        ("lat", components.latency),
                        ("act", components.activity),
                    ]
                    weakest = min(comp_values, key=lambda x: x[1])
                    lines.append(f"{status} {mac}  QoE: {qoe_str}  Weakest: {weakest[0]}={weakest[1]:.2f}")
                else:
                    lines.append(f"{status} {mac}  QoE: {qoe_str}")
            else:
                lines.append(f"{status} {mac}  QoE: {qoe_str}")
        
        return "\n".join(lines)
    
    # ================================================================
    # Combined Dashboard Views
    # ================================================================
    
    def as_compact_dashboard(self, limit: int = 10) -> str:
        """Single-screen compact dashboard with key info."""
        sections = [
            self.as_statistics_summary(),
            "",
            self.as_alerts(threshold=0.4, limit=5),
            "",
            self.as_overview_table(sort_by_qoe=True, limit=limit),
        ]
        return "\n".join(sections)
    
    def as_full_dashboard(self, limit: int = 20) -> str:
        """Full detailed dashboard."""
        sections = [
            self.as_statistics_summary(),
            "",
            self.as_alerts(threshold=0.4, limit=5),
            "",
            self.as_detailed_table(sort_by_qoe=True, limit=limit),
        ]
        return "\n".join(sections)
    
    # ================================================================
    # Output Methods
    # ================================================================
    
    def show(
        self,
        mode: str = "compact",
        sort_by_qoe: bool = True,
        limit: Optional[int] = None,
        pipe: Optional[IO] = None,
        replace: bool = False,
        disable_color: bool = False,
    ):
        """
        Display dashboard to pipe OR socket OR stdout.
        
        Args:
            mode: "compact", "full", "overview", "detailed", "stats", or "alerts"
            sort_by_qoe: Sort stations by QoE score
            limit: Max number of stations to show
            pipe: Output destination (socket, file, or None for stdout)
            replace: Clear screen before output (ANSI escape codes)
            disable_color: Disable color output
        """
        # Disable colors for non-terminal output or if requested
        if disable_color or (pipe and not isinstance(pipe, socket.socket)):
            self.use_color = False
        
        # Select output mode
        if mode == "compact":
            output = self.as_compact_dashboard(limit=limit or 10)
        elif mode == "full":
            output = self.as_full_dashboard(limit=limit or 20)
        elif mode == "overview":
            output = self.as_overview_table(sort_by_qoe=sort_by_qoe, limit=limit)
        elif mode == "detailed":
            output = self.as_detailed_table(sort_by_qoe=sort_by_qoe, limit=limit or 10)
        elif mode == "stats":
            output = self.as_statistics_summary()
        elif mode == "alerts":
            output = self.as_alerts(threshold=0.4, limit=limit or 10)
        else:
            output = self.as_compact_dashboard(limit=limit or 10)
        
        # Add timestamp
        timestamp = f"\nUpdated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        output = output + timestamp
        
        # Clear screen if requested
        if replace:
            output = "\033[H\033[J" + output
        
        # Send to output
        if pipe:
            write_stream(pipe, output)
        else:
            print(output)