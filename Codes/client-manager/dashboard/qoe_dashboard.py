from typing import Optional, IO
from db.qoe_db import QoEDB
import sys

class QoEDashboard:
    """Pretty, tabulated display of current QoEDB contents."""

    def __init__(self, qoe_db: QoEDB):
        self.db = qoe_db

    def as_table(self, sort_by_qoe: bool = True, limit: Optional[int] = None) -> str:
        """Return formatted table string of all QoE entries."""
        rows = []

        title_line = "Station QoE Dashboard\n-------------------------------"

        for sta_mac, qoe in self.db.all().items():
            rows.append({
                "STA MAC": sta_mac,
                "QoE": round(qoe, 4)
            })

        if not rows:
            return f"{title_line}\nNo QoE entries."

        # Sort by QoE value descending if requested
        if sort_by_qoe:
            rows.sort(key=lambda r: r["QoE"], reverse=True)

        if limit:
            rows = rows[:limit]

        headers = ["STA MAC", "QoE"]
        col_widths = [max(len(str(row[h])) for row in [dict(zip(headers, headers))] + rows) for h in headers]

        # Build table lines
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        sep_line = "-+-".join("-" * w for w in col_widths)
        body_lines = [
            " | ".join(str(row[h]).ljust(w) for h, w in zip(headers, col_widths))
            for row in rows
        ]

        return "\n".join([title_line, header_line, sep_line] + body_lines)

    def show(
        self,
        sort_by_qoe: bool = True,
        limit: Optional[int] = None,
        pipe: Optional[IO] = None,
        replace: bool = False
    ):
        """Display the dashboard — print or send to a FIFO pipe."""
        output = self.as_table(sort_by_qoe=sort_by_qoe, limit=limit)

        if replace and pipe:
            output = "\033[H\033[J" + output

        try:
            if pipe:
                pipe.write(output + "\n")
                pipe.flush()
            else:
                print(output, flush=True)
        except Exception as e:
            sys.stderr.write(f"[QoEDashboard] Pipe write error: {e}\n")
