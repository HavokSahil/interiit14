from typing import Optional, IO
from db.nbrank_db import NeighborRankingDB
from model.neighbor import Neighbor
import sys

class NeighborRankingDashboard:
    """Pretty, tabulated display of current NeighborRankingDB contents."""

    def __init__(self, nr_db: NeighborRankingDB):
        self.db = nr_db

    def as_table(self, limit: Optional[int] = None) -> str:
        """Return formatted table string of all neighbor rankings."""
        rows = []

        title_line = "Neighbor Ranking Dashboard\n----------------------------------"

        # Flatten all neighbors
        for sta_mac, neighbors in self.db.all().items():
            for rank, neighbor in enumerate(neighbors, start=1):
                if neighbor.ssid != "N/A" and neighbor.ssid.isprintable():
                    rows.append({
                        "STA MAC": sta_mac,
                        "Rank": rank,
                        "BSSID": neighbor.bssid,
                        "SSID": neighbor.ssid,
                        "RCPI": getattr(neighbor, "rcpi", "N/A"),
                        "RSNI": getattr(neighbor, "rsni", "N/A"),
                        "Channel": getattr(neighbor, "channel", "N/A"),
                    })

        if not rows:
            return f"{title_line}\nNo neighbor rankings."

        if limit:
            rows = rows[:limit]

        headers = ["STA MAC", "Rank", "BSSID", "SSID", "RCPI", "RSNI", "Channel"]
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
        limit: Optional[int] = None,
        pipe: Optional[IO] = None,
        replace: bool = False
    ):
        """Display the dashboard — print or send to a FIFO pipe."""
        output = self.as_table(limit=limit)

        if replace and pipe:
            output = "\033[H\033[J" + output

        try:
            if pipe:
                pipe.write(output + "\n")
                pipe.flush()
            else:
                print(output, flush=True)
        except Exception as e:
            sys.stderr.write(f"[NeighborRankingDashboard] Pipe write error: {e}\n")
