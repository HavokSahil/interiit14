from typing import Optional, IO
from model.neighbor import Neighbor
from db.neighbor_db import NeighborDB
import sys

DEFAULT_STA_MAC = "00:00:00:00:00:00"

class NeighborDashboard:
    """Tabulated display of current NeighborDB contents, grouped by station MAC."""

    def __init__(self, neighbordb: NeighborDB):
        self.db = neighbordb

    def as_table(self, sort_by: Optional[str] = "channel", limit: Optional[int] = None) -> str:
        """Return formatted table string of all neighbors grouped by STA MAC."""

        title_line = "Neighbor Dashboard\n------------------------------"
    
        all_neighbors = self.db.all()  # Dict[str, list[Neighbor]]
        if not all_neighbors:
            return f"{title_line}\nNo neighbors in database."

        sections = []
        for sta_mac, neighbors in all_neighbors.items():
            header = f"Neighbors for STA: {sta_mac}"
            # Sort neighbors if attribute exists
            if sort_by and hasattr(Neighbor, sort_by):
                neighbors.sort(key=lambda n: getattr(n, sort_by) or 0)
            if limit:
                neighbors = neighbors[:limit]

            # Table headers
            table_headers = [
                "BSSID", "SSID", "OpClass", "Channel", "PHY Type",
                "OpClass Desc", "PHY Desc"
            ]

            rows = []
            for n in neighbors:
                rows.append([
                    n.bssid or "-",
                    n.ssid or "-",
                    str(n.oper_class or "-"),
                    str(n.channel or "-"),
                    str(n.phy_type or "-"),
                    n.oper_class_desc or "-",
                    n.phy_type_desc or "-",
                ])

            
            col_widths = [max(len(str(cell)) for cell in col) for col in zip(table_headers, *rows)]
            header_line = " | ".join(h.ljust(w) for h, w in zip(table_headers, col_widths))
            sep_line = "-+-".join("-" * w for w in col_widths)
            body_lines = [" | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)) for row in rows]

            sections.append("\n".join([title_line, header, header_line, sep_line] + body_lines))

        return "\n\n".join(sections)

    def show(
        self,
        sort_by: Optional[str] = "channel",
        limit: Optional[int] = None,
        pipe: Optional[IO] = None,
        replace: bool = False
    ):
        """
        Display the dashboard — print or send to a FIFO pipe.

        Args:
            sort_by: sort key
            limit: number of entries per STA
            pipe: open FIFO handle
            replace: overwrite previous output
        """
        output = self.as_table(sort_by=sort_by, limit=limit)

        if replace and pipe:
            # Clear terminal on pipe side using ANSI codes
            output = "\033[H\033[J" + output

        try:
            if pipe:
                pipe.write(output)
                pipe.flush()
            else:
                print(output, flush=True)
        except Exception as e:
            sys.stderr.write(f"[NeighborDashboard] Pipe write error: {e}\n")
