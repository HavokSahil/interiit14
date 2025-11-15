from typing import Optional, IO
from model.neighbor import Neighbor
from db.neighbor_db import NeighborDB
import sys
import socket


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



class NeighborDashboard:
    """Tabulated display of current NeighborDB contents, grouped by station MAC."""

    def __init__(self, neighbordb: NeighborDB):
        self.db = neighbordb

    def as_table(
        self,
        sort_by: Optional[str] = "channel",
        limit: Optional[int] = None
    ) -> str:

        title_line = "Neighbor Dashboard\n------------------------------"

        all_neighbors = self.db.all()  # Dict[str, list[Neighbor]]
        if not all_neighbors:
            return f"{title_line}\nNo neighbors in database."

        sections = []

        for sta_mac, neighbors in all_neighbors.items():
            header = f"Neighbors for STA: {sta_mac}"

            # Sorting
            if sort_by and hasattr(Neighbor, sort_by):
                neighbors.sort(key=lambda n: getattr(n, sort_by) or 0)

            # Limiting
            if limit:
                neighbors = neighbors[:limit]

            # Table layout
            table_headers = [
                "BSSID", "SSID", "OpClass", "Channel",
                "PHY Type", "OpClass Desc", "PHY Desc"
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

            # Compute column widths
            col_widths = [
                max(len(str(cell)) for cell in col)
                for col in zip(table_headers, *rows)
            ]

            header_line = " | ".join(h.ljust(w) for h, w in zip(table_headers, col_widths))
            sep_line = "-+-".join("-" * w for w in col_widths)
            body_lines = [
                " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
                for row in rows
            ]

            # Attach this station section
            sections.append(
                "\n".join([title_line, header, header_line, sep_line] + body_lines)
            )

        return "\n\n".join(sections)

    def show(
        self,
        sort_by: Optional[str] = "channel",
        limit: Optional[int] = None,
        pipe: Optional[IO] = None,
        replace: bool = False
    ):
        """Display dashboard to pipe OR socket OR stdout."""
        output = self.as_table(sort_by=sort_by, limit=limit)

        if replace:
            output = "\033[H\033[J" + output

        write_stream(pipe, output)
