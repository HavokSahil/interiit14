from typing import Optional, IO
from db.nbrank_db import NeighborRankingDB
from model.neighbor import Neighbor
import sys
import socket

def write_stream(pipe_or_sock, text: str):
    """Write text to either a pipe (file) or a socket transparently."""
    if not pipe_or_sock:
        return

    # Socket support
    if isinstance(pipe_or_sock, socket.socket):
        try:
            pipe_or_sock.sendall((text + "\n").encode())
        except Exception:
            pass
        return

    # FIFO / file / stdout
    try:
        pipe_or_sock.write(text + "\n")
        pipe_or_sock.flush()
    except Exception:
        pass

def rcpi_to_db(rcpi_val):
    """Convert RCPI (0..220) to dBm. RCPI to RSSI dBm: dBm = RCPI/2 - 110. Show N/A if invalid."""
    if rcpi_val is None or not isinstance(rcpi_val, int) or rcpi_val < 0 or rcpi_val > 220:
        return "N/A"
    return f"{rcpi_val/2 - 110:.1f} dBm"

def rsni_to_db(rsni_val):
    """Convert RSNI (0..255) to dB. RSNI to dB: value = RSNI/2 dB. Show N/A if invalid."""
    if rsni_val is None or not isinstance(rsni_val, int) or rsni_val < 0 or rsni_val > 255:
        return "N/A"
    return f"{rsni_val/2:.1f} dB"

class NeighborRankingDashboard:
    """Prettier, tabulated display of current NeighborRankingDB contents."""

    def __init__(self, nr_db: NeighborRankingDB):
        self.db = nr_db

    def as_table(self, limit: Optional[int] = None) -> str:
        rows = []

        # Prettified title
        title = " Neighbor Ranking Dashboard "
        border = "═" * len(title)
        title_line = f"╔{border}╗\n║{title}║\n╚{border}╝"

        for sta_mac, neighbors in self.db.all().items():
            for rank, neighbor in enumerate(neighbors, start=1):
                # Don't show unprintable SSID
                if neighbor.ssid != "N/A" and neighbor.ssid.isprintable():
                    rcpi_val = getattr(neighbor, "rcpi", None)
                    rsni_val = getattr(neighbor, "rsni", None)
                    rows.append({
                        "STA MAC": sta_mac,
                        "Rank": rank,
                        "BSSID": neighbor.bssid,
                        "SSID": neighbor.ssid.replace("IITP", "Campus")[:18],
                        "RCPI": rcpi_to_db(rcpi_val),
                        "RSNI": rsni_to_db(rsni_val),
                        "Channel": getattr(neighbor, "channel", "N/A"),
                    })

        if not rows:
            return f"{title_line}\nNo neighbor rankings."

        if limit:
            rows = rows[:limit]

        headers = ["STA MAC", "Rank", "BSSID", "SSID", "RCPI", "RSNI", "Channel"]

        # Calculate column widths (account for header + content)
        col_widths = [
            max(len(str(row.get(h, ""))) for row in [dict(zip(headers, headers))] + rows)
            for h in headers
        ]

        # Table drawing
        hsep = "─"
        vsep = "│"
        joint = "┼"

        # Render header line with box drawing
        header_line = " │ ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        sep_line = "─┼─".join("─" * w for w in col_widths)

        # Prettier body lines
        body_lines = [
            " │ ".join(str(row[h]).ljust(w) for h, w in zip(headers, col_widths))
            for row in rows
        ]

        table = "\n".join([
            title_line,
            header_line,
            sep_line,
            *body_lines
        ])
        return table

    def show(
        self,
        limit: Optional[int] = None,
        pipe: Optional[IO] = None,
        replace: bool = False
    ):
        """Display the dashboard — print or send to FIFO or socket."""
        output = self.as_table(limit=limit)

        if replace:
            output = "\033[H\033[J" + output

        write_stream(pipe, output)
