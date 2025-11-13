from typing import Optional, Union, IO
from model.station import Station
from db.station_db import StationDB
import sys


class StationDashboard:
    """Pretty, tabulated display of current StationDB contents."""

    def __init__(self, stationdb: StationDB):
        self.db = stationdb

    def as_table(self, sort_by: Optional[str] = "mac", limit: Optional[int] = None) -> str:
        """Return formatted table string of all stations."""
        stations = self.db.all()

        if not stations:
            return "No stations connected."

        # Sort stations if requested and attribute exists
        if sort_by and hasattr(Station, sort_by):
            stations.sort(key=lambda s: getattr(s, sort_by) or 0)

        if limit:
            stations = stations[:limit]

        headers = [
            "MAC", "Signal(dBm)", "TxRate(Mbps)", "RxRate(Mbps)",
            "TxPkt", "RxPkt", "TxBytes", "RxBytes", "Inactive(ms)", "Connected(s)"
        ]

        rows = []
        for s in stations:
            rows.append([
                s.mac or "-",
                str(s.signal or "-"),
                str(s.tx_bitrate or "-"),
                str(s.rx_bitrate or "-"),
                str(s.tx_packets or "-"),
                str(s.rx_packets or "-"),
                str(s.tx_bytes or "-"),
                str(s.rx_bytes or "-"),
                str(s.inactive_msec or "-"),
                str(s.connected_sec or s.connected_time or "-"),
            ])

        col_widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *rows)]

        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        sep_line = "-+-".join("-" * w for w in col_widths)
        body_lines = [
            " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
            for row in rows
        ]

        return "\n".join([header_line, sep_line] + body_lines)

    def show(
        self,
        sort_by: Optional[str] = "mac",
        limit: Optional[int] = None,
        pipe: Optional[IO] = None,
        replace: bool = False
    ):
        """
        Display the dashboard — print or send to a FIFO pipe.

        Args:
            sort_by: attribute to sort stations by
            limit: number of stations to show
            pipe: open FIFO handle (from PipePool)
            replace: overwrite output (for live updates)
        """
        output = self.as_table(sort_by=sort_by, limit=limit)

        if replace:
        # Clear the screen before printing
            output = "\033[H\033[J" + output


        if pipe:
            try:
                if replace:
                    pipe.write(output)
                else:
                    pipe.write(output + "\n")
                pipe.flush()
            except Exception as e:
                sys.stderr.write(f"[StationDashboard] Pipe write error: {e}\n")
        else:
            print(output, flush=True)
