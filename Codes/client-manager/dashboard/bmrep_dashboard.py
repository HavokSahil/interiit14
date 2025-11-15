from typing import Optional, IO
from db.bmrep_db import BeaconMeasurementDB
from logger import Logger
from model.measurement import BeaconMeasurement
import sys
import socket

def write_stream(pipe_or_sock, text: str):
    """Write text to either a pipe (file) or a socket transparently."""
    if pipe_or_sock is None:
        return

    if isinstance(pipe_or_sock, socket.socket):
        try:
            pipe_or_sock.sendall((text + "\n").encode())
        except Exception as e:
            Logger.log_info(f"write_stream: Failed to write to the socket {e}")
        return

    try:
        pipe_or_sock.write(text + "\n")
        pipe_or_sock.flush()
    except Exception:
        Logger.log_info("write_stream: Failed to write to the stream")


class BeaconMeasurementDashboard:
    """Prettier, tabulated display of current BeaconMeasurementDB contents."""

    def __init__(self, bm_db: BeaconMeasurementDB):
        self.db = bm_db

    def as_table(self, sort_by: Optional[str] = "rcpi", limit: Optional[int] = None) -> str:
        rows = []
        title = " Beacon Measurement Dashboard "
        border = "═" * len(title)
        title_line = f"╔{border}╗\n║{title}║\n╚{border}╝"

        for sta_mac, bms in self.db.raw().items():
            for bm in bms:
                for report in bm.beacon_reports:
                    ssid = report.parse_ssid()
                    if ssid != "N/A" and ssid.isprintable():
                        rows.append({
                            "STA MAC": sta_mac,
                            "Token": bm.measurement_token,
                            "SSID": ssid.replace("IITP", "Campus")[:16],
                            "BSSID": report.bssid,
                            "Channel": report.channel_number,
                            "RCPI": report.rssi_dbm,
                            "RSNI": report.snr_db,
                            "Start TSF": report.measurement_start_time,
                            "Duration TU": report.measurement_duration,
                            "Antenna": report.antenna_id
                        })

        if not rows:
            return f"{title_line}\nNo beacon measurements."

        headers = ["STA MAC", "Token", "SSID", "BSSID", "Channel",
                   "RCPI", "RSNI", "Start TSF", "Duration TU", "Antenna"]

        # Sorting
        sort_by_col = sort_by if sort_by and sort_by.upper() in (h.upper() for h in headers) else "RCPI"
        for canonical in headers:
            if canonical.lower() == (sort_by or "").lower():
                sort_by_col = canonical
                break
        if sort_by_col in rows[0]:
            rows.sort(key=lambda r: (r[sort_by_col] if r[sort_by_col] is not None else float("-inf")), reverse=(sort_by_col in ("RCPI", "RSNI")))

        if limit:
            rows = rows[:limit]

        # Calculate column widths
        col_widths = [
            max(len(str(row.get(h, "-"))) for row in [dict(zip(headers, headers))] + rows)
            for h in headers
        ]

        # Fancy table characters
        hsep = "─"
        vsep = "│"
        joint = "┼"
        head_joint_left = "├"
        head_joint_right = "┤"
        head_joint_center = "┼"
        corner_top_left = "╔"
        corner_top_right = "╗"
        corner_bot_left = "╚"
        corner_bot_right = "╝"

        # Table header
        header_line = " │ ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        sep_line = "─┼─".join("─" * w for w in col_widths)
        body_lines = [
            " │ ".join(str(row.get(h, "-")).ljust(w) for h, w in zip(headers, col_widths)) for row in rows
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
        sort_by: Optional[str] = "rcpi",
        limit: Optional[int] = None,
        pipe: Optional[IO] = None,
        replace: bool = False
    ):
        output = self.as_table(sort_by=sort_by, limit=limit)

        if replace:
            output = "\033[H\033[J" + output

        write_stream(pipe, output)
