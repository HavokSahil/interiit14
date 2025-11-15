from typing import Optional, IO
from db.lmrep_db import LinkMeasurementDB
from model.measurement import LinkMeasurement
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

    # FIFO / file-like stream
    try:
        pipe_or_sock.write(text + "\n")
        pipe_or_sock.flush()
    except Exception:
        pass



class LinkMeasurementDashboard:
    """Pretty, boxed/tabulated display of current LinkMeasurementDB contents."""

    def __init__(self, db: LinkMeasurementDB):
        self.db = db

    def as_table(self, sort_by: Optional[str] = "measurement_token", limit: Optional[int] = None) -> str:
        # Flatten all LinkMeasurement instances in DB (may be a list per station)
        measurements = []
        for sta_mac, lm_list in self.db.raw().items():  # Flatten by station
            if isinstance(lm_list, list):
                measurements.extend(lm_list)
            else:
                measurements.append(lm_list)

        # Boxed, pretty title
        title = " Link Measurement Dashboard "
        border = "═" * len(title)
        title_line = f"╔{border}╗\n║{title}║\n╚{border}╝"

        if not measurements:
            return f"{title_line}\nNo Link Measurement reports."

        # Sorting (default by 'measurement_token', falls back to other fields if given)
        if sort_by and hasattr(LinkMeasurement, sort_by):
            measurements.sort(key=lambda r: (getattr(r, sort_by) if getattr(r, sort_by) is not None else float('-inf')), reverse=False)

        if limit:
            measurements = measurements[:limit]

        # Column headers and column-friendly field extraction
        headers = [
            "STA MAC", "Token", "TX Power (dBm)", "Link Margin (dB)",
            "RCPI", "RSNI", "Rx Ant", "Tx Ant"
        ]

        # Data rows
        rows = []
        for r in measurements:
            rows.append([
                r.sta_mac or "-",
                str(r.measurement_token) if r.measurement_token is not None else "-",
                str(r.tx_power) if r.tx_power is not None else "-",
                str(r.link_margin) if r.link_margin is not None else "-",
                str(r.rcpi) if r.rcpi is not None else "-",
                str(r.rsni) if r.rsni is not None else "-",
                str(r.rx_antenna_id) if r.rx_antenna_id is not None else "-",
                str(r.tx_antenna_id) if r.tx_antenna_id is not None else "-",
            ])

        # Calculate pretty column widths (headers+content)
        col_widths = [
            max(len(str(cell)) for cell in [header] + [row[i] for row in rows])
            for i, header in enumerate(headers)
        ]

        # Fancy table box-drawing
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

        def make_line(parts, sep="─", joint="┼"):
            return joint.join(sep * w for w in col_widths)

        # Table header line (with boxes)
        header_line = " │ ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        sep_line = "─┼─".join("─" * w for w in col_widths)

        # Data rows
        body_lines = [
            " │ ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
            for row in rows
        ]

        # Assemble with boxed header
        result = [
            title_line,
            header_line,
            sep_line,
            *body_lines
        ]
        return "\n".join(result)

    def show(
        self,
        sort_by: Optional[str] = "measurement_token",
        limit: Optional[int] = None,
        pipe: Optional[IO] = None,
        replace: bool = False
    ):
        output = self.as_table(sort_by=sort_by, limit=limit)

        if replace:
            output = "\033[H\033[J" + output

        # Unified writer
        write_stream(pipe, output)
