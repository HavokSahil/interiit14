from typing import Optional, IO
from db.lmrep_db import LinkMeasurementDB
from model.measurement import LinkMeasurement
import sys

class LinkMeasurementDashboard:
    """Pretty, tabulated display of current LinkMeasurementDB contents."""

    def __init__(self, db: LinkMeasurementDB):
        self.db = db

    def as_table(self, sort_by: Optional[str] = "measurement_token", limit: Optional[int] = None) -> str:
        """Return formatted table string of all LinkMeasurements."""
        measurements = []
        for sta_mac, lms in self.db.raw().items():  # Flatten by station
            measurements.append(lms)

        title_line = "Link Measurement Dashboard\n-----------------------------------"

        if not measurements:
            return f"{title_line}\nNo Link Measurement reports."

        # Sort by attribute if valid
        if sort_by and hasattr(LinkMeasurement, sort_by):
            measurements.sort(key=lambda r: getattr(r, sort_by) or 0)

        if limit:
            measurements = measurements[:limit]

        headers = [
            "STA MAC", "Token", "TX Power(dBm)", "Link Margin(dB)",
            "RCPI", "RSNI", "Rx Ant", "Tx Ant"
        ]

        rows = []
        for r in measurements:
            # Find the STA MAC for this measurement
            rows.append([
                r.sta_mac or "-",
                str(r.measurement_token),
                str(r.tx_power if r.tx_power is not None else "-"),
                str(r.link_margin if r.link_margin is not None else "-"),
                str(r.rcpi if r.rcpi is not None else "-"),
                str(r.rsni if r.rsni is not None else "-"),
                str(r.rx_antenna_id if r.rx_antenna_id is not None else "-"),
                str(r.tx_antenna_id if r.tx_antenna_id is not None else "-")
            ])

        col_widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *rows)]
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        sep_line = "-+-".join("-" * w for w in col_widths)
        body_lines = [
            " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
            for row in rows
        ]

        return "\n".join([title_line, header_line, sep_line] + body_lines)

    def show(
        self,
        sort_by: Optional[str] = "measurement_token",
        limit: Optional[int] = None,
        pipe: Optional[IO] = None,
        replace: bool = False
    ):
        """
        Display the dashboard — print or send to a FIFO pipe.

        Args:
            sort_by: attribute to sort measurements by
            limit: number of measurements to show
            pipe: open FIFO handle (from PipePool)
        """
        output = self.as_table(sort_by=sort_by, limit=limit)

        if replace and pipe:
            output = "\033[H\033[J" + output

        try:
            if pipe:
                pipe.write(output + "\n")
                pipe.flush()
            else:
                print(output, flush=True)
        
        except Exception as e:
            sys.stderr.write(f"[LinkMeasurementDashboard] Pipe write error: {e}\n")
