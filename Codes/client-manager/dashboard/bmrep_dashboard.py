from typing import Optional, Union, IO
from db.bmrep_db import BeaconMeasurementDB
from model.measurement import BeaconMeasurement
import sys

class BeaconMeasurementDashboard:
    """Pretty, tabulated display of current BeaconMeasurementDB contents."""

    def __init__(self, bm_db: BeaconMeasurementDB):
        self.db = bm_db

    def as_table(self, sort_by: Optional[str] = "rcpi", limit: Optional[int] = None) -> str:
        """Return formatted table string of all BeaconReports in the database."""
        rows = []

        # Flatten all BeaconReports from all measurements
        for sta_mac, bms in self.db.all().items():
            for bm in bms:
                for report in bm.beacon_reports:
                    rows.append({
                        "STA MAC": sta_mac,
                        "Token": bm.measurement_token,
                        "SSID": report.parse_ssid(),
                        "BSSID": report.bssid,
                        "Channel": report.channel_number,
                        "RCPI": report.rssi_dbm,
                        "RSNI": report.snr_db,
                        "Start TSF": report.measurement_start_time,
                        "Duration TU": report.measurement_duration,
                        "Antenna": report.antenna_id
                    })

        if not rows:
            return "No beacon measurements."

        # Sort by a valid column
        if sort_by and sort_by in rows[0]:
            rows.sort(key=lambda r: r[sort_by] or 0)

        if limit:
            rows = rows[:limit]

        headers = ["STA MAC", "Token", "SSID" ,"BSSID", "Channel", "RCPI", "RSNI", "Start TSF", "Duration TU", "Antenna"]
        col_widths = [max(len(str(row[h])) for row in [dict(zip(headers, headers))] + rows) for h in headers]

        # Build table lines
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        sep_line = "-+-".join("-" * w for w in col_widths)
        body_lines = [
            " | ".join(str(row[h]).ljust(w) for h, w in zip(headers, col_widths))
            for row in rows
        ]

        return "\n".join([header_line, sep_line] + body_lines)

    def show(
        self,
        sort_by: Optional[str] = "rcpi",
        limit: Optional[int] = None,
        pipe: Optional[IO] = None,
        replace: bool = False
    ):
        """Display the dashboard — print or send to a FIFO pipe."""
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
            sys.stderr.write(f"[BeaconMeasurementDashboard] Pipe write error: {e}\n")
