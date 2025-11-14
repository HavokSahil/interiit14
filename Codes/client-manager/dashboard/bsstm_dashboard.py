from typing import Optional, IO
from db.bsstm_db import BSSTransitionResponseDB
from model.measurement import BSSTransitionResponse
import sys

class BSSTransitionResponseDashboard:
    """Pretty, tabulated display of current BSSTransitionResponseDB contents."""

    def __init__(self, db: BSSTransitionResponseDB):
        self.db = db

    def as_table(self, sort_by: Optional[str] = "dialog_token", limit: Optional[int] = None) -> str:
        """Return formatted table string of all BSS Transition responses."""
        responses = []
        for resps in self.db.all().values():  # Flatten by station
            responses.extend(resps)

        title_line = "BSS Transition Reports Dashboard\n----------------------------------"

        if not responses:
            return f"{title_line}\nNo BSS Transition responses."

        # Sort by attribute if valid
        if sort_by and hasattr(BSSTransitionResponse, sort_by):
            responses.sort(key=lambda r: getattr(r, sort_by) or 0)

        if limit:
            responses = responses[:limit]

        headers = [
            "STA MAC", "Dialog Token", "BSSID", "Status", "Disassoc Timer",
            "Validity Interval", "Num Targets"
        ]

        rows = []
        for r in responses:
            # Find the STA MAC for this response
            sta_mac = None
            for smac, resps in self.db.all().items():
                if r.dialog_token in resps and resps[r.dialog_token] == r:
                    sta_mac = smac
                    break

            rows.append([
                sta_mac or "-",
                str(r.dialog_token),
                r.bssid or "-",
                str(r.status_code),
                str(r.disassoc_timer or "-"),
                str(r.validity_interval or "-"),
                str(len(r.target_bss) if r.target_bss else 0),
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
        sort_by: Optional[str] = "dialog_token",
        limit: Optional[int] = None,
        pipe: Optional[IO] = None,
        replace: bool = False
    ):
        """
        Display the dashboard — print or send to a FIFO pipe.

        Args:
            sort_by: attribute to sort responses by
            limit: number of responses to show
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
            sys.stderr.write(f"[BSSTransitionResponseDashboard] Pipe write error: {e}\n")

