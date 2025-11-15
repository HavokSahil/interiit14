from typing import Optional, IO
from db.bsstm_db import BSSTransitionResponseDB
from model.measurement import BSSTransitionResponse
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

class BSSTransitionResponseDashboard:
    """Prettier, boxed/tabulated display of current BSSTransitionResponseDB contents."""

    def __init__(self, db: BSSTransitionResponseDB):
        self.db = db

    def as_table(self, sort_by: Optional[str] = "dialog_token", limit: Optional[int] = None) -> str:
        responses = []
        for resps in self.db.all().values():  # Flatten by station
            responses.extend(resps)

        # Pretty, box-drawn title
        title = " BSS Transition Reports Dashboard "
        border = "═" * len(title)
        title_line = f"╔{border}╗\n║{title}║\n╚{border}╝"

        if not responses:
            return f"{title_line}\nNo BSS Transition responses."

        # Sort by attribute if valid
        if sort_by and hasattr(BSSTransitionResponse, sort_by):
            responses.sort(key=lambda r: (getattr(r, sort_by) if getattr(r, sort_by) is not None else float('-inf')), reverse=False)

        if limit:
            responses = responses[:limit]

        headers = [
            "STA MAC", "Dialog Token", "BSSID", "Status", "BSS Term. Delay",
            "Neighbors",
        ]

        # Build data rows
        rows = []
        for r in responses:
            # Determine STA MAC for this response
            sta_mac = None
            for smac, resps in self.db.all().items():
                if r in resps:
                    sta_mac = smac
                    break

            # The following assumes 'r' is a BSSTransitionResponse @dataclass instance (see prompt).
            rows.append([
                sta_mac or "-",
                str(r.dialog_token) if r.dialog_token is not None else "-",
                r.target_bssid or "-",
                str(r.status_code) if r.status_code is not None else "-",
                str(r.bss_termination_delay) if r.bss_termination_delay is not None else "-",
                str(len(r.neighbor_reports) if r.neighbor_reports else 0),
            ])

        # Calculate pretty column widths
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
        sort_by: Optional[str] = "dialog_token",
        limit: Optional[int] = None,
        pipe: Optional[IO] = None,
        replace: bool = False
    ):
        output = self.as_table(sort_by=sort_by, limit=limit)

        if replace:
            output = "\033[H\033[J" + output

        write_stream(pipe, output)
