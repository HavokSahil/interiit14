from typing import Optional, IO
from model.station import Station
from db.station_db import StationDB
import sys
import socket

from logger import Logger

# ----------------------------------------------------------
# Shared I/O helpers (pipe OR socket)
# ----------------------------------------------------------

def write_stream(out, text: str):
    if not out:
        Logger.log_info("[StationDashboard] Output stream is None, nothing to write.")
        return

    # socket-like
    if isinstance(out, socket.socket):
        try:
            Logger.log_info("[StationDashboard] Writing to socket stream.")
            out.sendall((text + "\n").encode())
        except Exception as e:
            Logger.log_info(f"[StationDashboard] Exception writing to socket: {e}")
        return

    # normal file/fifo/stdout
    try:
        Logger.log_info("[StationDashboard] Writing to file-like stream.")
        out.write(text + "\n")
        out.flush()
    except Exception as e:
        Logger.log_info(f"[StationDashboard] Exception writing to file-like stream: {e}")


def read_stream(inp) -> str:
    if not inp:
        Logger.log_info("[StationDashboard] Input stream is None, nothing to read.")
        return ""

    if isinstance(inp, socket.socket):
        try:
            Logger.log_info("[StationDashboard] Reading from socket stream.")
            data = inp.recv(1024)
            result = data.decode().strip()
            Logger.log_info(f"[StationDashboard] Read from socket: {result}")
            return result
        except Exception as e:
            Logger.log_info(f"[StationDashboard] Exception reading from socket: {e}")
            return ""

    # normal pipe
    try:
        Logger.log_info("[StationDashboard] Reading from file-like stream.")
        result = inp.readline().strip()
        Logger.log_info(f"[StationDashboard] Read from pipe: {result}")
        return result
    except Exception as e:
        Logger.log_info(f"[StationDashboard] Exception reading from file-like stream: {e}")
        return ""


# ----------------------------------------------------------
# Main Dashboard
# ----------------------------------------------------------

class StationDashboard:
    """
    Display 1 station at a time.
    Supports interactive navigation through pipe/socket input.
    """

    def __init__(self, stationdb: StationDB):
        Logger.log_info("[StationDashboard] Initializing StationDashboard.")
        self.db = stationdb
        self.index = 0

    # ------------------------------------------------------
    # Rendering
    # ------------------------------------------------------

    def _render_station(self, station: Station, idx: int, total: int) -> str:
        Logger.log_info(f"[StationDashboard] Rendering station info: idx={idx}, total={total}, mac={station.mac}")
        # Decorative boxed title
        title = " Station Dashboard "
        border = "..." * len(title)
        title_box = f"...{border}...\n...{title}...\n...{border}..."

        info_line = f"Showing {idx + 1}/{total}   MAC: {station.mac}"
        sep = "..." * len(info_line)

        body = str(station)

        return f"{title_box}\n{info_line}\n{sep}\n{body}"

    # ------------------------------------------------------
    # Public interface (same signature as original)
    # ------------------------------------------------------

    def show(
        self,
        pipe_out: Optional[IO] = None,
        pipe_in: Optional[IO] = None,
        replace: bool = False,
        **kwargs
    ):
        Logger.log_info("[StationDashboard] show() called.")
        stations = self.db.all()

        if not stations:
            Logger.log_info("[StationDashboard] No stations connected.")
            empty = (
                "Station Dashboard\n"
                "---------------------------\n"
                "No stations connected."
            )
            self._write(pipe_out, empty, replace)
            return

        # clamp index
        self.index %= len(stations)
        station = stations[self.index]
        Logger.log_info(f"[StationDashboard] Displaying station index {self.index}, MAC {station.mac}")

        # render
        out = self._render_station(station, self.index, len(stations))
        self._write(pipe_out, out, replace)

        # navigation
        if pipe_in:
            Logger.log_info("[StationDashboard] Waiting for navigation command from input.")
            cmd = read_stream(pipe_in)
            Logger.log_info(f"[StationDashboard] Received navigation command: '{cmd}'")
            self._handle_command(cmd, stations)

    # ------------------------------------------------------
    # Command handling
    # ------------------------------------------------------

    def _handle_command(self, cmd: str, stations: list[Station]):
        if not cmd:
            Logger.log_info("[StationDashboard] No command received in _handle_command.")
            return

        Logger.log_info(f"[StationDashboard] Handling command: {cmd}")

        if cmd == "next":
            self.index = (self.index + 1) % len(stations)
            Logger.log_info(f"[StationDashboard] Navigated to next station: index {self.index}")

        elif cmd == "prev":
            self.index = (self.index - 1) % len(stations)
            Logger.log_info(f"[StationDashboard] Navigated to previous station: index {self.index}")

        elif cmd.isdigit():
            n = int(cmd)
            if 0 <= n < len(stations):
                self.index = n
                Logger.log_info(f"[StationDashboard] Navigated to station {n}: MAC {stations[n].mac}")
            else:
                Logger.log_info(f"[StationDashboard] Ignored out-of-bounds index command: {n}")

        elif cmd.startswith("mac "):
            mac = cmd.split(" ", 1)[1].strip()
            found = False
            for i, st in enumerate(stations):
                if st.mac == mac:
                    self.index = i
                    found = True
                    Logger.log_info(f"[StationDashboard] Navigated by MAC: index {i}, MAC {mac}")
                    break
            if not found:
                Logger.log_info(f"[StationDashboard] MAC not found in stations: {mac}")

    # ------------------------------------------------------
    # I/O wrapper
    # ------------------------------------------------------

    def _write(self, out: Optional[IO], text: str, replace: bool):
        Logger.log_info(f"[StationDashboard] _write called, replace={replace}")
        if replace:
            text = "\033[H\033[J" + text

        if out:
            write_stream(out, text)
            Logger.log_info("[StationDashboard] Wrote output to custom output stream.")
        else:
            print(text, flush=True)
            Logger.log_info("[StationDashboard] Wrote output to stdout.")
