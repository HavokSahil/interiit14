#!/usr/bin/env python3


class StationInfo:
    def __init__(self, mac, flags=None, rx_packets=None, tx_packets=None,
                 rx_bytes=None, tx_bytes=None, signal=None, rx_rate=None, tx_rate=None,
                 conn_time=None, min_txpower=None, max_txpower=None,
                 raw=None):
        self.mac: str = mac
        # basic info params
        self.flags: list = flags
        self.rx_packets: int = rx_packets
        self.tx_packets: int = tx_packets
        self.rx_bytes: int = rx_bytes
        self.tx_bytes: int = tx_bytes
        self.signal: int = signal
        self.rx_rate: dict = rx_rate
        self.tx_rate: dict = tx_rate
        self.conn_time: int = conn_time
        self.min_txpower: int = min_txpower
        self.max_txpower: int = max_txpower
        self.raw: dict = raw

    @staticmethod
    def _parse_flags(s: str):
        return [f for f in s.strip("[]").split("][")]

    @staticmethod
    def _parse_rate(s: str):
        parts = s.split()
        return {"rate": float(parts[0]), "mcs": int(parts[2]) if len(parts) > 2 else None }

    @staticmethod
    def from_raw(d: dict):
        return StationInfo(
            mac=d["mac"],
            flags=StationInfo._parse_flags(d["flags"]),
            rx_packets=int(d["rx_packets"]),
            tx_packets=int(d["tx_packets"]),
            rx_bytes=int(d["rx_bytes"]),
            tx_bytes=int(d["tx_bytes"]),
            signal=int(d["signal"]),
            rx_rate=StationInfo._parse_rate(d["rx_rate_info"]),
            tx_rate=StationInfo._parse_rate(d["tx_rate_info"]),
            conn_time=int(d["connected_time"]),
            min_txpower=int(d["min_txpower"]),
            max_txpower=int(d["max_txpower"]),
            raw=d
        )

    def __str__(self):
        return (
            f"Station {self.mac}\n"
            f"  Signal: {self.signal} dBm\n"
            f"  Flags: {', '.join(self.flags)}\n"
            f"  RX: {self.rx_packets} pkts, {self.rx_bytes} bytes @ {self.rx_rate['rate']} Mbps (MCS {self.rx_rate['mcs']})\n"
            f"  TX: {self.tx_packets} pkts, {self.tx_bytes} bytes @ {self.tx_rate['rate']} Mbps (MCS {self.tx_rate['mcs']})"
        )
