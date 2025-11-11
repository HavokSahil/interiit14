from __future__ import annotations

import re
from typing import Any


class StationBasicInfo:
    """Represents the status information of an associated station."""

    def __init__(self):
        self.info_dict: dict[str, Any] = {}
        self.raw: str | None = None

        # Station attributes inferred from the sample comment
        self.mac: str | None = None
        self.flags: list[str] = []
        self.aid: int | None = None
        self.capability: int | None = None
        self.listen_interval: int | None = None
        self.supported_rates: list[float] | None = None
        self.timeout_next: str | None = None
        self.rx_packets: int | None = None
        self.tx_packets: int | None = None
        self.rx_bytes: int | None = None
        self.tx_bytes: int | None = None
        self.inactive_msec: int | None = None
        self.signal: int | None = None
        self.rx_rate_info: int | None = None
        self.tx_rate_info: int | None = None
        self.connected_time: int | None = None
        self.mbo_cell_capa: int | None = None
        self.supp_op_classes: str | None = None
        self.min_txpower: int | None = None
        self.max_txpower: int | None = None
        self.ext_capab: str | None = None

    @staticmethod
    def from_content(content: str) -> "StationBasicInfo":
        info = StationBasicInfo()
        info.parse_content(content)
        return info

    @staticmethod
    def decode_supported_rates(rate_hex_str: str) -> list[float]:
        """Convert supported rate codes into Mbps values."""
        rates: list[float] = []
        for token in rate_hex_str.strip().split():
            token = token.lower()
            if not token:
                continue
            if not token.startswith("0x"):
                token = f"0x{token}"
            try:
                value = int(token, 16)
            except ValueError:
                continue
            # The lower 7 bits encode the rate in 500 Kbps increments.
            rates.append((value & 0x7F) * 0.5)
        return rates

    def parse_content(self, content: str) -> None:
        """Parse station info content in key=value format."""
        self.raw = content
        lines = content.strip().splitlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if "=" not in line:
                if self.mac is None and self._is_mac_address(line):
                    self.mac = line
                    self.info_dict["mac"] = line
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key == "flags":
                flags = re.findall(r"\[(.*?)\]", value)
                self.flags = flags
                self.info_dict[key] = flags
                continue

            if key == "supported_rates":
                rates = self.decode_supported_rates(value)
                self.supported_rates = rates
                self.info_dict[key] = rates
                continue

            converted = self._convert_value(value)
            self.info_dict[key] = converted

        self._populate_attributes()

    def _populate_attributes(self) -> None:
        """Populate instance attributes from the parsed dictionary."""
        self.aid = self._get_int("aid")
        self.capability = self._get_int("capability")
        self.listen_interval = self._get_int("listen_interval")
        self.timeout_next = self._get_str("timeout_next")
        self.rx_packets = self._get_int("rx_packets")
        self.tx_packets = self._get_int("tx_packets")
        self.rx_bytes = self._get_int("rx_bytes")
        self.tx_bytes = self._get_int("tx_bytes")
        self.inactive_msec = self._get_int("inactive_msec")
        self.signal = self._get_int("signal")
        self.rx_rate_info = self._get_int("rx_rate_info")
        self.tx_rate_info = self._get_int("tx_rate_info")
        self.connected_time = self._get_int("connected_time")
        self.mbo_cell_capa = self._get_int("mbo_cell_capa")
        self.supp_op_classes = self._get_str("supp_op_classes")
        self.min_txpower = self._get_int("min_txpower")
        self.max_txpower = self._get_int("max_txpower")
        self.ext_capab = self._get_str("ext_capab")

    def _convert_value(self, value: str) -> Any:
        """Convert string values into typed Python equivalents."""
        if value == "N/A":
            return None

        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)

        if value.lower().startswith("0x"):
            try:
                return int(value, 16)
            except ValueError:
                return value

        return value

    def _get_int(self, key: str) -> int | None:
        value = self.info_dict.get(key)
        if isinstance(value, int):
            return value
        return None

    def _get_str(self, key: str) -> str | None:
        value = self.info_dict.get(key)
        if isinstance(value, str):
            return value
        return None

    @staticmethod
    def _is_mac_address(value: str) -> bool:
        return bool(re.fullmatch(r"[0-9a-fA-F]{2}(:[0-9a-fA-F]{2}){5}", value))

    def __dict__(self) -> dict[str, Any]:
        return self.info_dict

    def __str__(self) -> str:
        attrs = [
            f"mac={self.mac}",
            f"flags={self.flags}",
            f"aid={self.aid}",
            f"signal={self.signal}",
            f"rx_packets={self.rx_packets}",
            f"tx_packets={self.tx_packets}",
            f"supported_rates={self.supported_rates}",
            f"connected_time={self.connected_time}",
        ]
        return "<StaInfo " + " ".join(attrs) + ">"
