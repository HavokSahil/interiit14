from __future__ import annotations

from filecmp import clear_cache
import re
from typing import Any
from client.models.format import *


class StationBasicInfo:
    """Represents the status information of an associated station."""

    def __init__(self):
        self.info_dict: dict[str, Any] = {}
        self.raw: str | None = None

        # Primary identifiers
        self.mac: str | None = None
        self.flags: list[str] = []

        # Basic STA info
        self.aid: int | None = None
        self.capability: list[str] | None = None
        self.listen_interval: int | None = None
        self.supported_rates: list[float] | None = None
        self.timeout_next: str | None = None

        # Traffic counters
        self.rx_packets: int | None = None
        self.tx_packets: int | None = None
        self.rx_bytes: int | None = None
        self.tx_bytes: int | None = None
        self.rx_airtime: int | None = None
        self.tx_airtime: int | None = None
        self.beacons_count: int | None = None
        self.rx_drop_misc: int | None = None
        self.backlog_packets: int | None = None
        self.backlog_bytes: int | None = None
        self.fcs_error_count: int | None = None
        self.beacon_loss_count: int | None = None
        self.expected_throughput: int | None = None
        self.tx_retry_count: int | None = None
        self.tx_retry_failed: int | None = None

        # Rate / airtime info
        self.tx_bitrate: int | None = None
        self.rx_bitrate: int | None = None
        self.tx_duration: int | None = None
        self.rx_duration: int | None = None

        # PHY layer fields
        self.rx_mcs: int | None = None
        self.tx_mcs: int | None = None
        self.rx_vhtmcs: int | None = None
        self.tx_vhtmcs: int | None = None
        self.rx_he_nss: int | None = None
        self.tx_he_nss: int | None = None
        self.rx_vht_nss: int | None = None
        self.tx_vht_nss: int | None = None
        self.rx_dcm: int | None = None
        self.tx_dcm: int | None = None
        self.rx_guard_interval: int | None = None
        self.tx_guard_interval: int | None = None

        # Signal / timing
        self.signal: int | None = None
        self.avg_signal: int | None = None
        self.avg_beacon_signal: int | None = None
        self.avg_ack_signal: int | None = None
        self.inactive_msec: int | None = None
        self.connected_sec: int | None = None

        # Extra info
        self.rx_rate_info: int | None = None
        self.tx_rate_info: int | None = None
        self.connected_time: int | None = None
        self.mbo_cell_capa: int | None = None
        self.supp_op_classes: str | None = None
        self.min_txpower: int | None = None
        self.max_txpower: int | None = None
        self.ext_capab: list[str] | None = None

    @staticmethod
    def from_content(content: str) -> "StationBasicInfo":
        info = StationBasicInfo()
        info.parse_content(content)
        return info

    @staticmethod
    def decode_supported_rates(rate_hex_str: str) -> list[float]:
        """Convert supported rate codes into Mbps values."""
        rates = []
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
            rates.append((value & 0x7F) * 0.5)
        return rates

    def parse_content(self, content: str) -> None:
        """Parse hostapd STA info content into attributes."""
        self.raw = content
        lines = content.strip().splitlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect MAC
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

            if key == "capability":
                self.capability = self.decode_capability(value)
                continue
                
            if key == "ext_capab":
                self.ext_capab = self.decode_ext_capab(value)
                continue

            if key == "supp_op_classes":
                self.supp_op_classes = self.decode_supported_op_classes(value)
                continue

            self.info_dict[key] = self._convert_value(value)
        self._populate_attributes()

    @staticmethod
    def decode_capability(capability: str | None) -> list[str]:
        """Return list of supported capability flags."""
        if capability is None:
            return []
        capability = int(capability, 16)
        caps = []
        for bit, name in CAPABILITY_FLAGS.items():
            if capability & bit:
                caps.append(name)
        return caps

    @staticmethod
    def decode_ext_capab(ext_capab: str | None) -> list[str]:
        """Decode the variable-length ext_capab hex string into feature names."""
        if not ext_capab:
            return []
        try:
            data = bytes.fromhex(ext_capab)
        except ValueError:
            return []

        caps = []
        for bit, name in EXT_CAPABILITY_FLAGS.items():
            byte_idx = bit // 8
            bit_pos = bit % 8
            if byte_idx < len(data) and (data[byte_idx] & (1 << bit_pos)):
                caps.append(name)
        return caps

    @staticmethod
    def decode_supported_op_classes(supp_op_classes: str | None) -> list[dict[str, str]]:
        """
        Decode the 'supp_op_classes' hex string into a list of supported operating classes.
        Example input: '51515354737475767778797a7b7c7d7e7f808182'
        """
        if not supp_op_classes:
            return []

        try:    
            data = bytes.fromhex(supp_op_classes)
        except ValueError:
            return []

        results = []
        for b in data:
            op_class = b
            description = OPERATING_CLASS_TABLE.get(op_class, "Unknown")
            results.append({
                "op_class": op_class,
                "description": description,
            })
        return results


    def _convert_value(self, value: str) -> Any:
        """Convert textual values to Python types."""
        if value == "N/A":
            return None
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)
        if value.lower().startswith("0x"):
            try:
                return int(value, 16)
            except ValueError:
                pass
        return value

    def _get_int(self, key: str) -> int | None:
        val = self.info_dict.get(key)
        return val if isinstance(val, int) else None

    def _get_str(self, key: str) -> str | None:
        val = self.info_dict.get(key)
        return val if isinstance(val, str) else None

    @staticmethod
    def _is_mac_address(value: str) -> bool:
        return bool(re.fullmatch(r"[0-9a-fA-F]{2}(:[0-9a-fA-F]{2}){5}", value))

    def _populate_attributes(self) -> None:
        """Fill instance vars from parsed dictionary automatically."""
        for key, value in self.info_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        return self.info_dict

    def __str__(self) -> str:
        """String representation with all available details."""
        lines = []
        lines.append("=== StationBasicInfo ===")
        for attr in vars(self):
            # Skip info_dict itself if desired, it's a dict dump of all
            if attr == "info_dict":
                continue
            value = getattr(self, attr)
            lines.append(f"{attr}: {value!r}")
        # Optionally show info_dict at the end
        lines.append("info_dict: {")
        for k, v in self.info_dict.items():
            lines.append(f"    {k!r}: {v!r},")
        lines.append("}")
        return "\n".join(lines)
