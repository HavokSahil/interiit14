import re
from model.station import Station
from defaults.info import *
from typing import Any
from model.mac_address import MacAddress

class StationParser:
    @staticmethod
    def from_content(content: str) -> "Station":
        station = Station()
        StationParser._parse_content(station, content)
        station.get_ip()
        return station

    def _parse_content(station: Station, content: str) -> None:
        """Parse hostapd STA info content into attributes."""
        station.raw = content
        lines = content.strip().splitlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect MAC
            if "=" not in line:
                if station.mac is None and MacAddress.is_valid(line.strip()):
                    station.mac = line
                    station.info_dict["mac"] = line
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key == "flags":
                flags = re.findall(r"\[(.*?)\]", value)
                station.flags = flags
                station.info_dict[key] = flags
                continue

            if key == "supported_rates":
                rates = StationParser._decode_supported_rates(value)
                station.supported_rates = rates
                station.info_dict[key] = rates
                continue

            if key == "capability":
                station.capability = StationParser._decode_capability(value)
                continue
                
            if key == "ext_capab":
                station.ext_capab = StationParser._decode_ext_capab(value)
                continue

            if key == "supp_op_classes":
                station.supp_op_classes = StationParser._decode_supported_op_classes(value)
                continue

            station.info_dict[key] = StationParser._convert_value(value)
        StationParser._populate_attributes(station)

    @staticmethod
    def _decode_supported_rates(rate_hex_str: str) -> list[float]:
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

    @staticmethod
    def _decode_capability(capability: str | None) -> list[str]:
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
    def _decode_ext_capab(ext_capab: str | None) -> list[str]:
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
    def _decode_supported_op_classes(supp_op_classes: str | None) -> list[dict[str, str]]:
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

    @staticmethod
    def _convert_value(value: str) -> Any:
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

    def _get_int(station, key: str) -> int | None:
        val = station.info_dict.get(key)
        return val if isinstance(val, int) else None

    def _get_str(station, key: str) -> str | None:
        val = station.info_dict.get(key)
        return val if isinstance(val, str) else None

    @staticmethod
    def _is_mac_address(value: str) -> bool:
        return bool(re.fullmatch(r"[0-9a-fA-F]{2}(:[0-9a-fA-F]{2}){5}", value))

    def _populate_attributes(station: Station) -> None:
        """Fill instance vars from parsed dictionary automatically."""
        for key, value in station.info_dict.items():
            if hasattr(station, key):
                setattr(station, key, value)