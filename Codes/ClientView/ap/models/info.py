import binascii
import re
from client.models.format import *
import struct
from typing import Optional, Dict, Any


class NeighborInfo:
    """Represents one neighbor entry from hostapd's SHOW_NEIGHBOR output."""

    def __init__(self):
        self.bssid: str | None = None
        self.ssid: str | None = None
        self.nr_raw: str | None = None
        self.bssid_info: int | None = None
        self.oper_class: int | None = None
        self.channel: int | None = None
        self.phy_type: int | None = None
        self.oper_class_desc: str | None = None
        self.phy_type_desc: str | None = None
        self.subelements: str | None = None

    @staticmethod
    def make_nr(
        bssid: str,
        bssid_info: int,
        oper_class: int,
        channel: int,
        phy_type: int,
        subelements: bytes = b""
    ) -> bytes:
        """
        Construct the NR (Neighbor Report) binary field (IEEE 802.11 7.3.2.90.10).
        Args:
            bssid: str - MAC address in string format (e.g., "aa:bb:cc:dd:ee:ff")
            bssid_info: int - BSSID Information (4 bytes, little-endian)
            oper_class: int - Operating class (1 byte)
            channel: int - Channel number (1 byte)
            phy_type: int - PHY type (1 byte)
            subelements: bytes - Optional trailing subelements (variable length)
        Returns:
            bytes - NR field binary encoding
        """
        bssid_bytes = bytes.fromhex(bssid.replace(":", ""))
        bssid_info_bytes = bssid_info.to_bytes(4, "little")
        field = b"".join([
            bssid_bytes,
            bssid_info_bytes,
            oper_class.to_bytes(1, "little"),
            channel.to_bytes(1, "little"),
            phy_type.to_bytes(1, "little"),
            subelements,
        ])
        return field

    @staticmethod
    def from_line(line: str) -> "NeighborInfo":
        n = NeighborInfo()
        parts = line.strip().split()
        if not parts:
            return n

        # first token = BSSID
        n.bssid = parts[0] if re.match(r"([0-9a-f]{2}:){5}[0-9a-f]{2}", parts[0], re.I) else None

        for token in parts[1:]:
            if token.startswith("ssid="):
                n.ssid = NeighborInfo._decode_hex_ssid(token.split("=", 1)[1])
            elif token.startswith("nr="):
                n.nr_raw = token.split("=", 1)[1]
                n._parse_nr()

        return n

    def _parse_nr(self):
        if not self.nr_raw:
            return
        try:
            data = binascii.unhexlify(self.nr_raw)
        except binascii.Error:
            return

        if len(data) < 13:
            return

        self.bssid = ":".join(f"{b:02x}" for b in data[0:6])
        self.bssid_info = int.from_bytes(data[6:10], "little")
        self.oper_class = data[10]
        self.channel = data[11]
        self.phy_type = data[12]
        self.oper_class_desc = OPERATING_CLASS_TABLE.get(self.oper_class, "Unknown")
        self.phy_type_desc = PHY_TYPE_TABLE.get(self.phy_type, "Unknown")
        if len(data) > 13:
            self.subelements = data[13:].hex()

    @staticmethod
    def _decode_hex_ssid(hex_str: str) -> str:
        try:
            return bytes.fromhex(hex_str).decode("utf-8", errors="replace")
        except Exception:
            return hex_str

    def as_dict(self) -> dict[str, Any]:
        return {
            "bssid": self.bssid,
            "ssid": self.ssid,
            "bssid_info": self.bssid_info,
            "oper_class": self.oper_class,
            "oper_class_desc": self.oper_class_desc,
            "channel": self.channel,
            "phy_type": self.phy_type,
            "phy_type_desc": self.phy_type_desc,
            "subelements": self.subelements,
        }
    
    def to_nr_hex(self) -> str:
        """
        Convert this NeighborInfo to a hex string suitable for SET_NEIGHBOR command.
        Returns the hex-encoded NR (Neighbor Report) field.
        """
        if not all([self.bssid, self.bssid_info is not None, 
                   self.oper_class is not None, self.channel is not None, 
                   self.phy_type is not None]):
            raise ValueError("NeighborInfo missing required fields for NR encoding")
        
        subelements_bytes = b""
        if self.subelements:
            try:
                subelements_bytes = bytes.fromhex(self.subelements)
            except (ValueError, binascii.Error):
                pass  # If subelements can't be parsed, use empty bytes
        
        nr_bytes = self.make_nr(
            bssid=self.bssid,
            bssid_info=self.bssid_info,
            oper_class=self.oper_class,
            channel=self.channel,
            phy_type=self.phy_type,
            subelements=subelements_bytes
        )
        return nr_bytes.hex()
    
    def __str__(self):
        return (f"<Neighbor bssid={self.bssid} ssid={self.ssid} "
                f"class={self.oper_class_desc} ch={self.channel} "
                f"phy={self.phy_type_desc}>")
