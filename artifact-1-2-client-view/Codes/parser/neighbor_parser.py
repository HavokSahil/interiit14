from db.neighbor_db import NeighborDB
from model.measurement import BeaconReport
from model.neighbor import Neighbor
import re
from typing import Any
import binascii
from defaults.info import *

class NeighborParser:
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
    def from_line(line: str) -> "Neighbor":
        n = Neighbor()
        parts = line.strip().split()
        if not parts:
            return n

        # first token = BSSID
        n.bssid = parts[0] if re.match(r"([0-9a-f]{2}:){5}[0-9a-f]{2}", parts[0], re.I) else None

        for token in parts[1:]:
            if token.startswith("ssid="):
                n.ssid = NeighborParser._decode_hex_ssid(token.split("=", 1)[1])
            elif token.startswith("nr="):
                n.nr_raw = token.split("=", 1)[1]
                NeighborParser._parse_nr(n)

        return n

    @staticmethod
    def _parse_nr(neib: Neighbor):
        if not neib.nr_raw:
            return
        try:
            data = binascii.unhexlify(neib.nr_raw)
        except binascii.Error:
            return

        if len(data) < 13:
            return

        neib.bssid = ":".join(f"{b:02x}" for b in data[0:6])
        neib.bssid_info = int.from_bytes(data[6:10], "little")
        neib.oper_class = data[10]
        neib.channel = data[11]
        neib.phy_type = data[12]
        neib.oper_class_desc = OPERATING_CLASS_TABLE.get(neib.oper_class, "Unknown")
        neib.phy_type_desc = PHY_TYPE_TABLE.get(neib.phy_type, "Unknown")
        if len(data) > 13:
            neib.subelements = data[13:].hex()

    @staticmethod
    def _decode_hex_ssid(hex_str: str) -> str:
        try:
            return bytes.fromhex(hex_str).decode("utf-8", errors="replace")
        except Exception:
            return hex_str

    @staticmethod
    def as_dict(neib: Neighbor) -> dict[str, Any]:
        return {
            "bssid": neib.bssid,
            "ssid": neib.ssid,
            "bssid_info": neib.bssid_info,
            "oper_class": neib.oper_class,
            "oper_class_desc": neib.oper_class_desc,
            "channel": neib.channel,
            "phy_type": neib.phy_type,
            "phy_type_desc": neib.phy_type_desc,
            "subelements": neib.subelements,
        }
    
    @staticmethod
    def to_nr_hex(neib: Neighbor) -> str:
        """
        Convert this NeighborInfo to a hex string suitable for SET_NEIGHBOR command.
        Returns the hex-encoded NR (Neighbor Report) field.
        """
        if not all([neib.bssid, neib.bssid_info is not None, 
                   neib.oper_class is not None, neib.channel is not None, 
                   neib.phy_type is not None]):
            raise ValueError("NeighborInfo missing required fields for NR encoding")
        
        subelements_bytes = b""
        if neib.subelements:
            try:
                subelements_bytes = bytes.fromhex(neib.subelements)
            except (ValueError, binascii.Error):
                pass  # If subelements can't be parsed, use empty bytes
        
        nr_bytes = NeighborParser.make_nr(
            bssid=neib.bssid,
            bssid_info=neib.bssid_info,
            oper_class=neib.oper_class,
            channel=neib.channel,
            phy_type=neib.phy_type,
            subelements=subelements_bytes
        )
        return nr_bytes.hex()


def neighbor_from_beacon_report(br: BeaconReport) -> Neighbor:
    """
    Convert a BeaconReport into a Neighbor object.
    Only fields available from BeaconReport are filled.
    """
    n = Neighbor()
    # obtain the neighbor database
    nbdb = NeighborDB()
    nbrep = nbdb.get(br.ssid)

    if nbrep:
        nbrep.rcpi = br.rcpi
        nbrep.rsni = br.rsni
        return nbrep

    n.bssid = br.bssid
    n.ssid = br.parse_ssid()
    n.channel = br.channel_number
    n.oper_class = br.operating_class

    n.oper_class_desc = br.operating_class
    n.phy_type_desc = 0

    n.subelements = None # NOTE: this is optional
    n.bssid_info = 0 # TODO: put something here too
    n.phy_type = 0 # TODO: put something here (0 means unknown)

    n.rsni = br.rsni
    n.rcpi = br.rcpi

    n.nr_raw = NeighborParser.to_nr_hex(n)
    return n
