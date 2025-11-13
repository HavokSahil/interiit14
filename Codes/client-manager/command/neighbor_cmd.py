from typing import Dict, Optional, Any
from model.neighbor import Neighbor
import struct, re

class NeighborCommandBuilder:
    """
    Build and parse hostapd 'SET_NEIGHBOR' commands.

    Example with NeighborInfo:
        neighbor = NeighborInfo.from_line("00:11:22:33:44:55 ssid=MyNet nr=...")
        req = NeighborCommandBuilder(
            neighbor=neighbor,
            lci={"latitude": 37.4219, "longitude": -122.0840, "altitude": 10.0},
            civic={"country": "US", "city": "MountainView"},
            stationary=True,
            bss_parameter=5
        )
        print(req.build())

    Example with dict (backward compatible):
        req = NeighborCommandBuilder(
            bssid="00:11:22:33:44:55",
            ssid="MyNet",
            nr={"bssid_info": 0x1234, "reg_class": 81, "channel": 6, "phy_type": 7},
            lci={"latitude": 37.4219, "longitude": -122.0840, "altitude": 10.0},
            civic={"country": "US", "city": "MountainView"},
            stationary=True,
            bss_parameter=5
        )
        print(req.build())
    """

    def __init__(
        self,
        neighbor: Optional[Neighbor] = None,
        bssid: Optional[str] = None,
        ssid: Optional[str] = None,
        nr: Optional[Dict[str, Any]] = None,
        lci: Optional[Dict[str, Any]] = None,
        civic: Optional[Dict[str, Any]] = None,
        stationary: bool = False,
        bss_parameter: Optional[int] = None,
    ):
        if neighbor is not None:
            # Use NeighborInfo if provided
            self.neighbor = neighbor
            self.bssid = neighbor.bssid
            self.ssid = neighbor.ssid
            self.nr = None  # Will use neighbor.to_nr_hex() instead
        else:
            # Backward compatible: use dict-based approach
            if bssid is None or ssid is None or nr is None:
                raise ValueError("Either 'neighbor' must be provided, or 'bssid', 'ssid', and 'nr' must all be provided")
            self.neighbor = None
            self.bssid = bssid
            self.ssid = ssid
            self.nr = nr
        
        self.lci = lci
        self.civic = civic
        self.stationary = stationary
        self.bss_parameter = bss_parameter

    def build(self) -> str:
        """Builds the full 'SET_NEIGHBOR' command string with proper encoding."""
        if self.bssid is None:
            raise ValueError("BSSID is required to build SET_NEIGHBOR command")
        
        parts = [f"SET_NEIGHBOR {self.bssid}"]

        # SSID (hex encoded)
        if self.ssid:
            ssid_hex = self._encode_ssid_hex(self.ssid)
            parts.append(f'ssid={ssid_hex}')

        # NR (Neighbor Report) -> use NeighborInfo if available, otherwise encode from dict
        if self.neighbor is not None:
            nr_hex = self.neighbor.to_nr_hex()
        elif self.nr is not None:
            nr_hex = self._encode_neighbor_report(self.nr)
        else:
            raise ValueError("Either 'neighbor' or 'nr' must be provided")
        
        parts.append(f'nr={nr_hex}')

        # Optional subelements
        if self.lci:
            lci_hex = self._encode_lci(self.lci)
            parts.append(f'lci={lci_hex}')
        if self.civic:
            civic_hex = self._encode_civic(self.civic)
            parts.append(f'civic={civic_hex}')
        if self.stationary:
            parts.append("stat")
        if self.bss_parameter is not None:
            parts.append(f"bss_parameter={self.bss_parameter}")

        return " ".join(parts)

    @staticmethod
    def _encode_neighbor_report(cfg: Dict[str, Any]) -> str:
        """
        Encode Neighbor Report fields into a minimal valid binary hex representation.
        IEEE 802.11k format: [BSSID(6)][BSSID Info(4)][Reg Class(1)][Channel(1)][PHY Type(1)]
        """
        bssid = bytes(int(x, 16) for x in cfg.get("bssid", "00:00:00:00:00:00").split(":"))
        bssid_info = cfg.get("bssid_info", 0)
        reg_class = cfg.get("reg_class", 81)
        channel = cfg.get("channel", 1)
        phy_type = cfg.get("phy_type", 7)
        raw = bssid + struct.pack(">IBBB", bssid_info, reg_class, channel, phy_type)
        return raw.hex()

    @staticmethod
    def _encode_lci(lci: Dict[str, Any]) -> str:
        """
        Very simplified LCI encoding:
        [latitude(8 bytes double)][longitude(8 bytes double)][altitude(4 bytes float)]
        """
        lat = float(lci.get("latitude", 0.0))
        lon = float(lci.get("longitude", 0.0))
        alt = float(lci.get("altitude", 0.0))
        raw = struct.pack(">ddf", lat, lon, alt)
        return raw.hex()

    @staticmethod
    def _encode_civic(civic: Dict[str, Any]) -> str:
        """
        Simplified civic info encoding:
        country + city as ASCII, prefixed with lengths.
        Example: [len_country][country_bytes][len_city][city_bytes]
        """
        country = civic.get("country", "??").encode("ascii", "ignore")
        city = civic.get("city", "").encode("ascii", "ignore")
        raw = struct.pack(">B", len(country)) + country + struct.pack(">B", len(city)) + city
        return raw.hex()

    @staticmethod
    def _encode_ssid_hex(ssid: str) -> str:
        """Encode SSID as a hex string."""
        return ssid.encode('utf-8').hex()

    def __str__(self) -> str:
        return self.build()
