from enum import IntFlag
from typing import List, Optional
from model.neighbor import Neighbor
import struct

class ReqMode(IntFlag):
    """IEEE 802.11v BSS Transition Management Request Mode bits."""
    PREFERRED_CAND_LIST_INCLUDED = 0x01
    ABRUPT_TRANSITION            = 0x02
    DISASSOC_IMMINENT            = 0x04
    BSS_TERMINATION_INCLUDED     = 0x08
    ESS_DISASSOC_IMMINENT        = 0x10
    CANDIDATE_LIST_PROVIDED_BY_STA = 0x20


class BssTmRequestBuilder:
    """
    Builds a BSS Transition Management Request (BSS_TM_REQ) command for hostapd_cli.
    Supports multiple neighbors.
    """

    def __init__(
        self,
        sta_addr: str,
        req_mode: ReqMode,
        disassoc_timer: Optional[int] = None,
        validity_interval: Optional[int] = None,
        neighbors: Optional[List["Neighbor"]] = None,
        dialog_token: Optional[int] = None,
    ):
        self.sta_addr = sta_addr
        self.req_mode = req_mode
        self.disassoc_timer = disassoc_timer
        self.validity_interval = validity_interval
        self.neighbors = neighbors or []
        self.dialog_token = dialog_token

    def build(self) -> str:
        """Build full hostapd_cli BSS_TM_REQ command string."""
        if not self.sta_addr:
            raise ValueError("sta_addr is required")

        parts = [f"BSS_TM_REQ {self.sta_addr}"]
        parts.append(f"req_mode=0x{self.req_mode:02x}")

        if self.disassoc_timer is not None:
            parts.append(f"disassoc_timer={self.disassoc_timer}")
        if self.validity_interval is not None:
            parts.append(f"validity_interval={self.validity_interval}")
        if self.dialog_token is not None:
            parts.append(f"dialog_token={self.dialog_token}")

        # encode neighbors
        for n in self.neighbors:
            parts.append(f"nei={self._encode_neighbor(n)}")

        return " ".join(parts)

    @staticmethod
    def _encode_neighbor(n: "Neighbor") -> str:
        """Encode Neighbor instance into Neighbor Report hex."""
        if not all([n.bssid, n.bssid_info, n.oper_class, n.channel, n.phy_type]):
            raise ValueError(f"Incomplete neighbor info: {n}")

        bssid_bytes = bytes(int(x, 16) for x in n.bssid.split(":"))
        raw = bssid_bytes + struct.pack(">IBBB", n.bssid_info, n.oper_class, n.channel, n.phy_type)
        return raw.hex()

    def __str__(self):
        return self.build()
