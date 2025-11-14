from scapy.layers.dot11 import *
import re
from typing import Tuple, Optional
from enum import Enum

from db.lmrep_db import LinkMeasurementDB
from db.bmrep_db import BeaconMeasurementDB
from db.bsstm_db import BSSTransitionResponseDB
from db.nbrank_db import NeighborRankingDB
from db.qoe_db import QoEDB
from db.station_db import StationDB
from parser.measurement_parser import parse_beacon_measurement, parse_link_measurement, parse_bss_tm_response
from logger import Logger


class MgmtType(Enum):
    LM_RESPONSE     = 3
    BM_RESPONSE     = 1
    BSS_TM_RESPONSE = 8
    UNKNOWN         = -1

class RxMux:
    """Multiplex incoming 802.11 frames into specific storages based on Action frame category."""

    def __init__(self) -> None:
        self.lm_db = LinkMeasurementDB()
        self.bm_db = BeaconMeasurementDB()
        self.bss_tm_db = BSSTransitionResponseDB()
        self.st_db = StationDB()
        self.qe_db = QoEDB()
        self.nr_db = NeighborRankingDB()
        Logger.log_info(f"{self.lm_db} {self.bm_db} {self.bss_tm_db}")

    @staticmethod
    def parse_buf_string(s: str) -> bytes:
        """Extract buf= hex string and return bytes."""
        match = re.search(r'buf=([0-9a-fA-F]+)', s)
        if not match:
            Logger.log_err("No buf=hex_string found in input")
            raise ValueError("No buf=hex_string found in input")
        return bytes.fromhex(match.group(1))

    def cleardb(self, mac: str):
        self.lm_db.remove(mac)
        self.bm_db.remove(mac)
        self.st_db.remove(mac)
        self.qe_db.remove(mac)
        self.nr_db.remove(mac)


    def mux(self, frame_bytes: bytes) -> Optional[Tuple[MgmtType, RadioTap]]:
        """Parse a raw 802.11 frame and store in the appropriate DB."""
#        try:
        dot11 = Dot11(frame_bytes)
            # Only consider Management Action frames
        if dot11.type != 0 or dot11.subtype != 13:
            Logger.log_info(f"Ignored frame: type={dot11.type}, subtype={dot11.subtype}")
            return
        
        payload = bytes(dot11.payload)
        if not payload:
            Logger.log_info("Empty Action frame payload")
            return

        category = payload[0]
        action = payload[1]

        if category != 5 and category != 10:
            Logger.log_info("Frame is not of Radio measurement Report and not WNM report")
            return

        if category != 5:
            Logger.log_info("Frame is not of Radio measurement Report")

        sta_mac = dot11.addr2  # Transmitter MAC
        Logger.log_info(f"Received Action frame: sta_mac={sta_mac}, category={category}")

        # Beacon Measurement Response
        if action == MgmtType.BM_RESPONSE.value:
            Logger.log_info(f"Beacon Measurement Response from {sta_mac}")
            if self.bm_db is not None:
                bmobj = parse_beacon_measurement(dot11)
                self.bm_db.add(bmobj, sta_mac=sta_mac)
            else:
                Logger.log_info("mux: self.bm_db was None")
            return (MgmtType.BM_RESPONSE, dot11)

        # Link Measurement Response
        elif action == MgmtType.LM_RESPONSE.value:
            Logger.log_info(f"Link Measurement Response from {sta_mac}")
            if self.lm_db is not None:
                lmobj = parse_link_measurement(dot11)
                lmobj.sta_mac = sta_mac
                self.lm_db.add(lmobj, sta_mac=sta_mac)
            else:
                Logger.log_info("mux: self.lm_db was None")
            return (MgmtType.LM_RESPONSE, dot11)

        # BSS Transition Management Response
        elif action == MgmtType.BSS_TM_RESPONSE.value:
            Logger.log_info(f"BSS Transition Response from {sta_mac}")
            if self.bss_tm_db is not None:
                bss_tm_obj = parse_bss_tm_response(dot11)
                self.bss_tm_db.add(bss_tm_obj, sta_mac)
            else:
                Logger.log_info("mux: self.bss_tm_db was None")
            return (MgmtType.BSS_TM_RESPONSE, dot11)

        else:
            Logger.log_info(f"Unknown Action category {category} from {sta_mac}")
            return (MgmtType.UNKNOWN, dot11)

#        except Exception as e:
            #Logger.log_err(f"Failed to parse frame: {e}")
            #return None
