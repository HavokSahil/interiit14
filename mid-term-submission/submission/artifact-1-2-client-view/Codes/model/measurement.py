from dataclasses import dataclass, field
from typing import Optional, Dict, List
from logger import Logger

@dataclass
class BeaconReport:
    operating_class: int
    channel_number: int
    measurement_start_time: int
    measurement_duration: int
    reported_frame_info: int
    rcpi: int
    rsni: int
    bssid: str
    antenna_id: int
    parent_tsf: int
    reported_frame_body: Optional[bytes] = None
    ssid: Optional[str] = None

    @property
    def rssi_dbm(self) -> float:
        return self.rcpi / 2 - 110

    @property
    def snr_db(self) -> float:
        return self.rsni / 2 - 10

    def parse_ssid(self) -> Optional[str]:
        if self.ssid is not None:
            return self.ssid

        if not self.reported_frame_body:
            Logger.log_info("parse_ssid: reported_frame_body is None")
            self.ssid = "N/A"
            return self.ssid

        # Skip fixed fields: Timestamp (8) + Beacon Interval (2) + Capabilities (2)
        idx = 12
        Logger.log_info(f"parse_ssid: Starting IE parse at offset {idx}")

        while idx + 2 <= len(self.reported_frame_body):
            ie_id = self.reported_frame_body[idx]
            ie_len = self.reported_frame_body[idx + 1]
            Logger.log_info(f"parse_ssid: ie_id={ie_id}, ie_len={ie_len}, idx={idx}")

            if idx + 2 + ie_len > len(self.reported_frame_body):
                Logger.log_info("parse_ssid: IE length exceeds buffer, stopping parse")
                self.ssid = "N/A"
                break

            if ie_id == 0:  # SSID IE
                try:
                    self.ssid = self.reported_frame_body[idx + 2: idx + 2 + ie_len].decode("utf-8", errors="ignore")
                    Logger.log_info(f"parse_ssid: Found SSID='{self.ssid}'")
                except Exception as e:
                    Logger.log_info(f"parse_ssid: Failed to decode SSID: {e}")
                    self.ssid = "N/A"
                return self.ssid

            idx += 2 + ie_len

        Logger.log_info("parse_ssid: No SSID IE found")
        self.ssid = "N/A"
        return "N/A"

    def to_dict(self) -> dict:
        """
        Dump the complete state of the BeaconReport.
        """
        return {
            "operating_class": self.operating_class,
            "channel_number": self.channel_number,
            "measurement_start_time": self.measurement_start_time,
            "measurement_duration": self.measurement_duration,
            "reported_frame_info": self.reported_frame_info,
            "rcpi": self.rcpi,
            "rsni": self.rsni,
            "bssid": self.bssid,
            "antenna_id": self.antenna_id,
            "parent_tsf": self.parent_tsf,
            "reported_frame_body": self.reported_frame_body.hex() if self.reported_frame_body is not None else None,
            "ssid": self.ssid,
            "rssi_dbm": self.rssi_dbm,
            "snr_db": self.snr_db,
        }


@dataclass
class BeaconMeasurement:
    sta_mac: Optional[str] = None
    measurement_token: int = None
    dialog_token: int = None
    beacon_reports: List[BeaconReport] = None

    def __post_init__(self):
        if self.beacon_reports is None:
            self.beacon_reports = []
        else:
            for report in self.beacon_reports:
                report.parse_ssid()

    @staticmethod
    def from_bytes(bts: bytes, sta_mac: str = None) -> "BeaconMeasurement":
        if len(bts) < 2:
            Logger.log_info("from_bytes: Byte sequence too short")
            raise ValueError("Byte sequence too short")

        action = bts[0]
        dialog_token = bts[1]
        Logger.log_info(f"from_bytes: action={action}, dialog_token={dialog_token}")

        idx = 2
        beacon_reports = []

        while idx + 2 <= len(bts):
            tag_number = bts[idx]
            tag_length = bts[idx + 1]
            Logger.log_info(f"from_bytes: tag_number={tag_number}, tag_length={tag_length}")

            if idx + 2 + tag_length > len(bts):
                Logger.log_info("from_bytes: Tag length exceeds buffer, stopping parse")
                break

            if tag_number == 39:  # Measurement Report
                tag_data = bts[idx + 2: idx + 2 + tag_length]

                if len(tag_data) < 29:
                    Logger.log_info("from_bytes: Not enough data for Beacon Report, skipping")
                    idx += 2 + tag_length
                    continue

                measurement_token = tag_data[0]
                measurement_mode = tag_data[1]
                measurement_type = tag_data[2]
                Logger.log_info(f"from_bytes: measurement_token={measurement_token}, measurement_mode={measurement_mode}, measurement_type={measurement_type}")

                if measurement_type != 5:
                    Logger.log_info(f"from_bytes: Not a Beacon Report (type={measurement_type}), skipping")
                    idx += 2 + tag_length
                    continue

                operating_class = tag_data[3]
                channel_number = tag_data[4]
                measurement_start_time = int.from_bytes(tag_data[5:13], 'little')
                measurement_duration = int.from_bytes(tag_data[13:15], 'little')
                reported_frame_info = tag_data[15]
                rcpi = tag_data[16]
                rsni = tag_data[17]
                bssid = ":".join(f"{b:02x}" for b in tag_data[18:24])
                antenna_id = tag_data[24]
                parent_tsf = int.from_bytes(tag_data[25:29], 'little')
                Logger.log_info(f"from_bytes: op_class={operating_class}, channel={channel_number}, start_time={measurement_start_time}, duration={measurement_duration}, rep_frame_info={reported_frame_info}, rcpi={rcpi}, rsni={rsni}, bssid={bssid}, antenna_id={antenna_id}, parent_tsf={parent_tsf}")

                # Parse Reported Frame Body subelement dynamically
                reported_frame_body = None
                sub_idx = 29
                while sub_idx + 2 <= len(tag_data):
                    sub_id = tag_data[sub_idx]
                    sub_len = tag_data[sub_idx + 1]
                    Logger.log_info(f"from_bytes: subelement_id={sub_id}, subelement_len={sub_len}")
                    if sub_idx + 2 + sub_len > len(tag_data):
                        Logger.log_info("from_bytes: Subelement length exceeds buffer, stopping subelement parse")
                        break
                    if sub_id == 1:  # Reported Frame Body
                        reported_frame_body = tag_data[sub_idx + 2: sub_idx + 2 + sub_len]
                        Logger.log_info(f"from_bytes: Captured Reported Frame Body, length={len(reported_frame_body)}")
                        break
                    sub_idx += 2 + sub_len

                beacon_reports.append(BeaconReport(
                    operating_class=operating_class,
                    channel_number=channel_number,
                    measurement_start_time=measurement_start_time,
                    measurement_duration=measurement_duration,
                    reported_frame_info=reported_frame_info,
                    rcpi=rcpi,
                    rsni=rsni,
                    bssid=bssid,
                    antenna_id=antenna_id,
                    parent_tsf=parent_tsf,
                    reported_frame_body=reported_frame_body
                ))

            idx += 2 + tag_length

        Logger.log_info(f"from_bytes: Parsed {len(beacon_reports)} Beacon Reports")
        return BeaconMeasurement(
            sta_mac=sta_mac,
            measurement_token=beacon_reports[0].rcpi if beacon_reports else None,
            dialog_token=dialog_token,
            beacon_reports=beacon_reports
        )

    def to_dict(self) -> dict:
        """
        Dump the complete state of the BeaconMeasurement.
        """
        return {
            "sta_mac": self.sta_mac,
            "measurement_token": self.measurement_token,
            "dialog_token": self.dialog_token,
            "beacon_reports": [br.to_dict() for br in self.beacon_reports] if self.beacon_reports else [],
        }

@dataclass
class LinkMeasurement:
    """802.11k Link Measurement Report"""
    sta_mac: Optional[str] = None
    measurement_token: int = None
    tx_power: Optional[int] = None             # dBm
    link_margin: Optional[int] = None          # dB
    rx_antenna_id: Optional[int] = None
    tx_antenna_id: Optional[int] = None
    rcpi: Optional[int] = None
    rsni: Optional[int] = None
    bssid: Optional[str] = None
    operating_class: Optional[int] = None
    channel_number: Optional[int] = None
    parent_tsf: Optional[int] = None

    @property
    def rssi_dbm(self) -> Optional[float]:
        return (self.rcpi / 2 - 110) if self.rcpi is not None else None

    @staticmethod
    def from_bytes(bts: bytes) -> "LinkMeasurement":
        if len(bts) < 10:
            raise ValueError("Byte sequence too short to parse LinkMeasurement")

        action = bts[0]
        measurement_token = bts[1]  # usually the second byte
        tpc_element_id = bts[2]
        tpc_element_length = bts[3]
        tx_power = bts[4]
        link_margin = bts[5]
        rx_antenna_id = bts[6]
        tx_antenna_id = bts[7]
        rcpi = bts[8]  # often the RCPI
        rsni = bts[9]  # often the RSNI

        # Optional fields (e.g., BSSID, operating class, channel number) can be extracted if present
        bssid = None
        operating_class = None
        channel_number = None
        parent_tsf = None

        idx = 10
        if len(bts) >= idx + 6:
            sta_mac = ":".join(f"{b:02x}" for b in bts[idx:idx+6])
            idx += 6
        if len(bts) > idx + 2:
            operating_class = bts[idx]
            channel_number = bts[idx+1]
            idx += 2
        if len(bts) >= idx + 8:
            # parent_tsf is typically 8 bytes, big endian
            parent_tsf = int.from_bytes(bts[idx:idx+8], "big")

        return LinkMeasurement(
            sta_mac = None,
            measurement_token=measurement_token,
            tx_power=tx_power,
            link_margin=link_margin,
            rx_antenna_id=rx_antenna_id,
            tx_antenna_id=tx_antenna_id,
            rcpi=rcpi,
            rsni=rsni,
            bssid=None,
            operating_class=operating_class,
            channel_number=channel_number,
            parent_tsf=parent_tsf
        )

    def to_dict(self) -> dict:
        """
        Dump the complete state of the LinkMeasurement.
        """
        return {
            "sta_mac": self.sta_mac,
            "measurement_token": self.measurement_token,
            "tx_power": self.tx_power,
            "link_margin": self.link_margin,
            "rx_antenna_id": self.rx_antenna_id,
            "tx_antenna_id": self.tx_antenna_id,
            "rcpi": self.rcpi,
            "rsni": self.rsni,
            "bssid": self.bssid,
            "operating_class": self.operating_class,
            "channel_number": self.channel_number,
            "parent_tsf": self.parent_tsf,
            "rssi_dbm": self.rssi_dbm,
        }


from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class BSSTransitionResponse:
    """
    Parsed representation of an 802.11v BSS Transition Management Response
    (Action Code = 8 under WNM category 10).
    """

    dialog_token: int                      # must match the Request...s token
    status_code: int                       # 0 = accept, >0 = reject (802.11v Table 9-348)
    bss_termination_delay: int = 0         # always present (1 byte)

    # Optional fields depending on AP implementation
    target_bssid: Optional[str] = None     # Only present if STA accepted transition
    candidate_list: List[Dict] = field(default_factory=list)
    neighbor_reports: List[Dict] = field(default_factory=list)

    # Vendor-specific tags (MBO, OCE, proprietary)
    vendor_ies: List[Dict[str, bytes]] = field(default_factory=list)

    # Raw unknown IEs for forward compatibility
    extra_ies: List[Dict[str, bytes]] = field(default_factory=list)

    # ----------------------------------------------------
    # Convenience properties
    # ----------------------------------------------------
    @property
    def accepted(self) -> bool:
        """True if the STA accepted the BSS transition suggestion."""
        return self.status_code == 0

    @property
    def rejected(self) -> bool:
        return not self.accepted

    @property
    def has_candidates(self) -> bool:
        """Whether the response contains any candidate BSS entries."""
        return len(self.candidate_list) > 0

    @property
    def has_vendor_extensions(self) -> bool:
        return len(self.vendor_ies) > 0

    @property
    def termination_imminent(self) -> bool:
        """Non-zero termination delay means AP plans to disassociate soon."""
        return self.bss_termination_delay > 0

    def to_dict(self) -> dict:
        """
        Dump the complete state of the BSSTransitionResponse.
        """
        # We assume that candidate_list, neighbor_reports, vendor_ies, and extra_ies are already serializable
        return {
            "dialog_token": self.dialog_token,
            "status_code": self.status_code,
            "bss_termination_delay": self.bss_termination_delay,
            "target_bssid": self.target_bssid,
            "candidate_list": list(self.candidate_list) if self.candidate_list else [],
            "neighbor_reports": list(self.neighbor_reports) if self.neighbor_reports else [],
            "vendor_ies": list(self.vendor_ies) if self.vendor_ies else [],
            "extra_ies": list(self.extra_ies) if self.extra_ies else [],
            "accepted": self.accepted,
            "rejected": self.rejected,
            "has_candidates": self.has_candidates,
            "has_vendor_extensions": self.has_vendor_extensions,
            "termination_imminent": self.termination_imminent,
        }

    def __repr__(self):
        return (
            f"<BSSTransitionResponse token={self.dialog_token} "
            f"status={self.status_code} "
            f"target={self.target_bssid} "
            f"candidates={len(self.candidate_list)} "
            f"vendorIEs={len(self.vendor_ies)}>"
        )
