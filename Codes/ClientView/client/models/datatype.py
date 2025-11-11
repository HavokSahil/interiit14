import re
import hashlib
import struct
from typing import Optional

class MacAddress:
    """Object for MAC address handling, parsing, and representation."""
    def __init__(self, mac: str):
        # Accepts colon, dash, or dot separators, or none, keep in hex
        pattern = re.compile(
            r"^(?P<mac>([0-9A-Fa-f]{2}[:\-\.]?){5,7}[0-9A-Fa-f]{2})$"
        )
        match = pattern.match(mac)
        if not match:
            raise ValueError(f"Invalid MAC address format: {mac}")
        self.raw = self._normalize(mac)
        self.octets = self.raw.split(":")

    @staticmethod
    def _normalize(mac: str) -> str:
        # Remove any separator, force lower, insert ":"
        normalized = re.sub(r'[^0-9A-Fa-f]', '', mac).lower()
        if len(normalized) != 12:
            raise ValueError(f"MAC address must be 12 hex digits: {mac}")
        return ':'.join(normalized[i:i+2] for i in range(0, 12, 2))

    def __str__(self):
        return self.raw

    def __repr__(self):
        return f"MacAddress('{self.raw}')"

    def is_multicast(self) -> bool:
        # Check least significant bit of first octet
        first_octet = int(self.octets[0], 16)
        return bool(first_octet & 0x01)

    def is_unicast(self) -> bool:
        return not self.is_multicast()

    def is_broadcast(self) -> bool:
        return all(o == 'ff' for o in self.octets)

    def is_local_administered(self) -> bool:
        # Second least significant bit of first octet is set
        first_octet = int(self.octets[0], 16)
        return bool(first_octet & 0x02)

    def oui(self) -> str:
        """Return the OUI (Organizationally Unique Identifier) as a string."""
        return ':'.join(self.octets[:3])

    def client_part(self) -> str:
        """Return the client-specific part (last three octets) as a string."""
        return ':'.join(self.octets[3:])

    def anonymized(self, hash_algorithm: str = "sha256", length: int = 12) -> str:
        """
        Return the MAC address as OUI:HASHED_CLIENT_PART.
        hash_algorithm: Hash function to use (e.g. 'sha256', 'md5').
        length: Number of hex characters from digest to use.
        """
        client = self.client_part()
        hasher = hashlib.new(hash_algorithm)
        hasher.update(client.encode('utf-8'))
        digest = hasher.hexdigest()[:length]
        # Reformat into MAC form: XX:XX:XX:xx:xx:xx
        hashed_client = ':'.join(digest[i:i+2] for i in range(0, min(6, length), 2))
        # If length < 6, fill with zeroes
        while hashed_client.count(':') < 2:
            hashed_client += ':00'
        return f"{self.oui()}:{hashed_client}"

    @staticmethod
    def is_valid(mac: str) -> bool:
        """
        Check if the given string is a valid MAC address format.
        Acceptable formats:
          - XX:XX:XX:XX:XX:XX
          - XX-XX-XX-XX-XX-XX
          - XXXXXXXXXXXX
        Returns True if valid, False otherwise.
        """
        if not isinstance(mac, str):
            return False
        mac = mac.strip()
        # 6 groups of 2 hex digits separated by : or -
        import re
        if re.fullmatch(r"([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})", mac):
            return True
        # 12 hex digits (no separators)
        if re.fullmatch(r"[0-9A-Fa-f]{12}", mac):
            return True
        return False


import struct
from typing import Optional, List, Tuple

class LinkMeasurementRequestFrameParser:
    """
    Parser for 802.11k Link Measurement Response frames.
    Parses the binary frame data from hex strings according to IEEE 802.11-2020.
    """
    
    def __init__(self, frame_hex: str):
        """
        Initialize parser with hex string of the frame.
        
        Args:
            frame_hex: Hex string of the frame (with or without spaces)
        """
        # Remove spaces and convert to bytes
        frame_hex = frame_hex.replace(' ', '').replace('\n', '').lower()
        try:
            self.frame_data = bytes.fromhex(frame_hex)
        except ValueError as e:
            raise ValueError(f"Invalid hex string: {e}")
        
        # Parsed fields - 802.11 MAC Header
        self.frame_control: Optional[int] = None
        self.duration: Optional[int] = None
        self.da: Optional[str] = None  # Destination Address
        self.sa: Optional[str] = None  # Source Address
        self.bssid: Optional[str] = None
        self.sequence_control: Optional[int] = None
        
        # Action frame fields
        self.category: Optional[int] = None
        self.action: Optional[int] = None
        self.dialog_token: Optional[int] = None
        
        # Link Measurement Report fields
        self.tpc_report_tx_power: Optional[int] = None  # Signed dBm
        self.tpc_report_link_margin: Optional[int] = None  # Signed dB
        self.rx_antenna_id: Optional[int] = None
        self.tx_antenna_id: Optional[int] = None
        self.rcpi: Optional[int] = None  # Received Channel Power Indicator
        self.rsni: Optional[int] = None  # Received Signal to Noise Indicator
        
        # Optional subelements
        self.subelements: List[Tuple[int, bytes]] = []
        
        self.raw_hex = frame_hex
        self.error_message: Optional[str] = None
        self.failure_stage: Optional[str] = None
    
    @staticmethod
    def _bytes_to_mac(addr_bytes: bytes) -> str:
        """Convert 6 bytes to MAC address string format."""
        return ':'.join(f'{b:02x}' for b in addr_bytes)
    
    @staticmethod
    def _signed_byte(value: int) -> int:
        """Convert unsigned byte to signed byte (-128 to 127)."""
        return value if value < 128 else value - 256
    
    def parse(self) -> bool:
        """
        Parse the frame data according to 802.11k specification.
        
        Link Measurement Report frame format:
        - MAC Header (24 bytes)
        - Category (1 byte): 5 (Radio Measurement)
        - Action (1 byte): 3 (Link Measurement Report)
        - Dialog Token (1 byte)
        - TPC Report Element (4 bytes): Element ID (35), Length (2), TX Power, Link Margin
        - Receive Antenna ID (1 byte)
        - Transmit Antenna ID (1 byte)
        - RCPI (1 byte)
        - RSNI (1 byte)
        - Optional Subelements
        
        Returns:
            True if parsing successful, False otherwise. Use get_error() to get failure details.
        """
        # Reset error state
        self.error_message = None
        self.failure_stage = None
        
        # Check minimum frame length
        if len(self.frame_data) < 24:
            self.failure_stage = "Frame Length Check"
            self.error_message = f"Frame too short: got {len(self.frame_data)} bytes, need at least 24 bytes for MAC header"
            return False
        
        # Parse 802.11 MAC header (24 bytes)
        try:
            header = struct.unpack('<HH6s6s6sH', self.frame_data[:24])
            self.frame_control = header[0]
            self.duration = header[1]
            self.da = self._bytes_to_mac(header[2])
            self.sa = self._bytes_to_mac(header[3])
            self.bssid = self._bytes_to_mac(header[4])
            self.sequence_control = header[5]
        except struct.error as e:
            self.failure_stage = "MAC Header Parsing"
            self.error_message = f"Failed to unpack MAC header: {e}"
            return False
        
        # Verify this is a management action frame
        frame_type = (self.frame_control >> 2) & 0x03
        if frame_type != 0:  # Must be management frame
            self.failure_stage = "Frame Type Verification"
            self.error_message = f"Not a management frame: frame_type={frame_type} (expected 0)"
            return False
        
        # Parse action frame body (starts at byte 24)
        if len(self.frame_data) < 27:
            self.failure_stage = "Action Frame Body Length"
            self.error_message = f"Frame too short for action frame: got {len(self.frame_data)} bytes, need at least 27 bytes (24 header + 3 action bytes)"
            return False
        
        body = self.frame_data[24:]
        self.category = body[0]
        self.action = body[1]
        self.dialog_token = body[2]
        
        # Verify this is a Link Measurement Report
        # Category 5 = Radio Measurement, Action 3 = Link Measurement Report
        if self.category != 5:
            self.failure_stage = "Category Verification"
            self.error_message = f"Invalid category: got {self.category} (expected 5 for Radio Measurement)"
            return False
        
        if self.action != 3:
            self.failure_stage = "Action Verification"
            self.error_message = f"Invalid action: got {self.action} (expected 3 for Link Measurement Report)"
            return False
        
        # Parse TPC Report Element (4 bytes starting at offset 3)
        if len(body) < 7:  # Need at least category + action + token + 4 bytes TPC
            self.failure_stage = "TPC Report Element Length"
            self.error_message = f"Frame too short for TPC Report: got {len(body)} bytes in body, need at least 7 bytes (3 action + 4 TPC)"
            return False
        
        tpc_element_id = body[3]
        tpc_length = body[4]
        
        # TPC Report Element ID should be 35, length should be 2
        if tpc_element_id != 35:
            self.failure_stage = "TPC Element ID Verification"
            self.error_message = f"Invalid TPC Element ID: got {tpc_element_id} (expected 35)"
            return False
        
        if tpc_length != 2:
            self.failure_stage = "TPC Element Length Verification"
            self.error_message = f"Invalid TPC Element length: got {tpc_length} (expected 2)"
            return False
        
        # TX Power and Link Margin are signed values
        self.tpc_report_tx_power = self._signed_byte(body[5])
        self.tpc_report_link_margin = self._signed_byte(body[6])
        
        # Parse remaining fields (starting at offset 7)
        if len(body) < 11:  # Need 4 more bytes for antenna IDs, RCPI, RSNI
            self.failure_stage = "Measurement Fields Length"
            self.error_message = f"Frame too short for measurement fields: got {len(body)} bytes in body, need at least 11 bytes (7 previous + 4 measurement fields)"
            return False
        
        self.rx_antenna_id = body[7]
        self.tx_antenna_id = body[8]
        self.rcpi = body[9]
        self.rsni = body[10]
        
        # Parse optional subelements (starting at offset 11)
        offset = 11
        while offset < len(body):
            if offset + 1 >= len(body):
                # Incomplete subelement header, but this is optional so we just stop
                break
            
            elem_id = body[offset]
            elem_len = body[offset + 1]
            
            if offset + 2 + elem_len > len(body):
                self.failure_stage = "Subelement Parsing"
                self.error_message = f"Subelement at offset {offset} extends beyond frame: element_id={elem_id}, length={elem_len}, available={len(body) - offset - 2} bytes"
                return False
            
            elem_data = body[offset + 2:offset + 2 + elem_len]
            self.subelements.append((elem_id, elem_data))
            offset += 2 + elem_len
        
        # Parsing successful
        return True
    
    def get_error(self) -> Optional[str]:
        """
        Get the error message from the last parse attempt.
        
        Returns:
            Error message string if parsing failed, None if successful or not yet parsed.
        """
        if self.error_message:
            return f"[{self.failure_stage}] {self.error_message}"
        return None
    
    def get_frame_type(self) -> str:
        """Get frame type from frame control field."""
        if self.frame_control is None:
            return "Unknown"
        frame_type = (self.frame_control >> 2) & 0x03
        frame_subtype = (self.frame_control >> 4) & 0x0F
        
        type_names = {
            0: "Management",
            1: "Control",
            2: "Data"
        }
        
        type_name = type_names.get(frame_type, "Unknown")
        return f"{type_name} (subtype: {frame_subtype})"
    
    def get_rcpi_dbm(self) -> Optional[float]:
        """Convert RCPI to dBm. RCPI = (Power in dBm + 110) * 2"""
        if self.rcpi is None or self.rcpi == 255:
            return None
        return (self.rcpi / 2.0) - 110.0
    
    def get_rsni_db(self) -> Optional[float]:
        """Convert RSNI to dB. RSNI = (SNR in dB + 10) * 2"""
        if self.rsni is None or self.rsni == 255:
            return None
        return (self.rsni / 2.0) - 10.0
    
    def __str__(self) -> str:
        """String representation of parsed frame."""
        parts = []
        parts.append("=== 802.11 MAC Header ===")
        parts.append(f"Frame Control: 0x{self.frame_control:04x} ({self.get_frame_type()})" if self.frame_control else "Frame Control: None")
        parts.append(f"Duration: {self.duration} μs" if self.duration else "Duration: None")
        parts.append(f"DA: {self.da}" if self.da else "DA: None")
        parts.append(f"SA: {self.sa}" if self.sa else "SA: None")
        parts.append(f"BSSID: {self.bssid}" if self.bssid else "BSSID: None")
        parts.append(f"Sequence Control: 0x{self.sequence_control:04x}" if self.sequence_control else "Sequence Control: None")
        
        parts.append("\n=== Action Frame ===")
        parts.append(f"Category: {self.category} (Radio Measurement)" if self.category == 5 else f"Category: {self.category}")
        parts.append(f"Action: {self.action} (Link Measurement Report)" if self.action == 3 else f"Action: {self.action}")
        parts.append(f"Dialog Token: 0x{self.dialog_token:02x}" if self.dialog_token is not None else "Dialog Token: None")
        
        parts.append("\n=== Link Measurement Report ===")
        parts.append(f"TPC TX Power: {self.tpc_report_tx_power} dBm" if self.tpc_report_tx_power is not None else "TPC TX Power: None")
        parts.append(f"TPC Link Margin: {self.tpc_report_link_margin} dB" if self.tpc_report_link_margin is not None else "TPC Link Margin: None")
        parts.append(f"RX Antenna ID: {self.rx_antenna_id}" if self.rx_antenna_id is not None else "RX Antenna ID: None")
        parts.append(f"TX Antenna ID: {self.tx_antenna_id}" if self.tx_antenna_id is not None else "TX Antenna ID: None")
        
        if self.rcpi is not None:
            rcpi_dbm = self.get_rcpi_dbm()
            parts.append(f"RCPI: {self.rcpi} ({rcpi_dbm:.1f} dBm)" if rcpi_dbm is not None else f"RCPI: {self.rcpi} (invalid)")
        else:
            parts.append("RCPI: None")
        
        if self.rsni is not None:
            rsni_db = self.get_rsni_db()
            parts.append(f"RSNI: {self.rsni} ({rsni_db:.1f} dB)" if rsni_db is not None else f"RSNI: {self.rsni} (invalid)")
        else:
            parts.append("RSNI: None")
        
        if self.subelements:
            parts.append("\n=== Optional Subelements ===")
            for elem_id, elem_data in self.subelements:
                parts.append(f"Element ID {elem_id}: {elem_data.hex()}")
        
        return "\n".join(parts)
    
    def __repr__(self) -> str:
        return f"LinkMeasurementRequestFrameParser(hex='{self.raw_hex[:32]}...')"



class LinkMeasurementResponseFrameParser:
    """
    Parser for 802.11k Link Measurement Response payloads.
    This decodes the response body returned by 'LINK-MSR-RESP-RX',
    e.g. hex like: 230208000000a242
    
    According to IEEE 802.11-2020, the payload format is:
    - TPC Report Element ID (1 byte): 35
    - Length (1 byte): 2
    - TX Power (1 byte): Signed dBm
    - Link Margin (1 byte): Signed dB
    - Receive Antenna ID (1 byte)
    - Transmit Antenna ID (1 byte)
    - RCPI (1 byte): Received Channel Power Indicator
    - RSNI (1 byte): Received Signal to Noise Indicator
    """
    
    def __init__(self, hex_string: str):
        """
        Initialize parser with hex string of the response payload.
        
        Args:
            hex_string: Hex string of the response payload (with or without spaces)
        """
        # Remove spaces and convert to bytes
        hex_string = hex_string.replace(' ', '').replace('\n', '').lower()
        try:
            self.data = bytes.fromhex(hex_string)
        except ValueError as e:
            raise ValueError(f"Invalid hex string: {e}")
        
        # Parsed fields
        self.tpc_element_id: Optional[int] = None
        self.tpc_length: Optional[int] = None
        self.tpc_report_tx_power: Optional[int] = None  # Signed dBm
        self.tpc_report_link_margin: Optional[int] = None  # Signed dB
        self.rx_antenna_id: Optional[int] = None
        self.tx_antenna_id: Optional[int] = None
        self.rcpi: Optional[int] = None  # Received Channel Power Indicator
        self.rsni: Optional[int] = None  # Received Signal to Noise Indicator
        
        self.raw_hex = hex_string
        self.error_message: Optional[str] = None
        self.failure_stage: Optional[str] = None
    
    @staticmethod
    def _signed_byte(value: int) -> int:
        """Convert unsigned byte to signed byte (-128 to 127)."""
        return value if value < 128 else value - 256
    
    def parse(self) -> bool:
        """
        Parse the response payload data according to 802.11k specification.
        
        Returns:
            True if parsing successful, False otherwise. Use get_error() to get failure details.
        """
        # Reset error state
        self.error_message = None
        self.failure_stage = None
        
        # Check minimum payload length (8 bytes)
        if len(self.data) < 8:
            self.failure_stage = "Payload Length Check"
            self.error_message = f"Payload too short: got {len(self.data)} bytes, need at least 8 bytes"
            return False
        
        # Parse TPC Report Element (first 4 bytes)
        try:
            self.tpc_element_id, self.tpc_length, tx_power_raw, link_margin_raw = \
                struct.unpack_from("BBBB", self.data, 0)
        except struct.error as e:
            self.failure_stage = "TPC Report Element Parsing"
            self.error_message = f"Failed to unpack TPC Report Element: {e}"
            return False
        
        # Verify TPC Element ID should be 35
        if self.tpc_element_id != 35:
            self.failure_stage = "TPC Element ID Verification"
            self.error_message = f"Invalid TPC Element ID: got {self.tpc_element_id} (expected 35)"
            return False
        
        # Verify TPC Element length should be 2
        if self.tpc_length != 2:
            self.failure_stage = "TPC Element Length Verification"
            self.error_message = f"Invalid TPC Element length: got {self.tpc_length} (expected 2)"
            return False
        
        # TX Power and Link Margin are signed values
        self.tpc_report_tx_power = self._signed_byte(tx_power_raw)
        self.tpc_report_link_margin = self._signed_byte(link_margin_raw)
        
        # Parse remaining fields (bytes 4-7)
        self.rx_antenna_id = self.data[4]
        self.tx_antenna_id = self.data[5]
        self.rcpi = self.data[6]
        self.rsni = self.data[7]
        
        # Parsing successful
        return True
    
    def get_error(self) -> Optional[str]:
        """
        Get the error message from the last parse attempt.
        
        Returns:
            Error message string if parsing failed, None if successful or not yet parsed.
        """
        if self.error_message:
            return f"[{self.failure_stage}] {self.error_message}"
        return None
    
    def get_rcpi_dbm(self) -> Optional[float]:
        """Convert RCPI to dBm. RCPI = (Power in dBm + 110) * 2"""
        if self.rcpi is None or self.rcpi == 255:
            return None
        return (self.rcpi / 2.0) - 110.0
    
    def get_rsni_db(self) -> Optional[float]:
        """Convert RSNI to dB. RSNI = (SNR in dB + 10) * 2"""
        if self.rsni is None or self.rsni == 255:
            return None
        return (self.rsni / 2.0) - 10.0
    
    def __str__(self) -> str:
        """String representation of parsed payload."""
        parts = []
        parts.append("=== Link Measurement Response Payload ===")
        parts.append(f"TPC Element ID: {self.tpc_element_id}" if self.tpc_element_id is not None else "TPC Element ID: None")
        parts.append(f"TPC Length: {self.tpc_length}" if self.tpc_length is not None else "TPC Length: None")
        parts.append(f"TPC TX Power: {self.tpc_report_tx_power} dBm" if self.tpc_report_tx_power is not None else "TPC TX Power: None")
        parts.append(f"TPC Link Margin: {self.tpc_report_link_margin} dB" if self.tpc_report_link_margin is not None else "TPC Link Margin: None")
        parts.append(f"RX Antenna ID: {self.rx_antenna_id}" if self.rx_antenna_id is not None else "RX Antenna ID: None")
        parts.append(f"TX Antenna ID: {self.tx_antenna_id}" if self.tx_antenna_id is not None else "TX Antenna ID: None")
        
        if self.rcpi is not None:
            rcpi_dbm = self.get_rcpi_dbm()
            parts.append(f"RCPI: {self.rcpi} ({rcpi_dbm:.1f} dBm)" if rcpi_dbm is not None else f"RCPI: {self.rcpi} (invalid)")
        else:
            parts.append("RCPI: None")
        
        if self.rsni is not None:
            rsni_db = self.get_rsni_db()
            parts.append(f"RSNI: {self.rsni} ({rsni_db:.1f} dB)" if rsni_db is not None else f"RSNI: {self.rsni} (invalid)")
        else:
            parts.append("RSNI: None")
        
        return "\n".join(parts)
    
    def __repr__(self) -> str:
        return f"LinkMeasurementResponseFrameParser(hex='{self.raw_hex[:16]}...')"


