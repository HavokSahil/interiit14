import re
from typing import Optional, Tuple
from ap.control import ApController
from client.models.datatype import LinkMeasurementRequestFrameParser, LinkMeasurementResponseFrameParser, MacAddress

class StationController:
    def __init__(self, ap: ApController, mac: MacAddress) -> None:
        self.ap = ap
        self.mac = mac

    def request_link_measurement(self) -> bool:
        """
        Request a link measurement from the station.
        
        Returns:
            True if the link measurement was successfully requested and parsed, False otherwise.
        """
        self.ap.clear_events()
        
        if not self._send_link_measurement_request():
            return False
        
        if not self._parse_mgmt_frame_received():
            return False
        
        return self._parse_link_measurement_response()
    
    def _send_link_measurement_request(self) -> bool:
        """Send the link measurement request command and verify it was accepted."""
        if not self.ap.send_command(f"REQ_LINK_MEASUREMENT {self.mac.raw}"):
            return False
        
        reply = self.ap.receive()
        return reply is not None and reply != "FAIL"
    
    def _parse_mgmt_frame_received(self) -> bool:
        """
        Parse the AP-MGMT-FRAME-RECEIVED event.
        
        Returns:
            True if the event was successfully received and parsed, False otherwise.
        """
        event = self.ap.receive_event()
        if not event or "AP-MGMT-FRAME-RECEIVED" not in event:
            return False
        
        buf_hex = self._extract_buf_from_event(event)
        if not buf_hex:
            return False
        
        print(f"Parsed AP-MGMT-FRAME-RECEIVED: buf={buf_hex}")
        
        parser = LinkMeasurementRequestFrameParser(buf_hex)
        if not parser.parse():
            error = parser.get_error()
            print(f"Parser failed: {error}")
            return False
        
        print(parser)
        return True
    
    def _parse_link_measurement_response(self) -> bool:
        """
        Parse the LINK-MSR-RESP-RX event.
        
        Returns:
            True if the response was successfully received and parsed, False otherwise.
        """
        event = self.ap.receive_event()
        if not event or "LINK-MSR-RESP-RX" not in event:
            return False
        
        resp_data = self._extract_response_data(event)
        if not resp_data:
            return False
        
        resp_mac, resp_number, resp_hex = resp_data
        print(f"Parsed LINK-MSR-RESP-RX: mac={resp_mac}, number={resp_number}, hex={resp_hex}")
        
        parser = LinkMeasurementResponseFrameParser(resp_hex)
        if not parser.parse():
            error = parser.get_error()
            print(f"Parser failed: {error}")
            return False
        
        print(parser)
        return True
    
    def _extract_buf_from_event(self, event: str) -> Optional[str]:
        """
        Extract the buf hex string from AP-MGMT-FRAME-RECEIVED event.
        
        Args:
            event: The event string, e.g., "<3>AP-MGMT-FRAME-RECEIVED buf=..."
        
        Returns:
            The hex string if found, None otherwise.
        """
        match = re.search(r'buf=([0-9a-fA-F]+)', event)
        return match.group(1) if match else None
    
    def _extract_response_data(self, event: str) -> Optional[Tuple[str, str, str]]:
        """
        Extract data from LINK-MSR-RESP-RX event.
        
        Args:
            event: The event string, e.g., "<3>LINK-MSR-RESP-RX <mac> <number> <hex>"
        
        Returns:
            Tuple of (mac_address, number, hex_string) if successful, None otherwise.
        """
        parts = event.split()
        if len(parts) < 4:
            return None
        
        # parts[0] = "<3>LINK-MSR-RESP-RX"
        # parts[1] = MAC address
        # parts[2] = number
        # parts[3] = hex string
        return (parts[1], parts[2], parts[3])
