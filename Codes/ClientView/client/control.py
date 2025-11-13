from control.logger import *
from ap.control import ApController
from client.models.datatype import LinkMeasurementRequestFrameParser, LinkMeasurementResponseFrameParser, BeaconResponseFrameParser, MacAddress
from client.parser import extract_buf_from_event, extract_link_measurement_response_data, extract_beacon_response_data, extract_beacon_req_tx_status
from client.builder import *
from typing import Dict, Optional, List
from scapy.layers.dot11 import Dot11

class StationController:
    def __init__(self, ap: ApController, mac: MacAddress) -> None:
        self.ap = ap
        self.mac = mac

        self.link_measurement_request: LinkMeasurementRequestFrameParser | None = None
        self.link_measurement_report: LinkMeasurementResponseFrameParser | None = None

        # Store beacon measurements: key is dialog_token, value is BeaconMeasurementEntry
        self.beacon_measurements: List[BeaconResponseFrameParser] = list()

    def req_beacon(self, builder: ReqBeaconBuilder) -> bool:
        """
        Request a beacon measurement from the station.
        
        Returns:
            True if the beacon measurement was successfully requested and parsed, False otherwise.
        """
        self.ap.clear_events()
        builder.dest_mac = self.mac
        req = builder.build()
        if not self.ap.send_command(f"{req}"):
            return False
        if not self.check_beacon_req_acknowledge(timeout=1.0):
            Logger.log_err("check_beacon_req_acknowledge: failed")
            return False
        while True:
            response = self._parse_beacon_response_frame()
            if not response:
                break
        
        return True

    def request_link_measurement(self) -> bool:
        """
        Request a link measurement from the station.
        
        Returns:
            True if the link measurement was successfully requested and parsed, False otherwise.
        """
        self.ap.clear_events()
        return self._parse_link_measurement_response()
    
    def _send_link_measurement_request(self) -> bool:
        """Send the link measurement request command and verify it was accepted."""
        if not self.ap.send_command(f"REQ_LINK_MEASUREMENT {self.mac.raw}"):
            return False
        
        reply = self.ap.receive()
        return reply is not None and reply != "FAIL"
    
    def _parse_link_measurement_response(self) -> bool:
        """
        Parse the LINK-MSR-RESP-RX event.
        
        Returns:
            True if the response was successfully received and parsed, False otherwise.
        """
        event = self.ap.receive_event()
        if not event or "LINK-MSR-RESP-RX" not in event:
            return False
        
        resp_data = extract_link_measurement_response_data(event)
        if not resp_data:
            return False
        
        resp_mac, resp_number, resp_hex = resp_data
        print(f"Parsed LINK-MSR-RESP-RX: mac={resp_mac}, number={resp_number}, hex={resp_hex}")
        
        parser = LinkMeasurementResponseFrameParser(resp_hex)
        if not parser.parse():
            error = parser.get_error()
            print(f"Parser failed: {error}")
            return False
        
        self.link_measurement_report = parser.__dict__()
        return True
    
    def _parse_beacon_response_frame(self) -> bool:
        """
        Parse the BEACON-RESP-RX event and associate it with the corresponding request.
        
        Returns:
            BeaconResponseFrameParser if successful, None otherwise.
        """
        event = self.ap.receive_event()
        if not event:
            return False
            
        resp_hex = extract_buf_from_event(event)
        print(f"Parsed buff={resp_hex}")
        
        pkt = Dot11(bytes.fromhex(resp_hex))
        pkt.show()
        
        # Try to associate response with request using the number field
        # The number field in BEACON-RESP-RX might correspond to measurement_token or dialog_token
        self.beacon_measurements.append(parser)
        
        return True
    
    def get_beacon_measurement(self, dialog_token: int) -> Optional[BeaconResponseFrameParser]:
        """
        Get a beacon measurement entry by dialog token.
        
        Args:
            dialog_token: The dialog token of the beacon request
            
        Returns:
            BeaconMeasurementEntry if found, None otherwise.
        """
        return self.beacon_measurements.get(dialog_token)
    
    def get_all_beacon_measurements(self) -> Dict[int, BeaconResponseFrameParser]:
        """
        Get all stored beacon measurements.
        
        Returns:
            Dictionary mapping dialog tokens to BeaconMeasurementEntry objects.
        """
        return self.beacon_measurements.copy()
    
    def check_beacon_req_acknowledge(self, timeout: float = 3.0) -> bool:
        """
        Check for BEACON-REQ-TX-STATUS event to verify if a beacon request was acknowledged.
        
        Args:
            timeout: Maximum time to wait for the acknowledgment event (in seconds)
        
        Returns:
            Tuple of (token, acknowledged) if acknowledgment received, None otherwise.
            - token: The measurement token or dialog token from the event
            - acknowledged: True if ack=1, False if ack=0
        """
        
        event = self.ap.receive_event()
        if not event or "BEACON-REQ-TX-STATUS" not in event:
            return False

        status_data = extract_beacon_req_tx_status(event)
        if not status_data:
            return None
        
        mac, number, acknowledged = status_data
        print(f"Parsed BEACON-REQ-TX-STATUS: mac={mac}, number={number}, ack={acknowledged}")
        
        return acknowledged