#!/usr/bin/env python3
"""
802.11k Radio Measurement Request Frame Builder
Constructs proper measurement request frames according to IEEE 802.11-2016
"""

import struct
import random

class RRMFrameBuilder:
    # Action frame categories
    ACTION_RADIO_MEASUREMENT = 5
    
    # Radio Measurement Action codes
    RADIO_MEASUREMENT_REQUEST = 0
    RADIO_MEASUREMENT_REPORT = 1
    
    # Measurement types
    MEASURE_TYPE_BASIC = 0
    MEASURE_TYPE_CCA = 1
    MEASURE_TYPE_RPI = 2
    MEASURE_TYPE_CHANNEL_LOAD = 3
    MEASURE_TYPE_NOISE_HISTOGRAM = 4
    MEASURE_TYPE_BEACON = 5
    MEASURE_TYPE_FRAME = 6
    MEASURE_TYPE_STA_STATISTICS = 7
    MEASURE_TYPE_LCI = 8
    MEASURE_TYPE_TRANSMIT_STREAM = 9
    
    # Measurement request modes
    MODE_PARALLEL = 0x01
    MODE_ENABLE = 0x02
    MODE_REQUEST = 0x04
    MODE_REPORT = 0x08
    MODE_DURATION_MANDATORY = 0x10
    
    def __init__(self):
        self.dialog_token = random.randint(1, 255)
        
    def _get_next_dialog_token(self):
        """Get next dialog token (wraps at 255)"""
        token = self.dialog_token
        self.dialog_token = (self.dialog_token + 1) % 256
        if self.dialog_token == 0:
            self.dialog_token = 1
        return token
        
    def build_beacon_request(self, operating_class=81, channel=6, 
                            duration=50, mode=0, bssid="ff:ff:ff:ff:ff:ff",
                            ssid=None):
        """
        Build 802.11k Beacon Request
        
        Args:
            operating_class: Operating class (81 = 2.4GHz channels 1-13)
            channel: Channel to scan (0 = all channels, 255 = current)
            duration: Measurement duration in TUs (1 TU = 1024 microseconds)
            mode: Measurement mode (0 = passive, 1 = active, 2 = beacon table)
            bssid: BSSID to scan (ff:ff:ff:ff:ff:ff = all)
            ssid: SSID to scan (None = all)
        """
        frame = bytearray()
        
        # Action frame header
        frame.append(self.ACTION_RADIO_MEASUREMENT)  # Category
        frame.append(self.RADIO_MEASUREMENT_REQUEST)  # Action
        frame.append(self._get_next_dialog_token())   # Dialog Token
        frame.extend(struct.pack('<H', 0))             # Number of repetitions
        
        # Measurement Request element
        frame.append(38)  # Element ID (Measurement Request)
        
        # Build measurement request body
        req_body = bytearray()
        req_body.append(random.randint(1, 255))  # Measurement Token
        req_body.append(self.MODE_ENABLE)         # Measurement Request Mode
        req_body.append(self.MEASURE_TYPE_BEACON) # Measurement Type
        
        # Beacon Request specific fields
        beacon_req = bytearray()
        beacon_req.append(operating_class)        # Operating Class
        beacon_req.append(channel)                # Channel Number
        beacon_req.extend(struct.pack('<H', random.randint(0, 65535)))  # Randomization Interval
        beacon_req.extend(struct.pack('<H', duration))  # Measurement Duration
        beacon_req.append(mode)                   # Measurement Mode
        beacon_req.extend(self._mac_to_bytes(bssid))  # BSSID
        
        # Optional subelements
        subelements = bytearray()
        
        if ssid:
            # SSID subelement
            ssid_bytes = ssid.encode('utf-8')
            subelements.append(0)  # SSID subelement ID
            subelements.append(len(ssid_bytes))
            subelements.extend(ssid_bytes)
            
        # Reporting Information subelement (request all fields)
        subelements.append(1)   # Reporting Information subelement ID
        subelements.append(2)   # Length
        subelements.extend(struct.pack('<H', 0x0007))  # Report all
        
        # Reporting Detail subelement (0 = no fixed fields/elements, 1 = requested only, 2 = all)
        subelements.append(2)   # Reporting Detail subelement ID
        subelements.append(1)   # Length
        subelements.append(2)   # All fixed fields and elements
        
        beacon_req.extend(subelements)
        req_body.extend(beacon_req)
        
        # Set length
        frame.append(len(req_body))
        frame.extend(req_body)
        
        return frame.hex()
        
    def build_channel_load_request(self, operating_class=81, channel=6, duration=50):
        """
        Build 802.11k Channel Load Request
        Measures channel utilization
        """
        frame = bytearray()
        
        # Action frame header
        frame.append(self.ACTION_RADIO_MEASUREMENT)
        frame.append(self.RADIO_MEASUREMENT_REQUEST)
        frame.append(self._get_next_dialog_token())
        frame.extend(struct.pack('<H', 0))
        
        # Measurement Request element
        frame.append(38)  # Element ID
        
        req_body = bytearray()
        req_body.append(random.randint(1, 255))  # Measurement Token
        req_body.append(self.MODE_ENABLE)
        req_body.append(self.MEASURE_TYPE_CHANNEL_LOAD)
        
        # Channel Load Request fields
        req_body.append(operating_class)
        req_body.append(channel)
        req_body.extend(struct.pack('<H', random.randint(0, 1000)))  # Random Interval
        req_body.extend(struct.pack('<H', duration))
        
        frame.append(len(req_body))
        frame.extend(req_body)
        
        return frame.hex()
        
    def build_noise_histogram_request(self, operating_class=81, channel=6, duration=50):
        """
        Build 802.11k Noise Histogram Request
        Measures noise floor distribution
        """
        frame = bytearray()
        
        frame.append(self.ACTION_RADIO_MEASUREMENT)
        frame.append(self.RADIO_MEASUREMENT_REQUEST)
        frame.append(self._get_next_dialog_token())
        frame.extend(struct.pack('<H', 0))
        
        frame.append(38)  # Element ID
        
        req_body = bytearray()
        req_body.append(random.randint(1, 255))
        req_body.append(self.MODE_ENABLE)
        req_body.append(self.MEASURE_TYPE_NOISE_HISTOGRAM)
        
        req_body.append(operating_class)
        req_body.append(channel)
        req_body.extend(struct.pack('<H', random.randint(0, 1000)))
        req_body.extend(struct.pack('<H', duration))
        
        frame.append(len(req_body))
        frame.extend(req_body)
        
        return frame.hex()
        
    def build_sta_statistics_request(self, peer_mac, duration=50, group_identity=0):
        """
        Build 802.11k Station Statistics Request
        
        Args:
            peer_mac: MAC address of peer station
            duration: Measurement duration
            group_identity: Statistics group (0-15, see 802.11 spec)
        """
        frame = bytearray()
        
        frame.append(self.ACTION_RADIO_MEASUREMENT)
        frame.append(self.RADIO_MEASUREMENT_REQUEST)
        frame.append(self._get_next_dialog_token())
        frame.extend(struct.pack('<H', 0))
        
        frame.append(38)  # Element ID
        
        req_body = bytearray()
        req_body.append(random.randint(1, 255))
        req_body.append(self.MODE_ENABLE)
        req_body.append(self.MEASURE_TYPE_STA_STATISTICS)
        
        req_body.extend(self._mac_to_bytes(peer_mac))
        req_body.extend(struct.pack('<H', random.randint(0, 1000)))
        req_body.extend(struct.pack('<H', duration))
        req_body.append(group_identity)
        
        frame.append(len(req_body))
        frame.extend(req_body)
        
        return frame.hex()
        
    def _mac_to_bytes(self, mac_str):
        """Convert MAC address string to bytes"""
        return bytes.fromhex(mac_str.replace(':', ''))
        
    def parse_measurement_report(self, report_hex):
        """
        Parse measurement report frame
        Returns dict with parsed data
        """
        data = bytes.fromhex(report_hex)
        
        if len(data) < 3:
            return None
            
        result = {
            'category': data[0],
            'action': data[1],
            'dialog_token': data[2],
        }
        
        # Parse measurement report elements
        if len(data) > 5:
            offset = 5
            while offset < len(data) - 2:
                elem_id = data[offset]
                elem_len = data[offset + 1]
                
                if elem_id == 39:  # Measurement Report element
                    result['measurement_token'] = data[offset + 2]
                    result['measurement_mode'] = data[offset + 3]
                    result['measurement_type'] = data[offset + 4]
                    result['report_data'] = data[offset + 5:offset + 2 + elem_len].hex()
                    
                offset += 2 + elem_len
                
        return result

# Example usage
if __name__ == "__main__":
    builder = RRMFrameBuilder()
    
    print("Beacon Request (scan channel 6):")
    frame = builder.build_beacon_request(channel=6, mode=0, ssid="MyNetwork")
    print(f"  {frame}\n")
    
    print("Channel Load Request:")
    frame = builder.build_channel_load_request(channel=6)
    print(f"  {frame}\n")
    
    print("Noise Histogram Request:")
    frame = builder.build_noise_histogram_request(channel=11)
    print(f"  {frame}\n")
