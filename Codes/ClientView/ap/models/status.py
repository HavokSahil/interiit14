import re
from enum import Enum

class ApStatusIndicator(Enum):
    AP_ENABLED  = 1
    AP_DISABLED = 2


class ApStatus:
    def __init__(self):
        self.status_dict = dict()
        self.state = ApStatusIndicator.AP_DISABLED
        self.phy = None
        self.freq = None
        self.hw_mode = None
        self.country_code = None
        self.channel = None
        self.beacon_int = None
        self.supported_rates = None
        self.max_txpower = None
        self.bss = None
        self.bssid = None
        self.ssid = None
        self.num_sta = None
        self.raw = None


    @staticmethod
    def from_content(content: str):
        status = ApStatus()
        status.parse_status(content)
        return status

    @staticmethod
    def decode_supported_rates(rate_hex_str) -> list:
        """Convert the supported rate hex string into Mbps list"""
        rate_map: dict[str, int] = { # map of hex to Mbps
            "0x02": 1,
            "0x04": 2,
            "0x0b": 5.5,
            "0x16": 11,
            "0x0c": 6,
            "0x12": 9,
            "0x18": 12,
            "0x24": 18,
            "0x30": 24,
            "0x48": 36,
            "0x60": 48,
            "0x6c": 54,
        }
        rates = list[int]()
        for rate in rate_hex_str.strip().split():
            rates.append(rate_map[rate]) # convert hex to Mbps
        return rates


    def parse_status(self, content: str):
        """Parse status content from hostapd in key=value format."""
        self.raw = content
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or '=' not in line:
                continue
            
            # Split key and value
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle array notation like bss[0], bssid[0], etc.
                array_match = re.match(r'^(\w+)\[(\d+)\]$', key)
                if array_match:
                    base_key = array_match.group(1)
                    index = int(array_match.group(2))
                    
                    if base_key not in self.status_dict:
                        self.status_dict[base_key] = {}
                    self.status_dict[base_key][index] = self._convert_value(value)
                else:
                    # Special handling for supported_rates
                    if key == 'supported_rates':
                        # Convert hex string to list of Mbps rates
                        # Input format: "02 04 0b 16..." -> convert to "0x02 0x04 0x0b 0x16..."
                        rate_hex_str = ' '.join([f'0x{rate.lower()}' for rate in value.split()])
                        self.status_dict[key] = self.decode_supported_rates(rate_hex_str)
                    else:
                        # Regular key-value pair
                        self.status_dict[key] = self._convert_value(value)
        
        # Set the state based on the 'state' field - check if ENABLED is in the value
        if 'state' in self.status_dict:
            state_value = self.status_dict['state']
            if isinstance(state_value, str) and 'ENABLED' in state_value.upper():
                self.state = ApStatusIndicator.AP_ENABLED
            else:
                self.state = ApStatusIndicator.AP_DISABLED
        
        # Populate instance attributes from status_dict
        self.phy = self.status_dict.get('phy')
        self.freq = self.status_dict.get('freq')
        self.hw_mode = self.status_dict.get('hw_mode')
        self.country_code = self.status_dict.get('country_code')
        self.channel = self.status_dict.get('channel')
        self.beacon_int = self.status_dict.get('beacon_int')
        self.supported_rates = self.status_dict.get('supported_rates')
        self.max_txpower = self.status_dict.get('max_txpower')
        self.bss = self.status_dict.get('bss')
        self.bssid = self.status_dict.get('bssid')
        self.ssid = self.status_dict.get('ssid')
        self.num_sta = self.status_dict.get('num_sta')
    
    def _convert_value(self, value: str):
        """Convert string value to appropriate type (int, hex, None, or string)."""
        if value == 'N/A':
            return None
        
        # Try to convert to integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try to convert hex values (0x...)
        if value.startswith('0x') or value.startswith('0X'):
            try:
                return int(value, 16)
            except ValueError:
                pass
        
        # Return as string
        return value

    def __dict__(self):
        return self.status_dict

    def __str__(self):
        # Provide a readable string representation of the AP status
        attrs = [
            f"state={self.state.name}",
            f"phy={self.phy}",
            f"freq={self.freq}",
            f"hw_mode={self.hw_mode}",
            f"country_code={self.country_code}",
            f"channel={self.channel}",
            f"beacon_int={self.beacon_int}",
            f"supported_rates={self.supported_rates}",
            f"max_txpower={self.max_txpower}",
            f"bss={self.bss}",
            f"bssid={self.bssid}",
            f"ssid={self.ssid}",
            f"num_sta={self.num_sta}"
        ]
        return "<ApStatus " + " ".join(attrs) + ">"