import re
from model.ap import AP, APStatus

class APParser:
    @staticmethod
    def from_content(content: str):
        status = APStatus()
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

    def parse_status(ap: AP, content: str):
        """Parse status content from hostapd in key=value format."""
        ap.raw = content
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
                    
                    if base_key not in ap.status_dict:
                        ap.status_dict[base_key] = {}
                    ap.status_dict[base_key][index] = ap._convert_value(value)
                else:
                    # Special handling for supported_rates
                    if key == 'supported_rates':
                        # Convert hex string to list of Mbps rates
                        # Input format: "02 04 0b 16..." -> convert to "0x02 0x04 0x0b 0x16..."
                        rate_hex_str = ' '.join([f'0x{rate.lower()}' for rate in value.split()])
                        ap.status_dict[key] = ap.decode_supported_rates(rate_hex_str)
                    else:
                        # Regular key-value pair
                        ap.status_dict[key] = ap._convert_value(value)
        
        # Set the state based on the 'state' field - check if ENABLED is in the value
        if 'state' in ap.status_dict:
            state_value = ap.status_dict['state']
            if isinstance(state_value, str) and 'ENABLED' in state_value.upper():
                ap.state = APStatus.AP_ENABLED
            else:
                ap.state = APStatus.AP_DISABLED
        
        # Populate instance attributes from status_dict
        ap.phy = ap.status_dict.get('phy')
        ap.freq = ap.status_dict.get('freq')
        ap.hw_mode = ap.status_dict.get('hw_mode')
        ap.country_code = ap.status_dict.get('country_code')
        ap.channel = ap.status_dict.get('channel')
        ap.beacon_int = ap.status_dict.get('beacon_int')
        ap.supported_rates = ap.status_dict.get('supported_rates')
        ap.max_txpower = ap.status_dict.get('max_txpower')
        ap.bss = ap.status_dict.get('bss')
        ap.bssid = ap.status_dict.get('bssid')
        ap.ssid = ap.status_dict.get('ssid')
        ap.num_sta = ap.status_dict.get('num_sta')
    
    @staticmethod
    def _convert_value(ap, value: str):
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