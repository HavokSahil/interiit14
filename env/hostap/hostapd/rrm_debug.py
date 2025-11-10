#!/usr/bin/env python3
"""
RRM Debug Tool - Check client RRM capabilities in detail
"""

import sys
from hostapd_control import HostapdControl

def parse_rrm_caps(caps_hex):
    """
    Parse RRM capability bits
    RRM Enabled Capabilities is 5 bytes (40 bits)
    """
    if not caps_hex or caps_hex == '00000000' or caps_hex == '0000000000':
        return None
        
    try:
        # Remove any 0x prefix and convert to bytes
        caps_hex = caps_hex.replace('0x', '')
        caps_bytes = bytes.fromhex(caps_hex)
        
        capabilities = {
            'Link Measurement': bool(caps_bytes[0] & 0x01),
            'Neighbor Report': bool(caps_bytes[0] & 0x02),
            'Parallel Measurements': bool(caps_bytes[0] & 0x04),
            'Repeated Measurements': bool(caps_bytes[0] & 0x08),
            'Beacon Passive': bool(caps_bytes[0] & 0x10),
            'Beacon Active': bool(caps_bytes[0] & 0x20),
            'Beacon Table': bool(caps_bytes[0] & 0x40),
            'Beacon Measurement Reporting': bool(caps_bytes[0] & 0x80),
        }
        
        if len(caps_bytes) > 1:
            capabilities.update({
                'Frame Measurement': bool(caps_bytes[1] & 0x01),
                'Channel Load': bool(caps_bytes[1] & 0x02),
                'Noise Histogram': bool(caps_bytes[1] & 0x04),
                'Statistics Measurement': bool(caps_bytes[1] & 0x08),
                'LCI Measurement': bool(caps_bytes[1] & 0x10),
                'LCI Azimuth': bool(caps_bytes[1] & 0x20),
                'Transmit Stream/Category': bool(caps_bytes[1] & 0x40),
                'Triggered Transmit Stream/Category': bool(caps_bytes[1] & 0x80),
            })
            
        if len(caps_bytes) > 2:
            capabilities.update({
                'AP Channel Report': bool(caps_bytes[2] & 0x01),
                'RM MIB': bool(caps_bytes[2] & 0x02),
                'Operating Chan Max Duration': (caps_bytes[2] >> 2) & 0x07,
                'Nonoperating Chan Max Duration': (caps_bytes[2] >> 5) & 0x07,
            })
            
        return capabilities
    except:
        return None

def check_station_capabilities(ctrl_path="/var/run/hostapd/wlan0"):
    """Check detailed capabilities of all connected stations"""
    ctrl = HostapdControl(ctrl_path)
    
    try:
        ctrl.connect()
        stations = ctrl.get_stations()
        
        print("="*70)
        print("RRM CAPABILITY REPORT")
        print("="*70)
        print(f"\nFound {len(stations)} connected station(s)\n")
        
        for i, sta in enumerate(stations, 1):
            mac = sta.get('mac', 'Unknown')
            print(f"\n{'='*70}")
            print(f"Station {i}: {mac}")
            print(f"{'='*70}")
            
            # Basic info
            signal = sta.get('signal', 'N/A')
            rx_rate = sta.get('rx_rate', 'N/A')
            tx_rate = sta.get('tx_rate', 'N/A')
            
            print(f"\nBasic Info:")
            print(f"  Signal Strength: {signal} dBm")
            print(f"  RX Rate: {rx_rate}")
            print(f"  TX Rate: {tx_rate}")
            
            # Check for 802.11k/v/r support
            print(f"\nFast Roaming Support:")
            
            rrm_caps = sta.get('rrm_caps', '00000000')
            ext_capab = sta.get('ext_capab', '')
            
            print(f"  802.11k (RRM): ", end='')
            capabilities = parse_rrm_caps(rrm_caps)
            if capabilities:
                print("✓ SUPPORTED")
                print(f"\n  RRM Capabilities (raw): {rrm_caps}")
                print(f"\n  Detailed Capabilities:")
                for cap, supported in capabilities.items():
                    status = "✓" if supported else "✗"
                    print(f"    {status} {cap}")
            else:
                print("✗ NOT SUPPORTED")
                print(f"    Raw value: {rrm_caps}")
                
            # Check 802.11v (BSS Transition)
            print(f"\n  802.11v (BSS-TM): ", end='')
            if ext_capab and len(ext_capab) >= 6:
                # Bit 19 of extended capabilities indicates BSS Transition support
                try:
                    ext_bytes = bytes.fromhex(ext_capab)
                    if len(ext_bytes) > 2:
                        bss_transition = bool(ext_bytes[2] & 0x08)
                        print("✓ SUPPORTED" if bss_transition else "✗ NOT SUPPORTED")
                    else:
                        print("✗ NOT SUPPORTED")
                except:
                    print("? UNKNOWN")
            else:
                print("✗ NOT SUPPORTED")
                
            # Check 802.11r (Fast BSS Transition)
            print(f"  802.11r (FT): ", end='')
            akm = sta.get('akm', '')
            wpa = sta.get('wpa', '')
            if 'ft' in akm.lower() or 'ft' in wpa.lower():
                print("✓ SUPPORTED")
            else:
                print("✗ NOT SUPPORTED")
                
            # Additional useful info
            print(f"\nConnection Details:")
            connected_time = sta.get('connected_time', 'N/A')
            print(f"  Connected: {connected_time} seconds")
            
            # Print all raw station info for debugging
            print(f"\nRaw Station Info:")
            for key, value in sta.items():
                if key not in ['mac']:
                    print(f"  {key}: {value}")
                    
        print(f"\n{'='*70}\n")
        
    finally:
        ctrl.disconnect()

def suggest_fixes():
    """Suggest how to enable RRM on clients"""
    print("\n" + "="*70)
    print("HOW TO ENABLE RRM ON YOUR DEVICES")
    print("="*70)
    print("""
Android:
  - RRM support varies by manufacturer and Android version
  - Generally supported on Android 9+ with proper WiFi drivers
  - No user-facing settings to enable
  - Check: Settings → Developer Options → WiFi scan throttling (disable)

iOS/macOS:
  - RRM supported on iOS 10+ and macOS 10.12+
  - Automatically enabled, no configuration needed
  - Works best with Apple's enterprise WiFi settings

Linux (wpa_supplicant):
  - Add to wpa_supplicant.conf:
    network={
        ssid="YourNetwork"
        psk="password"
        rrm_neighbor_report=1
        rrm_beacon_report=1
    }

Windows 10/11:
  - Supported on most modern WiFi adapters
  - Enabled automatically if driver supports it
  - Check device manager for driver updates

Testing Device:
  - Try connecting from a different device to verify hostapd RRM works
  - Modern Android phones (Samsung, Google Pixel) usually support RRM
  - Linux laptop with recent Intel WiFi card (iwlwifi driver)
    
Your hostapd.conf should have:
  rrm_neighbor_report=1
  rrm_beacon_report=1
  bss_transition=1  # For 802.11v
    """)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Check RRM capabilities of connected stations')
    parser.add_argument('--interface', '-i', default='wlan0',
                       help='hostapd interface (default: wlan0)')
    parser.add_argument('--ctrl-path', '-c',
                       help='hostapd control interface path')
    
    args = parser.parse_args()
    
    ctrl_path = args.ctrl_path or f"/var/run/hostapd/{args.interface}"
    
    try:
        check_station_capabilities(ctrl_path)
        suggest_fixes()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
