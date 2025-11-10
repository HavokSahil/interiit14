#!/usr/bin/env python3
"""
hostapd Control Interface Handler
Communicates with hostapd via Unix socket
"""

import socket
import struct
import os
import select

class HostapdControl:
    def __init__(self, ctrl_path="/var/run/hostapd/wlan0"):
        self.ctrl_path = ctrl_path
        self.sock = None
        self.local_path = f"/tmp/hostapd_ctrl_{os.getpid()}"
        
    def connect(self):
        """Connect to hostapd control interface"""
        # Create Unix domain socket
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        
        # Bind to local path
        if os.path.exists(self.local_path):
            os.unlink(self.local_path)
        self.sock.bind(self.local_path)
        
        # Connect to hostapd
        self.sock.connect(self.ctrl_path)
        
        # Attach to receive unsolicited events
        self.send_command("ATTACH")
        print(f"Connected to hostapd at {self.ctrl_path}")
        
    def disconnect(self):
        """Disconnect from hostapd"""
        if self.sock:
            try:
                self.send_command("DETACH")
            except:
                pass
            self.sock.close()
            if os.path.exists(self.local_path):
                os.unlink(self.local_path)
                
    def send_command(self, cmd):
        """Send command to hostapd and get response"""
        self.sock.send(cmd.encode())
        
        # Wait for response with timeout
        ready = select.select([self.sock], [], [], 5.0)
        if ready[0]:
            response = self.sock.recv(4096).decode('utf-8', errors='ignore')
            return response.strip()
        return None
        
    def get_stations(self) -> list:
        """Get list of connected stations"""
        response = self.send_command("STA-FIRST")
        stations = []
        
        while response and not response.startswith("FAIL"):
            # Parse station MAC address (first line)
            lines = response.split('\n')
            if lines:
                mac = lines[0].strip()
                if mac and ':' in mac:
                    stations.append(mac)
                    
                    # Get detailed station info
                    sta_info = self.send_command(f"STA {mac}")
                    stations[-1] = self.parse_station_info(mac, sta_info)
            
            # Get next station
            response = self.send_command("STA-NEXT " + mac)

        return stations
        
    def parse_station_info(self, mac, info):
        """Parse station information"""
        sta_data = {'mac': mac}
        if not info:
            return sta_data
            
        for line in info.split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                sta_data[key.strip()] = value.strip()
                
        return sta_data
        
    def send_mgmt_frame(self, dst_mac, frame_body):
        """
        Send management frame to station
        dst_mac: Destination MAC address (XX:XX:XX:XX:XX:XX)
        frame_body: Frame body as hex string
        """
        cmd = f"MGMT_TX {dst_mac} {frame_body}"
        response = self.send_command(cmd)
        return response == "OK"
        
    def receive_events(self, timeout=1.0):
        """Receive unsolicited events from hostapd"""
        ready = select.select([self.sock], [], [], timeout)
        events = []
        
        while ready[0]:
            try:
                data = self.sock.recv(4096).decode('utf-8', errors='ignore')
                if data:
                    print(f"Raw EVENT: {data}")
                if data and data.startswith('<'):
                    events.append(data)
            except:
                break
            ready = select.select([self.sock], [], [], 0.1)
            
        return events

# Example usage
if __name__ == "__main__":
    ctrl = HostapdControl("/var/run/hostapd/wlan0")
    
    try:
        ctrl.connect()
        
        # Get connected stations
        stations = ctrl.get_stations()
        print(f"\nConnected stations: {len(stations)}")
        for sta in stations:
            print(f"  {sta.get('mac')} - Signal: {sta.get('signal', 'N/A')} dBm")
            
        # Listen for events for 5 seconds
        print("\nListening for events (5 seconds)...")
        import time
        start = time.time()
        while time.time() - start < 5:
            events = ctrl.receive_events(timeout=1.0)
            for event in events:
                print(f"Event: {event}")
                
    finally:
        ctrl.disconnect()
