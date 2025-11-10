#!/usr/bin/env python3
"""
RRM Daemon - Radio Resource Management Daemon for hostapd
Sends 802.11k measurement requests and processes responses
"""

import time
import signal
import sys
import argparse
from hostapd_control import HostapdControl
from rrm_frames import RRMFrameBuilder

class RRMDaemon:
    def __init__(self, ctrl_path="/var/run/hostapd/wlan0", interval=60):
        self.ctrl = HostapdControl(ctrl_path)
        self.builder = RRMFrameBuilder()
        self.interval = interval
        self.running = False
        self.measurements = {}  # Store measurement results by station
        
    def start(self):
        """Start the RRM daemon"""
        print("Starting RRM Daemon...")
        
        try:
            self.ctrl.connect()
            self.running = True
            
            # Main loop
            while self.running:
                self.measurement_cycle()
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.ctrl.disconnect()
            
    def stop(self):
        """Stop the daemon"""
        self.running = False
        
    def measurement_cycle(self):
        """Perform one measurement cycle on all stations"""
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting measurement cycle")
        
        # Get connected stations
        stations = self.ctrl.get_stations()
        print(f"Found {len(stations)} connected stations")
        
        for sta in stations:
            mac = sta.get('mac')
            if not mac:
                continue
                
            # Initialize measurement data for this station
            if mac not in self.measurements:
                self.measurements[mac] = {
                    'last_seen': time.time(),
                    'capabilities': {},
                    'measurements': []
                }
                
            self.measurements[mac]['last_seen'] = time.time()
            
            # Check if station supports RRM
            rrm_caps = sta.get('rrm_caps')
            if not rrm_caps or rrm_caps == '00000000':
                print(f"  {mac}: RRM not supported, skipping")
                #continue
                
            print(f"  {mac}: Sending measurement requests...")
            
            # Send beacon request (passive scan on current channel)
            self.send_beacon_request(mac, channel=255, mode=0)
            time.sleep(0.1)
            
            # Send channel load request
            self.send_channel_load_request(mac)
            time.sleep(0.1)
            
            # Process any pending responses
            self.process_events()
            
    def send_beacon_request(self, sta_mac, channel=6, mode=0, ssid=None):
        """
        Send beacon request to station
        mode: 0=passive, 1=active, 2=beacon table
        channel: 255=current, 0=all, or specific channel
        """
        frame = self.builder.build_beacon_request(
            channel=channel,
            mode=mode,
            ssid=ssid
        )
        
        success = self.ctrl.send_mgmt_frame(sta_mac, frame)
        if success:
            print(f"    ✓ Beacon request sent (ch={channel}, mode={mode})")
        else:
            print(f"    ✗ Failed to send beacon request")
            
        return success
        
    def send_channel_load_request(self, sta_mac, channel=6):
        """Send channel load measurement request"""
        frame = self.builder.build_channel_load_request(channel=channel)
        
        success = self.ctrl.send_mgmt_frame(sta_mac, frame)
        if success:
            print(f"    ✓ Channel load request sent (ch={channel})")
        else:
            print(f"    ✗ Failed to send channel load request")
            
        return success
        
    def send_noise_histogram_request(self, sta_mac, channel=6):
        """Send noise histogram measurement request"""
        frame = self.builder.build_noise_histogram_request(channel=channel)
        
        success = self.ctrl.send_mgmt_frame(sta_mac, frame)
        if success:
            print(f"    ✓ Noise histogram request sent (ch={channel})")
        else:
            print(f"    ✗ Failed to send noise histogram request")
            
        return success
        
    def process_events(self):
        """Process events from hostapd"""
        events = self.ctrl.receive_events(timeout=2.0)
        
        for event in events:
            # Look for RRM-related events
            if "RRM-NEIGHBOR-REP-REQUEST-RECEIVED" in event:
                self.handle_neighbor_request(event)
            elif "RRM-BEACON-REP-RECEIVED" in event:
                self.handle_beacon_report(event)
            elif "RX-MGMT" in event and "action=5" in event.lower():
                # Radio Measurement action frame
                self.handle_measurement_report(event)
            else:
                # Print other events for debugging
                if "AP-STA-CONNECTED" in event or "AP-STA-DISCONNECTED" in event:
                    print(f"  Event: {event}")
                    
    def handle_neighbor_request(self, event):
        """Handle neighbor report request from station"""
        print(f"  Neighbor report requested: {event}")
        
    def handle_beacon_report(self, event):
        """Handle beacon report from station"""
        print(f"  Beacon report received: {event}")
        # Parse and store beacon report data
        
    def handle_measurement_report(self, event):
        """Handle measurement report frame"""
        print(f"  Measurement report: {event}")
        # Parse measurement report
        
    def get_station_stats(self, sta_mac):
        """Get stored measurement data for a station"""
        return self.measurements.get(sta_mac, {})
        
    def print_statistics(self):
        """Print accumulated statistics"""
        print("\n" + "="*60)
        print("RRM STATISTICS")
        print("="*60)
        
        for mac, data in self.measurements.items():
            print(f"\nStation: {mac}")
            print(f"  Last seen: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data['last_seen']))}")
            print(f"  Measurements collected: {len(data['measurements'])}")
            
def main():
    parser = argparse.ArgumentParser(description='RRM Daemon for hostapd')
    parser.add_argument('--interface', '-i', default='wlan0',
                       help='hostapd interface (default: wlan0)')
    parser.add_argument('--interval', '-t', type=int, default=60,
                       help='Measurement interval in seconds (default: 60)')
    parser.add_argument('--ctrl-path', '-c',
                       help='hostapd control interface path (default: /var/run/hostapd/INTERFACE)')
    parser.add_argument('--once', action='store_true',
                       help='Run one measurement cycle and exit')
    
    args = parser.parse_args()
    
    # Construct control path
    ctrl_path = args.ctrl_path or f"/var/run/hostapd/{args.interface}"
    
    # Create daemon
    daemon = RRMDaemon(ctrl_path=ctrl_path, interval=args.interval)
    
    # Handle signals
    def signal_handler(signum, frame):
        print("\nReceived signal, shutting down...")
        daemon.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run daemon
    if args.once:
        try:
            daemon.ctrl.connect()
            daemon.measurement_cycle()
            daemon.process_events()
            daemon.print_statistics()
        finally:
            daemon.ctrl.disconnect()
    else:
        daemon.start()

if __name__ == "__main__":
    main()
