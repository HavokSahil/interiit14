from typing import Dict, Any, Optional
from db.station_db import StationDB
from db.lmrep_db import LinkMeasurementDB
import json
import time
from model.station import Station
from model.measurement import LinkMeasurement        
from metrics.qoe import QoE
from http.server import BaseHTTPRequestHandler, HTTPServer

class StateAPI:
    def __init__(self):
        self.stdb = StationDB()
        self.lmdb = LinkMeasurementDB()
        self.qoe = QoE()
    
    def build_station_api_dict(self, station: Station, lm: Optional[LinkMeasurement] = None) -> Dict:
        """
        Generate API-friendly dict with key station metrics and QoE.
        """
        if not station or not station.mac:
            return {"error": "Invalid station object"}

        # Compute QoE components
        components = self.qoe.compute_qoe(station, lm)

        # Latest history info (optional)
        history_obj = self.qoe.get_history(station.mac)
        trend = history_obj.trend if history_obj is not None else "insufficient_data"
        volatility = history_obj.volatility if history_obj else 0.0
        avg_qoe = history_obj.average if history_obj else components.overall

        tx_retry_count = station.tx_retry_count if station.tx_retry_count else 0.01
        tx_packets_count = station.tx_packets if station.tx_packets else 0.01
        fcs_error_count = station.fcs_error_count if station.fcs_error_count else 0.01
        rx_packets_count = station.rx_packets if station.rx_packets else 0.01

        api_dict = {
            "mac": station.mac,
            "connected": "AUTHORIZED" in station.flags and "ASSOC" in station.flags,
            "signal": {
                "avg_signal": station.avg_signal,
                "rssi_dbm": lm.rssi_dbm if lm else None,
                "link_margin": lm.link_margin if lm else None,
                "score": round(components.signal_quality, 3)
            },
            "throughput": {
                "tx_bitrate": station.tx_bitrate,
                "rx_bitrate": station.rx_bitrate,
                "expected_throughput": station.expected_throughput,
                "score": round(components.throughput, 3)
            },
            "reliability": {
                "tx_retry_rate": tx_retry_count / tx_packets_count,
                "fcs_error_rate": fcs_error_count / rx_packets_count,
                "score": round(components.reliability, 3)
            },
            "latency": {
                "backlog_packets": station.backlog_packets,
                "inactive_msec": station.inactive_msec,
                "score": round(components.latency, 3)
            },
            "activity": {
                "tx_air_time": station.tx_airtime or 0,
                "rx_air_time": station.rx_airtime or 0,
                "total_tx_rx_packets": (station.tx_packets or 0) + (station.rx_packets or 0),
                "score": round(components.activity, 3)
            },
            "qoe": {
                "overall": round(components.overall, 3),
                "trend": trend,
                "volatility": round(volatility, 3),
                "average_history": round(avg_qoe, 3)
            },
            "timestamp": components.timestamp.isoformat()
        }
        return api_dict

    def req(self) -> Dict[str, Any]:
        
        response = {
            "timestamp": time.time(),
            "status": "ok",
            "component": "StateAPI",
            "version": "1.0",
            "length": 0,
            "data": []
        }

        self.qoe.update()

        for station in self.stdb.all():
            mac = station.mac
            lm = self.lmdb.get(mac)
            dataitem = self.build_station_api_dict(station, lm)
            if dataitem:
                response["data"].append(dataitem)
                response["length"] += 1
        return response

    def json(self) -> str:
        return json.dumps(self.req())

    def serve(self, host: str = "0.0.0.0", port: int = 8080):
        """
        Serve the StateAPI on a simple HTTP server, responding with the JSON API data.
        """
        api_instance = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/" or self.path.startswith("/api"):
                    response_json = api_instance.json()
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(response_json.encode("utf-8"))
                else:
                    self.send_response(404)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))

            def log_message(self, format, *args):
                # Suppress logging by default
                return

        server = HTTPServer((host, port), Handler)
        print(f"StateAPI HTTP server running at http://{host}:{port}/api")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Shutting down StateAPI HTTP server...")
        finally:
            server.server_close()