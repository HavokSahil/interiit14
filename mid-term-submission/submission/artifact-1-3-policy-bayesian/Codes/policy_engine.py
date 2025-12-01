#!/usr/bin/env python3
"""
Complete system with:
- Per-client HNSS, PER, throughput
- NLI, median airtime
- Interval-based NLI median (configurable)
- Time-of-Day recommendation (min NLI in history)
- PER 95th percentile + top 5% PER clients
- QoE scraping + QoE drop threshold
- Unified change_request system (binary)
- Prometheus metrics
- Persistent JSON history

"""

import json
import os
import time
import math
import requests
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any

from prometheus_client import start_http_server, Gauge


# -----------------------
# CONFIG VALUES
# -----------------------
API_URL = "http://192.170.12.96:8080/api"
POLL_INTERVAL = 5
HISTORY_FILE = "nli_history.json"

MAX_CLIENTS = 300
MAX_AIRTIME_UTIL = 100
EPS = 0.01

MEDIAN_INTERVAL_MINUTES = 30    # <--- Set to 1 while testing
TOD_REQUIRED_HOURS = 24         # hours needed for Time-of-Day activation

HNSS_MODERATE_THRESH = 0.6
HNSS_SEVERE_THRESH = 0.75
PER_THRESHOLD = 0.08            # 8%
QOE_DROP_THRESHOLD = 12.0       # 12% drop in QoE triggers change

PROM_PORT = 8000


# -----------------------
# PROMETHEUS METRICS
# -----------------------

g_hnss = Gauge("hnss", "Per-client HNSS", ["mac"])
g_hnss_alert = Gauge("hnss_alert", "HNSS alert", ["mac", "level"])

g_per_client = Gauge("packet_error_rate", "Per-client PER", ["mac"])
g_per_high = Gauge("packet_error_top5", "Worst 5% PER clients", ["mac"])
g_per_p95 = Gauge("packet_error_rate_p95", "95th percentile PER")

g_network_nli = Gauge("network_nli_latest", "Latest NLI")
g_network_nli_interval = Gauge("network_nli_interval_median", "Interval median NLI")
g_network_airtime_median = Gauge("network_airtime_median", "Median airtime")
g_median_throughput = Gauge("median_client_throughput", "Median throughput")

g_tod_available = Gauge("config_tod_available", "TOD available flag")
g_tod_ts = Gauge("config_tod_timestamp", "TOD unix timestamp")

g_change_request = Gauge("network_change_request", "Binary change request flag")


# -----------------------
# JSON HISTORY HELPERS
# -----------------------

def load_history(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f).get("median_30min", [])
    except:
        return []

def save_history(path: str, samples):
    with open(path, "w") as f:
        json.dump({"median_30min": samples}, f, indent=2)

def prune_history(samples, now):
    cutoff = now - timedelta(hours=TOD_REQUIRED_HOURS)
    return [s for s in samples if datetime.fromisoformat(s["ts"]) >= cutoff]


# -----------------------
# PERCENTILE HELPER
# -----------------------

def percentile(values: List[float], p: float):
    if not values:
        return None
    if p <= 0: return min(values)
    if p >= 100: return max(values)

    vals = sorted(values)
    k = (len(vals) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c: return vals[int(k)]
    return vals[f] * (c - k) + vals[c] * (k - f)


# -----------------------
# POLICY ENGINE
# -----------------------

class PolicyEngine:
    def __init__(self):
        now = datetime.utcnow()

        self.history = prune_history(load_history(HISTORY_FILE), now)
        save_history(HISTORY_FILE, self.history)

        self.samples = []
        self.last_median_ts = None

    def calc_hnss(self, ul, dl):
        return (ul - dl) / (ul + dl + EPS)

    def calc_nli(self, clients, airtime):
        return (clients / MAX_CLIENTS) + (airtime / MAX_AIRTIME_UTIL)

    def add_sample(self, nli):
        self.samples.append(nli)

    def maybe_interval_median(self, now):
        if self.last_median_ts is None:
            self.last_median_ts = now
            return None
        
        if now - self.last_median_ts < timedelta(minutes=MEDIAN_INTERVAL_MINUTES):
            return None

        if not self.samples:
            self.last_median_ts = now
            return None

        med = statistics.median(self.samples)

        entry = {"ts": now.isoformat(), "nli": float(med)}
        self.history.append(entry)

        self.history = prune_history(self.history, now)
        save_history(HISTORY_FILE, self.history)

        self.samples = []
        self.last_median_ts = now
        return med

    def get_tod(self):
        if not self.history:
            return None

        required_samples = int((TOD_REQUIRED_HOURS * 60) / MEDIAN_INTERVAL_MINUTES)
        if len(self.history) < required_samples:
            return None

        return min(self.history, key=lambda x: x["nli"])["ts"]


# -----------------------
# API FETCH
# -----------------------

def fetch_api():
    try:
        r = requests.get(API_URL, timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("Fetch error:", e)
        return None


# -----------------------
# MAIN LOOP
# -----------------------

def poll_loop(engine: PolicyEngine):
    while True:
        now = datetime.utcnow()
        pkt = fetch_api()

        if pkt:
            try:
                clients = pkt.get("data", [])
                if not clients:
                    time.sleep(POLL_INTERVAL)
                    continue

                client_count = len(clients)

                airtimes = []
                pers = []
                per_pairs = []
                throughputs = []

                severe_hnss_flag = False

                for c in clients:
                    mac = c.get("mac", "unknown")

                    reliab = c.get("reliability", {})
                    ul = float(reliab.get("tx_retry_rate", 0))
                    dl = float(reliab.get("dl_retry_rate", 0))

                    per_val = float(reliab.get("per", reliab.get("fcs_error_rate", 0)) or 0)

                    airtime = float(c.get("airtime", {}).get("utilization", 0))

                    tx = float(c.get("throughput", {}).get("tx_bitrate", 0))
                    rx = float(c.get("throughput", {}).get("rx_bitrate", 0))
                    thr = (tx + rx) / 2

                    airtimes.append(airtime)
                    pers.append(per_val)
                    per_pairs.append((mac, per_val))
                    throughputs.append(thr)

                    g_per_client.labels(mac=mac).set(per_val)

                    # HNSS
                    hnss = engine.calc_hnss(ul, dl)
                    g_hnss.labels(mac=mac).set(hnss)

                    # Alerts
                    g_hnss_alert.labels(mac=mac, level="moderate").set(1 if hnss >= HNSS_MODERATE_THRESH else 0)
                    g_hnss_alert.labels(mac=mac, level="severe").set(1 if hnss >= HNSS_SEVERE_THRESH else 0)

                    if hnss >= HNSS_SEVERE_THRESH:
                        severe_hnss_flag = True
                        print(f"[{now}] SEVERE HNSS: {mac} HNSS={hnss:.3f}")

                # ----------------------
                # Network metrics
                # ----------------------
                med_air = statistics.median(airtimes)
                g_network_airtime_median.set(med_air)

                nli = engine.calc_nli(client_count, med_air)
                g_network_nli.set(nli)

                engine.add_sample(nli)
                interval_median = engine.maybe_interval_median(now)
                if interval_median is not None:
                    g_network_nli_interval.set(interval_median)
                    print(f"[{now}] Interval NLI median = {interval_median:.3f}")

                # ----------------------
                # PER 95th percentile
                # ----------------------
                per_p95 = percentile(pers, 95)
                if per_p95 is not None:
                    g_per_p95.set(per_p95)

                # Reset all high-PER flags
                for mac, _ in per_pairs:
                    g_per_high.labels(mac=mac).set(0)

                worst_clients = []
                if per_p95 is not None:
                    for mac, val in per_pairs:
                        if val >= per_p95:
                            worst_clients.append(mac)
                            g_per_high.labels(mac=mac).set(1)

                    if worst_clients:
                        print(f"[{now}] PER 95th={per_p95:.2f} | Worst5%: {worst_clients}")

                # ----------------------
                # Median throughput
                # ----------------------
                median_thr = statistics.median(throughputs)
                g_median_throughput.set(median_thr)

                # ----------------------
                # QoE METRIC & CHANGE REQUEST
                # ----------------------
                qoe_data = clients[0].get("qoe", {})
                curr_qoe = float(qoe_data.get("overall", 1.0))
                avg_qoe = float(qoe_data.get("average_history", curr_qoe))

                qoe_drop = 0.0
                if avg_qoe > 0:
                    qoe_drop = ((avg_qoe - curr_qoe) / avg_qoe) * 100

                qoe_trigger = qoe_drop >= QOE_DROP_THRESHOLD
                per_trigger = per_p95 is not None and per_p95 >= PER_THRESHOLD

                change_request = 1 if (qoe_trigger or per_trigger or severe_hnss_flag) else 0
                g_change_request.set(change_request)

                if change_request:
                    print(f"[{now}] CHANGE REQUEST | QoE_drop={qoe_drop:.2f}% PER={per_p95:.2f} HNSS_severe={severe_hnss_flag}")

                # ----------------------
                # Time-of-Day recommendation
                # ----------------------
                tod = engine.get_tod()
                if tod:
                    g_tod_available.set(1)
                    tod_dt = datetime.fromisoformat(tod)
                    g_tod_ts.set(int(tod_dt.timestamp()))
                else:
                    g_tod_available.set(0)

                # Summary print
                print("========== SUMMARY ==========")
                print("Time:", now)
                print("Clients:", client_count)
                print("Median Airtime:", med_air)
                print("NLI:", nli)
                print("Median Throughput:", median_thr)
                print("QoE Drop:", f"{qoe_drop:.2f}%")
                print("Change Request:", change_request)
                print("=============================\n")

            except Exception as e:
                print("Processing error:", e)

        time.sleep(POLL_INTERVAL)


# -----------------------
# MAIN
# -----------------------

def main():
    engine = PolicyEngine()
    start_http_server(PROM_PORT)
    print(f"Prometheus metrics on :{PROM_PORT}/metrics")
    print(f"Median interval={MEDIAN_INTERVAL_MINUTES} min")
    poll_loop(engine)


if __name__ == "__main__":
    main()
