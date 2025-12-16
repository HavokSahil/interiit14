from collections import deque
from statistics import median
import numpy as np
import yaml
import policy_engine_evaluator_1_3
import pprint

WINDOW_SIZE = 10

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

out_headers = [
"client_count",
"rssi_p50",
"retry_rate_p95",
"retry_rate_p95",
"cca_busy_pct",
"throughput_mbps_p50",
"throughput_mbps_p10",
"rssi_p50",
"obss_pd_threshold",
"tx_power",
"noise_floor",
"bandwidth",
"airtime_fraction_p50",
"cca_busy_pct",
"roam_out_rate",
]

class LogDigest:
    def __init__(self, log_dir: str):
        with open("policy_slo_compliance/slo_catalog.yml") as f:
            self.slo_catalog = yaml.safe_load(f)
        self.role = "BE"
        self.signing_key = "AllYourBaseAreBelongToUs"
        self.items = deque(maxlen=WINDOW_SIZE)
        self.current: dict[str, dict] = {}
        self.client_params = "rssi_dbm", "sinr_db", "retry_rate", "airtime_fraction", "throughput_mbps"
        self.log_dir = log_dir

    def add_ap(self, ap: dict):
        ap_params = "tx_power", "channel", "bandwidth", "cca_busy_percentage", "obss_pd_threshold", "noise_floor", "roam_out_rate"
        self.current[ap["ap_id"]] = { i: ap[i] for i in ap_params }
        for param in self.client_params:
            self.current[ap["ap_id"]][param] = []

    def add_client(self, client: dict):
        associated_ap = self.current[client["associated_ap"]]
        for param in self.client_params:
            associated_ap[param].append(client[param])

    def end_log(self):
        client_count = -1
        for entry in self.current.values():
            for param in self.client_params:
                client_count = len(entry[param])
                if len(entry[param]) == 0:
                    entry[f"{param}_p10"] = 0
                    entry[f"{param}_p50"] = 0
                    entry[f"{param}_p95"] = 0
                else:
                    entry[f"{param}_p10"] = np.percentile([float(i) for i in entry[param]], 10)
                    entry[f"{param}_p50"] = median([float(i) for i in entry[param]])
                    entry[f"{param}_p95"] = np.percentile([float(i) for i in entry[param]], 95)
                del entry[param]
            RSSI_P95 = entry["rssi_dbm_p95"]
            RSSI_min = -95
            RSSI_max = -30
            entry["S"] = clamp((RSSI_P95 - RSSI_min) / (RSSI_max - RSSI_min), 0, 1)

            PER_P95 = entry["retry_rate_p95"]
            PER_max_pct = 20
            entry["T"] = clamp(1 - PER_P95 / PER_max_pct, 0, 1)

            Retry_max_pct = 30
            Retry_P95 = entry["retry_rate_p95"]
            entry["R"] = clamp(1 - Retry_P95 / Retry_max_pct, 0, 1)

            Airtime_P50 = entry["airtime_fraction_p50"]
            entry["L"] = clamp(1 - Airtime_P50 / 100, 0, 1)

            CCA_busy_P50 = float(entry["cca_busy_percentage"])
            entry["A"] = clamp(1 - CCA_busy_P50 / 100, 0, 1)
        out = {}
        for ap_id, ap_dict in self.current.items():
            ap_dict["ap_id"] = ap_id
            out[ap_id] = dict(zip(out_headers, [client_count,ap_dict["rssi_dbm_p50"],ap_dict["retry_rate_p95"],ap_dict["retry_rate_p95"],ap_dict["cca_busy_percentage"],ap_dict["throughput_mbps_p50"],ap_dict["throughput_mbps_p10"],ap_dict["rssi_dbm_p50"],ap_dict["obss_pd_threshold"],ap_dict["tx_power"],ap_dict["noise_floor"],ap_dict["bandwidth"],ap_dict["airtime_fraction_p50"],ap_dict["cca_busy_percentage"],ap_dict["roam_out_rate"]]))
            #pprint.pp(
                #policy_engine_evaluator_1_3.run_policy_engine(ap_dict, self.role, self.slo_catalog, self.signing_key)
            #)
        self.items.append(self.current)
        self.current = {}
        return out
