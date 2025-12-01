import numpy as np
import pandas as pd
from datetime import timedelta

# =====================================================
#               HARD–CODED BASELINES
# =====================================================
BASELINES = {
    'cca_busy_pct':     {'mu0': 0.52838, 'sigma0': 0.16312},
    'wifi_airtime_pct': {'mu0': 0.52838, 'sigma0': 0.16312},
    'noise_floor_dbm':  {'mu0': -51.45684, 'sigma0': 0.34354},
    'snr_db':           {'mu0': 81.11196, 'sigma0': 0.34603},
}

# =====================================================
#               LOW–LATENCY ALGORITHMS
# =====================================================

class EWMA:
    def __init__(self, alpha=0.3, L=3):
        self.alpha = alpha
        self.L = L
        self.s = None

    def update(self, x, mu0, sigma0):
        if self.s is None:
            self.s = mu0
        self.s = self.alpha * x + (1 - self.alpha) * self.s

        threshold = mu0 + self.L * sigma0
        return self.s > threshold


class SPRT:
    def __init__(self, A=0.2, B=5.0):
        self.A = np.log(A)
        self.B = np.log(B)
        self.S = 0

    def update(self, x, mu0, sigma0):
        # Likelihood ratio approximated with Gaussian log pdf
        z0 = -((x - mu0)**2) / (2 * sigma0**2)
        z1 = -((x - (mu0 + sigma0*3))**2) / (2 * sigma0**2)
        lr = z1 - z0
        self.S += lr

        if self.S > self.B:
            self.S = 0
            return True
        if self.S < self.A:
            self.S = 0
        return False


class CUSUM:
    def __init__(self, k=0.5, h=3):
        self.k = k
        self.h = h
        self.pos = 0
        self.neg = 0

    def update(self, x, mu0):
        self.pos = max(0, self.pos + (x - mu0 - self.k))
        self.neg = max(0, self.neg + (mu0 - x - self.k))
        if self.pos > self.h or self.neg > self.h:
            self.pos = 0
            self.neg = 0
            return True
        return False

# =====================================================
#               PIPELINE CORE MANAGER
# =====================================================

class HybridDetector:
    def __init__(self):
        # Metric-specific detectors
        self.detectors = {
            "cca_busy_pct":     EWMA(alpha=0.25, L=3),
            "wifi_airtime_pct": EWMA(alpha=0.25, L=3),
            "noise_floor_dbm":  SPRT(),
            "snr_db":           CUSUM()
        }

        # For voting & stitching
        self.consecutive = {m: 0 for m in self.detectors}
        self.prev_ts = None
        self.cooldown_until = None
        self.active_event_start = None
        self.alerts = []

        self.VOTE_REQUIRED = 2
        self.CONSEC_REQUIRED = 2
        self.COOLDOWN_SEC = 30
        self.MERGE_GAP_SEC = 15

    def process_row(self, row):
        ts = pd.to_datetime(row["timestamp"])
        channel = row["channel"]

        # cooldown
        if self.cooldown_until and ts < self.cooldown_until:
            return None

        fired = []

        # detection per metric
        for metric, detector in self.detectors.items():
            x = row[metric]
            mu0 = BASELINES[metric]['mu0']
            sigma0 = BASELINES[metric]['sigma0']

            if isinstance(detector, EWMA):
                out = detector.update(x, mu0, sigma0)
            elif isinstance(detector, SPRT):
                out = detector.update(x, mu0, sigma0)
            else:  # CUSUM
                out = detector.update(x, mu0)

            if out:
                self.consecutive[metric] += 1
                if self.consecutive[metric] >= self.CONSEC_REQUIRED:
                    fired.append(metric)
            else:
                self.consecutive[metric] = 0

        # voting logic
        if len(fired) >= self.VOTE_REQUIRED:

            # event stitching
            if self.active_event_start is None:
                self.active_event_start = ts
                event_start = ts
            else:
                if (ts - self.prev_ts).total_seconds() > self.MERGE_GAP_SEC:
                    # close previous event
                    self.alerts.append({
                        "alert_timestamp": str(self.prev_ts),
                        "channel": channel,
                        "event_start": str(self.active_event_start),
                        "event_end": str(self.prev_ts),
                        "fired_metrics": fired
                    })
                    self.active_event_start = ts

            self.prev_ts = ts

            # cooldown after event closes
            self.cooldown_until = ts + timedelta(seconds=self.COOLDOWN_SEC)

        return None

    def finalize(self):
        # close last event
        if self.active_event_start and self.prev_ts:
            self.alerts.append({
                "alert_timestamp": str(self.prev_ts),
                "channel": None,
                "event_start": str(self.active_event_start),
                "event_end": str(self.prev_ts),
                "fired_metrics": []
            })
        return self.alerts

# =====================================================
#              READ CSV & RUN PIPELINE
# =====================================================

def run_pipeline(row):
    detector = HybridDetector()

    detector.process_row(row)

    alerts = detector.finalize()
    return alerts
