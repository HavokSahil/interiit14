# feature_extraction_streaming.py
"""
Streaming feature extraction for CUSUM monitoring.

- Processes incoming IQ in small subblocks (default 10 ms).
- Maintains rolling windows at three timescales:
    FAST  (e.g. 100 ms)  -> energy / cca / short noise estimates
    MED   (e.g. 1 s)      -> snr aggregates, airtime smoothing
    SLOW  (e.g. 10 s)     -> packet-based metrics (PER, retry_rate) and long-term stats
- Provides hooks to feed decoded packet events (timestamped) so PER / retries and wifi_airtime
  can be computed correctly in the SLOW window.
- Outputs features:
    ['cca_busy_pct', 'wifi_airtime_pct', 'noise_floor_dbm',
     'nonwifi_duty_pct', 'retry_rate_pct', 'per_pct', 'snr_db']
"""
import numpy as np
import numpy.fft as fft
from collections import deque
import time
from rf_scanner_controller import RFScannerController
#import csv
import infer_real_capture_airshark
import change_detection
from ml_online import *
import concurrent.futures
import copy


# ------------------------
# Configuration parameters
# ------------------------
SAMPLE_RATE = 17.5e6            # Hz (set to your radio's sampling rate)
SUBBLOCK_MS = 10 # ms per processing subblock (must be small)
SUBBLOCK_SAMPLES = int(SAMPLE_RATE * (SUBBLOCK_MS / 1000.0))

# Window sizes (ms)
FAST_WIN_MS = 100.0           # e.g., 100 ms (energy/cca)
MED_WIN_MS = 1000.0           # e.g., 1 s (aggregated SNR / airtime smoothing)
SLOW_WIN_MS = 10000.0         # e.g., 10 s (packet metrics: PER, retries)

# Derived counts
SUBBLOCKS_PER_FAST = max(1, int(round(FAST_WIN_MS / SUBBLOCK_MS)))
SUBBLOCKS_PER_MED  = max(1, int(round(MED_WIN_MS / SUBBLOCK_MS)))
SUBBLOCKS_PER_SLOW = max(1, int(round(SLOW_WIN_MS / SUBBLOCK_MS)))

# Feature thresholds / tunables
CCA_SIGNAL_THRESHOLD_RATIO = 0.1    # fraction of max power to be considered busy for CCA
WIFI_PEAK_DB_ABOVE_MEAN = 5.0       # dB above mean PSD to emphasize wifi-type peaks
NOISE_EDGE_FRACTION = 0.10          # percent of PSD at edges used to estimate noise floor

# ADC / calibration placeholders (same as your file)
REF_IMPEDANCE = 50.0
ADC_SCALE = 2**10

# ------------------------
# Utility functions
# ------------------------
def dbm_from_power(power_watts):
    if power_watts <= 0:
        return -150.0
    return 10.0 * np.log10(power_watts) + 30.0

def power_spectral_density(iq_block):
    window = np.hamming(len(iq_block))
    windowed = iq_block * window
    spec = fft.fft(windowed)
    psd = np.abs(spec) ** 2 / len(iq_block)
    psd_shift = fft.fftshift(psd)
    psd_db = 10.0 * np.log10(psd_shift + 1e-12)
    return psd_db

# ------------------------
# Core per-subblock calculators
# ------------------------
def calc_time_power(iq_block):
    tp = np.abs(iq_block) ** 2
    return tp

def calc_subblock_energy_metrics(iq_block):
    time_power = calc_time_power(iq_block)
    mean_energy = float(np.mean(time_power))
    variance = float(np.var(time_power))
    power_watts_proxy = mean_energy * (1.0 / ADC_SCALE**2)
    rssi_dbm = dbm_from_power(power_watts_proxy)
    return mean_energy, variance, rssi_dbm, time_power

def calc_subblock_psd_metrics(iq_block):
    psd_db = power_spectral_density(iq_block)
    # noise floor as mean of edges
    N = len(psd_db)
    edge = max(1, int(N * NOISE_EDGE_FRACTION))
    noise_samples = np.concatenate((psd_db[:edge], psd_db[-edge:]))
    noise_floor_db = float(np.mean(noise_samples))
    signal_peak_db = float(np.max(psd_db))
    snr_db = signal_peak_db - noise_floor_db
    # Peak count heuristic
    peak_threshold = np.mean(psd_db) + WIFI_PEAK_DB_ABOVE_MEAN
    peak_count = int(np.sum(psd_db > peak_threshold))
    return noise_floor_db, snr_db, peak_count, psd_db

def calc_cca_fraction_from_time_power(time_power, signal_threshold_ratio=CCA_SIGNAL_THRESHOLD_RATIO):
    threshold = np.max(time_power) * signal_threshold_ratio
    busy = float(np.sum(time_power > threshold)) / len(time_power)
    return busy

# ------------------------
# Packet-level bookkeeping hooks
# ------------------------
class PacketEventStore:
    """
    Stores packet events for a sliding slow window. Accepts events by calling `add_packet_event`.
    Packet events are tuples: (timestamp_ms, is_wifi_bool, is_success_bool, retries_int)
    """
    def __init__(self, slow_window_ms):
        self.slow_window_ms = slow_window_ms
        self.events = deque()  # stores (ts_ms, is_wifi, is_success, retries)
    
    def add_packet_event(self, timestamp_ms, is_wifi, is_success, retries=0):
        self.events.append((timestamp_ms, is_wifi, is_success, retries))
        self._trim_old(timestamp_ms)
    
    def _trim_old(self, now_ms):
        cutoff = now_ms - self.slow_window_ms
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()
    
    def trim_to(self, now_ms):
        self._trim_old(now_ms)
    
    def stats(self):
        # Should be called after trimming with a current timestamp
        total_packets = len(self.events)
        if total_packets == 0:
            return {
                'total_packets': 0,
                'wifi_packets': 0,
                'nonwifi_packets': 0,
                'successful': 0,
                'retries_total': 0,
            }
        wifi_packets = sum(1 for e in self.events if e[1])
        nonwifi_packets = total_packets - wifi_packets
        successful = sum(1 for e in self.events if e[2])
        retries_total = sum(e[3] for e in self.events)
        return {
            'total_packets': total_packets,
            'wifi_packets': wifi_packets,
            'nonwifi_packets': nonwifi_packets,
            'successful': successful,
            'retries_total': retries_total,
        }

# ------------------------
# Streaming feature extractor
# ------------------------
class StreamingFeatureExtractor:
    def __init__(self,
                 sample_rate,
                 subblock_ms=SUBBLOCK_MS,
                 fast_subblocks=SUBBLOCKS_PER_FAST,
                 med_subblocks=SUBBLOCKS_PER_MED,
                 slow_subblocks=SUBBLOCKS_PER_SLOW):
        self.sample_rate = sample_rate
        self.subblock_ms = subblock_ms
        self.subblock_samples = 131072 # int(round(sample_rate * (subblock_ms / 1000.0)))

        # Deques to store last N subblock metrics for each timescale
        self.fast_energy = deque(maxlen=fast_subblocks)
        self.med_energy = deque(maxlen=med_subblocks)
        self.slow_energy = deque(maxlen=slow_subblocks)

        self.fast_cca = deque(maxlen=fast_subblocks)
        self.med_cca = deque(maxlen=med_subblocks)
        self.slow_cca = deque(maxlen=slow_subblocks)

        self.fast_noise = deque(maxlen=fast_subblocks)
        self.med_noise = deque(maxlen=med_subblocks)
        self.slow_noise = deque(maxlen=slow_subblocks)

        self.fast_snr = deque(maxlen=fast_subblocks)
        self.med_snr = deque(maxlen=med_subblocks)
        self.slow_snr = deque(maxlen=slow_subblocks)

        # For wifi/non-wifi airtime we will rely on packet events when available.
        self.packet_store = PacketEventStore(slow_window_ms=SLOW_WIN_MS)

        # timestamps
        self.start_time_ms = int(round(time.time() * 1000.0))
        self.last_output_ms = self.start_time_ms

    def process_subblock(self, iq_subblock, now_ms=None):
        """
        Process a single IQ subblock (complex numpy array of length subblock_samples).
        Returns a dict of features that have been updated for any timescale; keys are the feature names.
        """
        if len(iq_subblock) != self.subblock_samples:
            pass #raise ValueError(f"iq_subblock length {len(iq_subblock)} != expected {self.subblock_samples}")

        if now_ms is None:
            now_ms = int(round(time.time() * 1000.0))

        # Per-subblock metrics
        mean_energy, variance, rssi_dbm, time_power = calc_subblock_energy_metrics(iq_subblock)
        noise_floor_db, snr_db, peak_count, psd_db = calc_subblock_psd_metrics(iq_subblock)
        cca_frac = calc_cca_fraction_from_time_power(time_power)

        # Append to rolling lists
        for deq, val in [
            (self.fast_energy, mean_energy),
            (self.fast_cca, cca_frac),
            (self.fast_noise, noise_floor_db),
            (self.fast_snr, snr_db),
        ]:
            deq.append(val)

        for deq, val in [
            (self.med_energy, mean_energy),
            (self.med_cca, cca_frac),
            (self.med_noise, noise_floor_db),
            (self.med_snr, snr_db),
        ]:
            deq.append(val)

        for deq, val in [
            (self.slow_energy, mean_energy),
            (self.slow_cca, cca_frac),
            (self.slow_noise, noise_floor_db),
            (self.slow_snr, snr_db),
        ]:
            deq.append(val)

        # Build outputs for windows that are "ready" (i.e., have full-length deques OR simply produce every subblock)
        output = {
            'variance': variance
        }

        # FAST features (emit every subblock, but derived from last FAST window)
        if len(self.fast_energy) > 0:
            fast_mean_energy = float(np.mean(self.fast_energy))
            fast_cca = float(np.mean(self.fast_cca))  # cca_busy_pct
            fast_noise = float(np.mean(self.fast_noise))
            fast_snr = float(np.mean(self.fast_snr))

            # Wifi airtime heuristic (fast): if peaks present in PSD or snr high -> likely wifi
            # Here we use peak_count (from last subblock) as heuristic; in a real system use packet detection
            wifi_airtime_fast = fast_cca if peak_count > 0 else 0.0
            nonwifi_duty_fast = max(0.0, fast_cca - wifi_airtime_fast)

            output.update({
                'fast_mean_energy': fast_mean_energy,
                'cca_busy_pct_fast': fast_cca,
                'wifi_airtime_pct_fast': wifi_airtime_fast,
                'nonwifi_duty_pct_fast': nonwifi_duty_fast,
                'noise_floor_dbm_fast': fast_noise,
                'snr_db_fast': fast_snr
            })

        # MED features (emit aggregated)
        if len(self.med_energy) > 0:
            med_mean_energy = float(np.mean(self.med_energy))
            med_cca = float(np.mean(self.med_cca))
            med_noise = float(np.mean(self.med_noise))
            med_snr = float(np.mean(self.med_snr))
            wifi_airtime_med = med_cca  # smoothing assumption; replace with packet-derived when available
            nonwifi_duty_med = max(0.0, med_cca - wifi_airtime_med)
            output.update({
                'med_mean_energy': med_mean_energy,
                'cca_busy_pct_med': med_cca,
                'wifi_airtime_pct_med': wifi_airtime_med,
                'nonwifi_duty_pct_med': nonwifi_duty_med,
                'noise_floor_dbm_med': med_noise,
                'snr_db_med': med_snr
            })

        # SLOW features (packet-based; rely on packet_store for accurate values)
        self.packet_store.trim_to(now_ms)
        pkt_stats = self.packet_store.stats()
        total_pkts = pkt_stats['total_packets']

        if total_pkts > 0:
            wifi_pkts = pkt_stats['wifi_packets']
            nonwifi_pkts = pkt_stats['nonwifi_packets']
            success = pkt_stats['successful']
            retries_total = pkt_stats['retries_total']

            per_pct = 100.0 * (1.0 - (success / float(total_pkts)))
            retry_rate_pct = 100.0 * (retries_total / float(total_pkts))  # avg retries per packet scaled to percent-like
            wifi_airtime_slow = (wifi_pkts / float(total_pkts)) * float(np.mean(self.slow_cca))  # fraction of time with wifi + weighting by cca
            nonwifi_duty_slow = (nonwifi_pkts / float(total_pkts)) * float(np.mean(self.slow_cca))

            output.update({
                'per_pct_slow': per_pct,
                'retry_rate_pct_slow': retry_rate_pct,
                'cca_busy_pct_slow': float(np.mean(self.slow_cca)),
                'wifi_airtime_pct_slow': wifi_airtime_slow,
                'nonwifi_duty_pct_slow': nonwifi_duty_slow,
                'noise_floor_dbm_slow': float(np.mean(self.slow_noise)),
                'snr_db_slow': float(np.mean(self.slow_snr)),
                'total_packets_slow': total_pkts
            })
        else:
            # No packets in slow window: provide RF-only estimates and mark packet metrics as None / NaN
            output.update({
                'per_pct_slow': None,
                'retry_rate_pct_slow': None,
                'cca_busy_pct_slow': float(np.mean(self.slow_cca)) if len(self.slow_cca)>0 else 0.0,
                'wifi_airtime_pct_slow': 0.0,
                'nonwifi_duty_pct_slow': float(np.mean(self.slow_cca)) if len(self.slow_cca)>0 else 0.0,
                'noise_floor_dbm_slow': float(np.mean(self.slow_noise)) if len(self.slow_noise)>0 else 0.0,
                'snr_db_slow': float(np.mean(self.slow_snr)) if len(self.slow_snr)>0 else 0.0,
                'total_packets_slow': 0
            })

        return output

    # External API hook for packet events from a decoder
    def add_decoded_packet(self, timestamp_ms, is_wifi=True, is_success=True, retries=0):
        """
        Call this when you decode a packet (e.g., MAC/PHY) in real time.
        - timestamp_ms: absolute timestamp in ms
        - is_wifi: True if packet is classified as WiFi (else non-wifi)
        - is_success: True if packet was received correctly (else packet loss indicator)
        - retries: number of retries observed for that packet (0 if none)
        """
        self.packet_store.add_packet_event(timestamp_ms, is_wifi, is_success, retries)

# ------------------------
# Demo / test harness
# ------------------------
def demo_run(extractor, block, sample_time, channel):
    # Simulate streaming IQ: use same synthetic example as original file but feed in subblocks
#    SAMPLE_COUNT = 4096
#    np.random.seed(42)
#    noise_power = 0.01
#    noise = np.sqrt(noise_power/2) * (np.random.randn(SAMPLE_COUNT) + 1j*np.random.randn(SAMPLE_COUNT))
#
#    f_signal = 2e6
#    t = np.arange(SAMPLE_COUNT) / SAMPLE_RATE
#    signal_amplitude = 0.5
#    signal = signal_amplitude * (np.cos(2*np.pi*f_signal*t) + 1j*np.sin(2*np.pi*f_signal*t))
#    burst_start = int(SAMPLE_COUNT * 0.2)
#    burst_end   = int(SAMPLE_COUNT * 0.8)
#    iq_data = noise.copy()
#    iq_data[burst_start:burst_end] += signal[burst_start:burst_end]
#
#    # Simulate decoded packets occurring during the burst window
#    start_ms = int(round(time.time()*1000.0))
#    # assume 100 packets during the burst across the slow window
#    for i in range(100):
#        # distribute packets uniformly in burst time window
#        ts = start_ms + int((i / 100.0) * SLOW_WIN_MS / 2.0)  # some within slow window
#        # mark half as wifi successful, some with retries
#        extractor.add_decoded_packet(ts, is_wifi=True, is_success=(i%10 != 0), retries=(i%3))
#
#    # Feed IQ to extractor in subblocks
#    sb = extractor.subblock_samples
#    outputs = []
#    for i in range(0, len(iq_data), sb):
#        block = iq_data[i:i+sb]
#        if len(block) < sb:
#            # zero-pad last block for simplicity
#            block = np.pad(block, (0, sb-len(block)), mode='constant', constant_values=0+0j)
#    start = time.time()
    now_ms = int(round(time.time()*1000.0))
    out = extractor.process_subblock(block, now_ms=now_ms)
#    outputs.append(out)

    # Print last slow-window features (representative)
#    print("=== Representative Output (slow-window features) ===")
#    rep = outputs[-1]
#    keys = ['per_pct_slow', 'retry_rate_pct_slow', 'cca_busy_pct_slow',
#            'wifi_airtime_pct_slow', 'nonwifi_duty_pct_slow', 'noise_floor_dbm_slow', 'snr_db_slow', 'total_packets_slow']
#    for k in rep.keys():

#    map = { 0: 1, 1: 6, 2: 11 }
#    print(out)
    alerts = change_detection.run_pipeline({ "timestamp": sample_time, "channel": channel, "cca_busy_pct": out["cca_busy_pct_fast"], "wifi_airtime_pct": out["wifi_airtime_pct_fast"], "noise_floor_dbm": out["noise_floor_dbm_fast"], "snr_db": out["snr_db_fast"] })
#    with open("alerts.log", "a") as f:
#        f.write(str(alerts))
#    print("PROCESS_SUBBLOCK:", time.time()-start)
#    print()
    return [ out["fast_mean_energy"], out["noise_floor_dbm_fast"], out["snr_db_fast"] ], alerts, [ out["cca_busy_pct_fast"], out["fast_mean_energy"], out["wifi_airtime_pct_fast"], out["noise_floor_dbm_fast"], out["snr_db_fast"] ]

from flask import Flask, jsonify

app = Flask(__name__)

global api_output
api_output = {}
import threading
api_output_lock = threading.Lock()

# Define a JSON API route
@app.route("/sensing-data", methods=["GET"])
def get_sensing_data():
    with api_output_lock:
        return jsonify(api_output)

def extractor_thread():
    # Create extractor
    extractor = StreamingFeatureExtractor(SAMPLE_RATE)

    # Create and initialize controller
    scanner = RFScannerController()
    scanner.initialize()
#
#    # Configure parameters
#    scanner.set_center_frequency(100e6)  # 100 MHz
#    scanner.set_sample_rate(SAMPLE_RATE)
#    scanner.set_lna_gain(16)              # LNA gain 16 dB
#    scanner.set_vga_gain(20)              # VGA gain 20 dB
#
#    # Print device info
#    scanner.print_device_info()
#
#    scanner.start_stream(lambda x: demo_run(extractor, x))
#
#    time.sleep(1)
#
#    # demo_run()
#    scanner.stop_stream()
#
#    # Clean up
#    scanner.close()

    # Configure parameters
    #2412 MHz - C1
    #2437 MHz - C6
    #2462 MHz - C11
    CHANNEL_1 = 2412e6
    CHANNEL_6 = 2437e6
    CHANNEL_11 = 2462e6
    scanner.set_sample_rate(17.5e6)       # 17.5 MHz bandwidth
    scanner.set_lna_gain(16)              # LNA gain 16 dB
    scanner.set_vga_gain(20)              # VGA gain 20 dB

    # Print device info
    scanner.print_device_info()
#    outputs = []
    channel_map = { 0: 1, 1: 6, 2: 11 }
    # Check if offline model exists
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model directory not found at '{MODEL_DIR}'")
        print("Please run the 'offline_pipeline.py' script first.")
        exit()
#    csvfile = open("logs4", "a", newline="")
#    writer = csv.writer(csvfile)
    sf = []
    for idx, channel in enumerate([CHANNEL_1, CHANNEL_6, CHANNEL_11]):
        scanner.set_center_frequency(channel)
        sample_time = time.strftime("%Y-%m-%d %H:%M:%S")
        scanner.start_stream()
        time.sleep(0.150)
        scanner.stop_stream()
        #samples = scanner.get_samples()
        #print(len(samples))
        iq = scanner.get_samples()
        meta = { "frequency_hz": channel, "sample_rate": 17.5e6, "duration_seconds_requested": 0.150, "vga_gain": 20, "lna": 16 }
        # TOLOG: non wifi classifier
        infer_real_capture_airshark.main(meta, iq, "airshark_c45_model.pkl")
        sf_data, alerts, _ = demo_run(extractor, iq, sample_time, channel_map[idx])
        sf.append(np.array(sf_data))
    # Initialize the controller
    controller = OnlineController(
        num_channels=NUM_CHANNELS,
        model_dir=MODEL_DIR,
        quick_sense_duration=QUICK_SENSE_DURATION,
        alpha=ALPHA,
        weights=weights,
        data=data,
        epoch_budget=2500,
        sf=sf
    )

    print("\n--- STARTING ONLINE SIMULATION ---")
    total_reward_acc = 0

    epoch_rewards = []
    cumulative_rewards = []
    evaluation_metric = []  # (UCB, equal, random)
    damn10 = 0

    # Run the simulation
    
#    cur = 0
    output = { "scans": [ {}, {}, {}, {}, {}, {} ] }

    while True:
#        start_epoch = time.time()
        sf = []
        DWELL_TIME_INITIAL = 0.025
        for idx, channel in enumerate([CHANNEL_1, CHANNEL_6, CHANNEL_11]):
#            start = time.time()
            scanner.set_center_frequency(channel)
            sample_time = time.strftime("%Y-%m-%d %H:%M:%S")
            scanner.start_stream()
            time.sleep(DWELL_TIME_INITIAL)
            scanner.stop_stream()
#            print("TIME: ", time.time()-start)
            #samples = scanner.get_samples()
            #print(len(samples))
            iq = scanner.get_samples()
            meta = { "frequency_hz": channel, "sample_rate": 17.5e6, "duration_seconds_requested": DWELL_TIME_INITIAL, "vga_gain": 20, "lna": 16 }
            # TOLOG: non wifi classifier
            nonwifi_classifier = infer_real_capture_airshark.main(meta, iq, "airshark_c45_model.pkl")
            features = ['start_us','end_us','predicted_label','confidence','center_freq_abs_hz','bandwidth_hz','duty_cycle']
#            row_write = [time.time(), channel]
            sf_data, alerts, feature_extraction_features = demo_run(extractor, iq, sample_time, channel_map[idx])
            output["scans"][idx]["timestamp"] = time.time()
            output["scans"][idx]["center_frequency_hz"] = channel
            output["scans"][idx]["dwell_time"] = 25.0
            output["scans"][idx]["cca_busy_pct"] = feature_extraction_features[0]
            output["scans"][idx]["mean_energy"] = feature_extraction_features[1]
            output["scans"][idx]["wifi_airtime_pct"] = feature_extraction_features[2]
            output["scans"][idx]["noise_floor_dbm"] = feature_extraction_features[3]
            output["scans"][idx]["snr_db"] = feature_extraction_features[4]
            output["scans"][idx]["nonwifi_classifier_predictions"] = []
#            row_write.extend(feature_extraction_features)
            if nonwifi_classifier is not None:
                for index, row in nonwifi_classifier.iterrows():
                    output["scans"][idx]["nonwifi_classifier_predictions"].append({ feature: row[feature] for feature in features })
#            writer.writerow(row_write)
            sf.append(np.array(sf_data))
#        print("BEFORE EPOCH 1:", time.time()-start_epoch)
#        print()
        chosen_dwell_times, current_scaled_sfs, chosen_action_indices = controller.run_epoch_1(sf)
        sf = []
        for idx, channel in enumerate([CHANNEL_1, CHANNEL_6, CHANNEL_11]):
#            start = time.time()
            scanner.set_center_frequency(channel)
            sample_time = time.strftime("%Y-%m-%d %H:%M:%S")
            scanner.start_stream()
            time.sleep(chosen_dwell_times[idx] / 1000)
            scanner.stop_stream()
#            print("TIME: ", time.time()-start)
#            print("CHOSEN DWELL TIME: ", chosen_dwell_times[idx])
            #samples = scanner.get_samples()
            #print(len(samples))
            iq = scanner.get_samples()
            meta = { "frequency_hz": channel, "sample_rate": 17.5e6, "duration_seconds_requested": chosen_dwell_times[idx], "vga_gain": 20, "lna": 16 }
            # TOLOG: non wifi classifier
            nonwifi_classifier = infer_real_capture_airshark.main(meta, iq, "airshark_c45_model.pkl")
            features = ['start_us','end_us','predicted_label','confidence','center_freq_abs_hz','bandwidth_hz','duty_cycle']
            sf_data, alerts, feature_extraction_features = demo_run(extractor, iq, sample_time, channel_map[idx])
            output["scans"][idx+3]["timestamp"] = time.time()
            output["scans"][idx+3]["center_frequency_hz"] = channel
            output["scans"][idx+3]["dwell_time"] = chosen_dwell_times[idx]
            output["scans"][idx+3]["cca_busy_pct"] = feature_extraction_features[0]
            output["scans"][idx+3]["mean_energy"] = feature_extraction_features[1]
            output["scans"][idx+3]["wifi_airtime_pct"] = feature_extraction_features[2]
            output["scans"][idx+3]["noise_floor_dbm"] = feature_extraction_features[3]
            output["scans"][idx+3]["snr_db"] = feature_extraction_features[4]
            output["scans"][idx+3]["nonwifi_classifier_predictions"] = []
#            row_write = [time.time(), channel]
#            row_write.extend(feature_extraction_features)
            if nonwifi_classifier is not None:
                for index, row in nonwifi_classifier.iterrows():
                    output["scans"][idx+3]["nonwifi_classifier_predictions"].append({ feature: row[feature] for feature in features })
                #row_write.append([row[feature] for feature in features])
#            writer.writerow(row_write)
            sf.append(np.array(sf_data))
#        print("BEFORE EPOCH 2:", time.time()-start_epoch)
#        print()
        reward_this_epoch, equal_rewards, random_rewards, optimal, actual_rewards = controller.run_epoch_2(sf, chosen_dwell_times, current_scaled_sfs, chosen_action_indices)
        total_reward_acc += reward_this_epoch
        damn10 += optimal

        epoch_rewards.append(reward_this_epoch)
        cumulative_rewards.append(total_reward_acc)

        # Store values cleanly
        evaluation_metric.append((
                reward_this_epoch,
                equal_rewards,
                random_rewards,
                optimal
        ))
        output["epoch_end_timestamp"] = time.time()
        output["epoch_t"] = controller.epoch_t
        output["actual_rewards"] = [round(r, 1) for r in actual_rewards]
        output["reward_this_epoch"] = reward_this_epoch
        output["equal_rewards"] = equal_rewards
        output["random_rewards"] = random_rewards
        output["optimal"] = optimal
        with api_output_lock:
            global api_output
            api_output = copy.deepcopy(output)
    #            print("DONEEEEEEEEEEEEEE")
#        writer.writerow([time.time(), controller.epoch_t, chosen_dwell_times[0], chosen_dwell_times[1], chosen_dwell_times[2], [round(r, 1) for r in actual_rewards], evaluation_metric[-1]])
#        csvfile.flush()
#        print("EPOCH TIME: ", time.time()-start_epoch)

    # Clean up
    scanner.close()

def flask_thread():
    app.run()

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(extractor_thread)
        executor.submit(flask_thread)
