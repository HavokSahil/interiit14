#!/usr/bin/env python3
"""
infer_real_capture_airshark.py

Inference-only pipeline:
 - Reads HackRF capture metadata JSON (required)
 - Loads associated .s8 IQ file (int8 I,Q interleaved)
 - Computes STFT (NFFT=2048, 50% overlap) to obtain FFT bins
 - CROPS the FFT to the central 17.5 MHz region (Option A)
 - Aggregates the cropped FFT bins into exactly 56 output bins (312.5 kHz/bin)
 - Produces an Airshark-style PSD CSV:
       timestamp_us, noise_floor_dbm, bin_0_power_dbm, ..., bin_55_power_dbm,
       center_freq_hz, freq_min_hz, freq_max_hz, dwell_time_s, filename, md5, lna, vga_gain
 - Performs event detection, feature extraction (spread, entropy, NKLD, center, bandwidth)
 - Loads a provided C4.5-style decision tree model (joblib .pkl) and classifies events
 - Outputs:
       * <out_prefix>_psd_56bins.csv      (frame-level PSD)
       * <out_prefix>_extracted_events.csv (per-event features)
       * <out_prefix>_events_labeled.csv   (per-event features + predicted label + confidence)
       * <out_prefix>_detection_report.csv (compact start/end,label,conf,center_freq,bw,duty_cycle)
       * <out_prefix>_spectrogram.png      (visualization with detected events)
       * (optionally) decision tree PNG if model available

Important design choices (you chose Option A):
 - We crop the central 17.5 MHz of the FFT and map it to 56 bins (312.5 kHz each).
 - This reproduces the Atheros 56-bin behavior used in training.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import math
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# -----------------------------
# IMPORTANT TUNABLE PARAMETERS
# -----------------------------
# These reflect your requirement:
# - STFT settings must match training (NFFT=2048, hop=1024 => 50% overlap)
# - Output bins must be 56
NFFT = 2048
HOP = NFFT // 2               # 50% overlap
WINDOW = np.hanning(NFFT)     # Hann/Hanning window (smooths FFT)
N_OUTPUT_BINS = 56            # must match training
CROP_TOTAL_HZ = 17.5e6        # Option A: crop central 17.5 MHz

# Event detection threshold — you asked for this to be a single variable you can tweak.
# This value is dB above the median baseline (will be applied to per-frame PSD dB).
# Default set to -70 dBm absolute-like, but we compute relative baseline and add rel_thresh_db.
RELATIVE_THRESH_DB = 8.0   # frames having any bin > (baseline + RELATIVE_THRESH_DB) are considered active
MIN_EVENT_DURATION_US = 500.0  # minimum accepted event length in microseconds

# Small numeric floor to avoid log(0)
EPS = 1e-12

# -----------------------------
# Helper functions
# -----------------------------
def load_metadata(json_path):
    """Load JSON metadata describing the capture and perform basic checks."""
    with open(json_path, 'r') as f:
        meta = json.load(f)
    # required keys (based on your previously provided JSON)
    required = ['filepath', 'filename', 'frequency_hz', 'sample_rate', 'duration_seconds_requested', 'filesize_bytes']
    for k in required:
        if k not in meta:
            raise KeyError(f"Metadata JSON missing required key: {k}")
    return meta

def preprocess_iq(iq):
    """Remove DC offset and normalize int8 range to roughly [-1, 1]."""
    iq = iq - np.mean(iq)      # remove DC bias
    iq = iq / 128.0            # normalize (int8 range -128..127)
    return iq

def compute_stft_frames(iq, nfft=NFFT, hop=HOP, window=WINDOW):
    """
    Compute windowed STFT frames and return linear-power FFT (not dB) for each frame.
    Output: frames shape (num_frames, nfft) with power spectral density (|FFT|^2)
    """
    n = len(iq)
    frames = []
    idx = 0
    while idx + nfft <= n:
        seg = iq[idx:idx+nfft] * window
        spec = np.fft.fftshift(np.fft.fft(seg, n=nfft))
        power = np.abs(spec) ** 2
        frames.append(power)
        idx += hop
    if len(frames) == 0:
        return np.zeros((0, nfft)), hop
    return np.vstack(frames), hop

def central_crop_indices(nfft, sample_rate_hz, crop_total_hz=CROP_TOTAL_HZ):
    """
    Compute indices (0..nfft-1) corresponding to the central crop of width crop_total_hz.
    FFT bins are arranged with np.fft.fftshift so indices correspond to frequencies from -Fs/2..+Fs/2.
    """
    # frequency resolution per bin
    df = float(sample_rate_hz) / float(nfft)   # Hz per FFT bin
    half_crop = crop_total_hz / 2.0
    # number of bins to include (round to nearest integer)
    num_bins = int(round(crop_total_hz / df))
    if num_bins > nfft:
        raise ValueError(f"Requested crop ({crop_total_hz} Hz) larger than sampling bandwidth ({sample_rate_hz} Hz).")
    # ensure symmetry: if num_bins odd, make even to have symmetric index groups
    if num_bins % 2 != 0:
        num_bins -= 1
    # center index in fftshifted array
    center_idx = nfft // 2
    start_idx = center_idx - (num_bins // 2)
    end_idx = start_idx + num_bins  # exclusive
    inds = np.arange(start_idx, end_idx)
    return inds, df, num_bins

def aggregate_groups(frames_power, indices, n_output_bins=N_OUTPUT_BINS):
    """
    Aggregate the selected FFT bins (columns in frames_power) into exactly n_output_bins.
    We distribute the fft bins into groups as evenly as possible.
    Input:
        frames_power: shape (num_frames, nfft) linear power
        indices: array of indices selected for cropping (length crop_len)
    Output:
        agg_db: (num_frames, n_output_bins) in dB
        groups: list of arrays with indices per output bin (relative to frames_power columns)
    """
    if frames_power.size == 0:
        return np.zeros((0, n_output_bins)), []
    # cropped power
    cropped = frames_power[:, indices]   # shape (num_frames, crop_len)
    crop_len = cropped.shape[1]
    base = crop_len // n_output_bins
    rem = crop_len % n_output_bins
    groups = []
    agg_lin = np.zeros((cropped.shape[0], n_output_bins), dtype=float)
    idx = 0
    for b in range(n_output_bins):
        gs = base + (1 if b < rem else 0)
        inds = np.arange(idx, idx + gs)
        groups.append(indices[inds])   # store original FFT indices contributing to this bin
        # sum linear power over that group
        agg_lin[:, b] = np.sum(cropped[:, inds], axis=1)
        idx += gs
    # convert to dB (10*log10 of linear power)
    agg_db = 10.0 * np.log10(agg_lin + EPS)
    return agg_db, groups

# Event detection & features (Airshark-inspired)
def detect_events_from_psd(agg_db, sample_rate_hz, hop_samples, rel_thresh_db=RELATIVE_THRESH_DB, min_event_us=MIN_EVENT_DURATION_US):
    """
    Detect events on aggregated per-frame PSD in dB.
    Baseline estimate = median of per-frame medians.
    Event when any bin in a frame > baseline + rel_thresh_db.
    Returns:
        events: list of dicts with start_frame, end_frame, start_us, end_us, duration_us
        baseline_db: median baseline value in dB
        threshold_db: baseline_db + rel_thresh_db
    """
    if agg_db.size == 0:
        return [], None, None
    per_frame_med = np.median(agg_db, axis=1)
    baseline_db = float(np.median(per_frame_med))
    threshold_db = baseline_db + float(rel_thresh_db)
    events = []
    in_event = False
    start_frame = None
    num_frames = agg_db.shape[0]
    for f in range(num_frames):
        if np.max(agg_db[f, :]) > threshold_db:
            if not in_event:
                in_event = True
                start_frame = f
        else:
            if in_event:
                end_frame = f - 1
                # compute times (samples) of frame edges approximately
                start_sample = start_frame * hop_samples
                end_sample = end_frame * hop_samples + (NFFT - hop_samples)
                start_us = (start_sample / sample_rate_hz) * 1e6
                end_us = (end_sample / sample_rate_hz) * 1e6
                duration_us = end_us - start_us
                if duration_us >= min_event_us:
                    events.append({'start_frame': start_frame, 'end_frame': end_frame,
                                   'start_us': start_us, 'end_us': end_us, 'duration_us': duration_us})
                in_event = False
    # trailing event
    if in_event:
        end_frame = num_frames - 1
        start_sample = start_frame * hop_samples
        end_sample = end_frame * hop_samples + (NFFT - hop_samples)
        start_us = (start_sample / sample_rate_hz) * 1e6
        end_us = (end_sample / sample_rate_hz) * 1e6
        duration_us = end_us - start_us
        if duration_us >= min_event_us:
            events.append({'start_frame': start_frame, 'end_frame': end_frame,
                           'start_us': start_us, 'end_us': end_us, 'duration_us': duration_us})
    return events, baseline_db, threshold_db

def compute_nkld(event_bins_db, baseline_bins_db):
    """Normalized Kullback-Leibler divergence as in Airshark."""
    e_lin = 10 ** ((event_bins_db - 30.0) / 10.0)
    b_lin = 10 ** ((baseline_bins_db - 30.0) / 10.0)
    e_pdf = e_lin / np.sum(e_lin)
    b_pdf = b_lin / np.sum(b_lin)
    nkld = np.sum(e_pdf * np.log((e_pdf + EPS) / (b_pdf + EPS)))
    return float(nkld / (math.log(len(e_pdf) + EPS)))

def add_extra_features(avg_bin_dbm, baseline_dbm):
    """Compute spread, kurtosis, entropy, nkld for a given averaged event spectrum."""
    bins_lin = 10 ** ((avg_bin_dbm - 30.0) / 10.0)
    bins_norm = bins_lin / (np.sum(bins_lin) + EPS)
    spread = float(np.std(avg_bin_dbm))
    s = float(np.std(avg_bin_dbm))
    kurtosis = float(np.mean((avg_bin_dbm - np.mean(avg_bin_dbm))**4) / ((s**4) + EPS)) if s > 0 else 0.0
    entropy = float(-np.sum(bins_norm * np.log2(bins_norm + EPS)))
    nkld = compute_nkld(avg_bin_dbm, baseline_dbm)
    return spread, kurtosis, entropy, nkld

def extract_features_from_events(agg_db, events, baseline_bin_db, center_freq_hz, sample_rate_hz, hop_samples, groups):
    """
    For each detected event, compute the Airshark-style features used by the classifier.
    Returns a DataFrame with one row per event.
    """
    rows = []
    if agg_db.size == 0 or len(events) == 0:
        return pd.DataFrame(rows)
    n_bins = agg_db.shape[1]
    bin_width_hz = float(CROP_TOTAL_HZ) / float(n_bins)  # 17.5e6 / 56 => 312.5 kHz
    mid_idx = (n_bins - 1) / 2.0
    total_observation_us = (agg_db.shape[0] * hop_samples / sample_rate_hz) * 1e6

    for ev in events:
        s = ev['start_frame']
        e = ev['end_frame']
        frames = agg_db[s:e+1, :]                 # frames × bins (dB)
        avg_bin_db = np.mean(frames, axis=0)     # averaged over frames
        # linear representation for centroid
        avg_bin_lin = 10 ** ((avg_bin_db - 30.0) / 10.0)
        avg_bin_lin = np.maximum(avg_bin_lin, EPS)
        weights = avg_bin_lin / np.sum(avg_bin_lin)
        bin_indices = np.arange(n_bins)
        center_bin = float(np.sum(weights * bin_indices))
        # bandwidth_bins: count bins with significant fraction of energy (relative threshold)
        bw_bins = int(np.sum(weights > (0.05 * np.max(weights))))
        duration_sec = float(ev['duration_us'] / 1e6)
        avg_rssi_db = float(np.mean(np.median(frames, axis=1)))
        rssi_range_db = float(np.max(avg_bin_db) - np.min(avg_bin_db))
        spread, kurtosis, entropy, nkld = add_extra_features(avg_bin_db, baseline_bin_db)
        # absolute center frequency of this event (Hz)
        center_freq_abs_hz = float(center_freq_hz + (center_bin - mid_idx) * bin_width_hz)
        bandwidth_hz = float(bw_bins * bin_width_hz)
        duty_cycle = float(ev['duration_us'] / total_observation_us) if total_observation_us > 0 else 0.0

        row = {
            'start_us': float(ev['start_us']),
            'end_us': float(ev['end_us']),
            'duration_sec': duration_sec,
            'center_bin': center_bin,
            'bandwidth_bins': bw_bins,
            'avg_rssi_dbm': avg_rssi_db,
            'rssi_range_db': rssi_range_db,
            'spread': spread,
            'kurtosis': kurtosis,
            'entropy': entropy,
            'nkld': nkld,
            'center_freq_abs_hz': center_freq_abs_hz,
            'bandwidth_hz': bandwidth_hz,
            'duty_cycle': duty_cycle
        }
        rows.append(row)
    return pd.DataFrame(rows)

# -----------------------------
# Main inference flow
# -----------------------------
def main(meta, iq, model):
#    p = argparse.ArgumentParser(description="Convert .s8 (with JSON meta) into Airshark 56-bin PSD and classify events.")
#    p.add_argument('--model', required=True, help='Path to pretrained C4.5 decision tree model (joblib .pkl)')
#    p.add_argument('--out-prefix', default='capture_out', help='Prefix for output files')
#    p.add_argument('--rel-thresh-db', type=float, default=RELATIVE_THRESH_DB, help='Relative dB above baseline to detect events (tweakable)')
#    p.add_argument('--min-event-us', type=float, default=MIN_EVENT_DURATION_US, help='Minimum event duration (us)')
#    args = p.parse_args()
#
#    meta = { "frequency_hz": CHANNEL_1, "sample_rate": 17.5e6, "duration_seconds_requested": 0.150, "vga_gain": 20, "lna": 16 }

    # Extract metadata fields (all from JSON; none hard-coded)
    center_freq_hz = float(meta['frequency_hz'])
    sample_rate_hz = float(meta['sample_rate'])
    dwell_time_s = float(meta.get('duration_seconds_requested', (meta.get('filesize_bytes', 0) / 2) / sample_rate_hz))
    md5 = meta.get('md5', '')
    lna = meta.get('lna', None)
    vga_gain = meta.get('vga_gain', meta.get('vga', None))
    # Log
    print(f"[*] Metadata: center_freq={center_freq_hz} Hz, sample_rate={sample_rate_hz} Hz, dwell_time={dwell_time_s}s")

    print(f"    Loaded {len(iq)} complex samples.")

    # 2) Preprocess IQ (DC removal, normalization)
    iq = preprocess_iq(iq)

    # 3) STFT frames -> linear power per FFT bin
    print("[*] Computing STFT frames...")
    frames_power, hop_used = compute_stft_frames(iq, nfft=NFFT, hop=HOP, window=WINDOW)
    num_frames = frames_power.shape[0]
    print(f"    Number of frames: {num_frames}, NFFT={NFFT}, hop={hop_used}")

    # 4) Determine central crop indices that correspond to central CROP_TOTAL_HZ
    print("[*] Computing central crop indices for 17.5 MHz region...")
    crop_inds, df_bin, crop_len = central_crop_indices(NFFT, sample_rate_hz, crop_total_hz=CROP_TOTAL_HZ)
    print(f"    Crop length (fft bins): {crop_len}, df per FFT bin = {df_bin:.3f} Hz")

    # 5) Aggregate cropped FFT bins -> 56 output bins (312.5 kHz each)
    print("[*] Aggregating to 56 output bins (312.5 kHz each)...")
    agg_db, groups = aggregate_groups(frames_power, crop_inds, n_output_bins=N_OUTPUT_BINS)
    print(f"    Aggregated PSD shape: {agg_db.shape} (frames x bins)")

    # 6) Build Airshark-style PSD DataFrame (frame-level)
    print("[*] Building frame-level PSD DataFrame...")
    bin_cols = [f'bin_{i}_power_dbm' for i in range(N_OUTPUT_BINS)]
    timestamps_us = (np.arange(num_frames) * hop_used / sample_rate_hz) * 1e6   # frame -> approximate time in microseconds
    # noise_floor_dbm per-frame: use median across bins as a per-frame estimate
    noise_floor_per_frame = np.median(agg_db, axis=1)
    df_psd = pd.DataFrame(agg_db, columns=bin_cols)
    df_psd.insert(0, 'noise_floor_dbm', noise_floor_per_frame)
    df_psd.insert(0, 'timestamp_us', timestamps_us)
    # add metadata columns (explicitly from JSON)
    df_psd['center_freq_hz'] = center_freq_hz
    df_psd['freq_min_hz'] = center_freq_hz - (CROP_TOTAL_HZ / 2.0)
    df_psd['freq_max_hz'] = center_freq_hz + (CROP_TOTAL_HZ / 2.0)
    df_psd['dwell_time_s'] = dwell_time_s
    # df_psd['filename'] = filename
    df_psd['md5'] = md5
    df_psd['lna'] = lna
    df_psd['vga_gain'] = vga_gain

#    psd_csv = f"{args.out_prefix}_psd_56bins.csv"
#    df_psd.to_csv(psd_csv, index=False)
#    print(f"[*] Saved PSD CSV: {psd_csv}")

    # 7) Event detection on aggregated PSD
    print("[*] Detecting events (relative threshold) ...")
    events, baseline_db, threshold_db = detect_events_from_psd(agg_db, sample_rate_hz, hop_used,
                                                               rel_thresh_db=RELATIVE_THRESH_DB,
                                                               min_event_us=MIN_EVENT_DURATION_US)
#    print(f"    Baseline (median of per-frame medians): {baseline_db:.3f} dB")
#    print(f"    Detection threshold (baseline + rel): {threshold_db:.3f} dB")
    print(f"    Number of detected events: {len(events)}")

    # 8) Compute baseline per-bin using frames below threshold (used for NKLD)
    per_frame_max = np.max(agg_db, axis=1)
    noise_mask = per_frame_max <= threshold_db
    if np.any(noise_mask):
        baseline_bin_db = np.median(agg_db[noise_mask, :], axis=0)
    else:
        baseline_bin_db = np.median(agg_db, axis=0)  # fallback

    # 9) Extract features per event
    print("[*] Extracting Airshark features for each event...")
    features_df = extract_features_from_events(agg_db, events, baseline_bin_db, center_freq_hz, sample_rate_hz, hop_used, groups)
    if features_df.empty:
        print("[!] No valid events extracted; exiting.")
        return
#    feats_csv = f"{args.out_prefix}_extracted_events.csv"
#    features_df.to_csv(feats_csv, index=False)
#    print(f"[*] Saved extracted features: {feats_csv}")

    # 10) Load pretrained model
    model_path = Path(model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Provide a trained C4.5-style tree .pkl")
    model = joblib.load(str(model_path))
    print(f"[*] Loaded classifier model from {model_path}")

    # 11) Match features order expected by model (same used in training)
    feature_columns = ['duration_sec','center_bin','bandwidth_bins','avg_rssi_dbm','rssi_range_db','spread','kurtosis','entropy','nkld']
    for col in feature_columns:
        if col not in features_df.columns:
            raise KeyError(f"Expected feature '{col}' not found in extracted features.")
    X = features_df[feature_columns].values

    # 12) Predict and compute confidence (max probability)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    confidences = np.max(probs, axis=1)
    features_df['predicted_label'] = preds
    features_df['confidence'] = confidences

    # 13) Compute duty cycle relative to total capture for each event (and per-class summary)
    total_capture_sec = float(len(iq) / sample_rate_hz)
    features_df['duty_cycle_fraction'] = features_df['duration_sec'] / total_capture_sec
    # per-class duty cycle summary
    duty_summary = features_df.groupby('predicted_label')['duration_sec'].sum() / total_capture_sec
    print("[*] Duty cycle fractions per predicted class:")
    print(duty_summary.to_string())

    # 14) Save labeled events and compact report
#    labeled_csv = f"{args.out_prefix}_events_labeled.csv"
#    features_df.to_csv(labeled_csv, index=False)
    report_df = features_df[['start_us','end_us','predicted_label','confidence','center_freq_abs_hz','bandwidth_hz','duty_cycle']]
    # ensure columns in report: 'duty_cycle' exists? our extract uses 'duty_cycle' as fraction per event
    if 'duty_cycle' not in features_df.columns:
        report_df['duty_cycle'] = features_df['duty_cycle_fraction']
    return report_df
#    report_csv = f"{args.out_prefix}_detection_report.csv"
#    report_df.to_csv(report_csv, index=False)
#    print(f"[*] Saved labeled events: {labeled_csv}")
#    print(f"[*] Saved detection report: {report_csv}")

    # 15) Visualization: spectrogram + event overlays
#    try:
#        print("[*] Creating spectrogram visualization with detected events...")
#        plt.figure(figsize=(14,6))
#        # show frames x bins (transpose for bin index vertical)
#        plt.imshow(agg_db.T, aspect='auto', origin='lower', cmap='inferno',
#                   extent=[timestamps_us[0], timestamps_us[-1] if len(timestamps_us)>0 else 0, 0, N_OUTPUT_BINS])
#        plt.colorbar(label='Power (dB)')
#        plt.xlabel('Time (us)')
#        plt.ylabel('Bin index (0..55)')
#        plt.title(f"56-bin spectrogram (center {center_freq_hz/1e6:.3f} MHz)")
#
#        # overlay events as horizontal spans
#        for _, r in features_df.iterrows():
#            start = r['start_us']
#            end = r['end_us']
#            plt.axvspan(start, end, color='white', alpha=0.15)
#
#        png_path = f"{args.out_prefix}_spectrogram.png"
#        plt.savefig(png_path, dpi=200, bbox_inches='tight')
#        plt.close()
#        print(f"[*] Saved spectrogram visualization: {png_path}")
#    except Exception as e:
#        print(f"[!] Visualization failed: {e}")
#
#    # 16) (Optional) Save decision tree figure for inspection (if model was trained using sklearn)
#    try:
#        from sklearn import tree as sktree
#        plt.figure(figsize=(20,10))
#        sktree.plot_tree(model, feature_names=feature_columns, class_names=model.classes_, filled=True, rounded=True, fontsize=8)
#        tree_png = f"{args.out_prefix}_decision_tree.png"
#        plt.title("C4.5-style decision tree (loaded model)")
#        plt.savefig(tree_png, dpi=200, bbox_inches='tight')
#        plt.close()
#        print(f"[*] Saved decision tree visualization: {tree_png}")
#    except Exception as e:
#        print(f"[!] Decision tree visualization skipped or failed: {e}")
#
    print("[*] Inference pipeline completed successfully.")
