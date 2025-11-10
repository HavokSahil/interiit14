# RF Signal Feature Extraction Pipeline

**Author:** RF Signal Processing Team  
**Date:** October 25, 2025  
**Version:** 1.0

## Overview

This project provides a comprehensive Python implementation for extracting key features from RF (Radio Frequency) signals. It supports both real-valued waveform data and complex IQ (In-phase/Quadrature) data.

## Features Implemented

### 1. **Signal Conversions**
- **Waveform → IQ Conversion**: Uses Hilbert Transform
- **IQ → Waveform Conversion**: 3-step process (interpolation, frequency shift, real extraction)

### 2. **Spectral Analysis**
- **Spectrogram**: Time-frequency representation using STFT
- **Power Spectral Density (PSD)**: FFT-based frequency domain analysis
- **Center Frequency Estimation**: Peak detection and weighted averaging methods
- **Bandwidth Estimation**: Threshold-based occupied bandwidth calculation

### 3. **Time-Domain Analysis**
- **Burst Detection**: Envelope-based threshold detection
- **Duty Cycle Computation**: Active time ratio calculation
- **Signal Envelope Extraction**: Magnitude-based envelope tracking

### 4. **Modulation Analysis**
- **Simple Modulation Classification**: Heuristic rule-based classifier
- Distinguishes between AM, FM/PM, PSK, and QAM-like signals
- Uses amplitude variance and phase statistics

### 5. **Visualization**
- Spectrogram plots
- Power spectral density plots
- IQ constellation diagrams
- Time-domain I/Q waveforms with envelope

## Project Structure

```
├── README.md                          # This file
├── Report.txt                         # Detailed results and methodology
├── Codes/
│   ├── signal_features.py            # Main feature extraction module
│   └── demo_feature_extraction.py    # Demonstration/testing script
└── test_results/
    ├── qpsk_*.png                    # QPSK signal visualizations (4 files)
    ├── fm_*.png                      # FM signal visualizations (4 files)
    ├── burst_*.png                   # Burst signal visualizations (4 files)
    └── am_*.png                      # AM signal visualizations (4 files)
```

## Installation

### Requirements
```bash
pip install numpy scipy matplotlib
```

**Python Version:** 3.6 or higher

### Optional Dependencies
```bash
# For advanced features (not required for basic functionality)
pip install librosa torch
```

## Quick Start

### 1. Import the Module
```python
import numpy as np
from signal_features import *
```

### 2. Load Your IQ Data
```python
# Option A: From numpy array
iq_data = np.load('your_iq_samples.npy')
fs = 1e6  # Sampling rate in Hz

# Option B: Generate test signal
t = np.linspace(0, 0.001, 1000)
fc = 100e3
iq_data = np.exp(1j * 2 * np.pi * fc * t)
```

### 3. Extract Features

#### Spectrogram
```python
f, t, Sxx_db = compute_spectrogram(iq_data, fs, nperseg=256)
plot_spectrogram_result(f, t, Sxx_db)
```

#### Center Frequency & Bandwidth
```python
center_freq, psd, freqs = estimate_center_frequency(iq_data, fs)
bandwidth, f_lower, f_upper = estimate_bandwidth(iq_data, fs, threshold_db=-10)

print(f"Center Frequency: {center_freq/1e6:.3f} MHz")
print(f"Bandwidth: {bandwidth/1e3:.3f} kHz")
```

#### Burst Detection
```python
bursts, envelope = detect_bursts(iq_data, fs, threshold_db=-10)
duty_cycle, active_time, total_time = compute_duty_cycle(bursts, len(iq_data), fs)

print(f"Bursts detected: {len(bursts)}")
print(f"Duty cycle: {duty_cycle*100:.2f}%")
```

#### Modulation Classification
```python
mod_type, features = classify_modulation_simple(iq_data, fs)
print(f"Modulation type: {mod_type}")
```

#### Waveform ↔ IQ Conversion
```python
# Real waveform to IQ
iq_data = waveform_to_iq(real_waveform, fs)

# IQ to real waveform
real_waveform, fs_real = iq_to_waveform(iq_data, fs)
```

## Function Reference

### Core Functions

#### `waveform_to_iq(waveform_data, fs)`
Converts real-valued waveform to complex IQ samples using Hilbert transform.

**Parameters:**
- `waveform_data`: Real-valued time-domain signal
- `fs`: Sampling frequency in Hz

**Returns:**
- Complex IQ samples (ndarray)

---

#### `iq_to_waveform(iq_data, fs)`
Converts complex IQ to real-valued waveform.

**Parameters:**
- `iq_data`: Complex IQ samples
- `fs`: Original IQ sampling frequency

**Returns:**
- `waveform_data`: Real signal
- `fs_real`: Real signal sampling frequency (2× original)

---

#### `compute_spectrogram(iq_data, fs, nperseg=256, noverlap=None)`
Computes time-frequency spectrogram using STFT.

**Parameters:**
- `iq_data`: Complex IQ samples
- `fs`: Sampling frequency
- `nperseg`: FFT window size
- `noverlap`: Overlap samples

**Returns:**
- `f`: Frequency array
- `t`: Time array
- `Sxx_db`: Spectrogram in dB

---

#### `estimate_center_frequency(iq_data, fs, method='peak')`
Estimates center frequency of signal.

**Parameters:**
- `iq_data`: Complex IQ samples
- `fs`: Sampling frequency
- `method`: 'peak' or 'weighted'

**Returns:**
- `center_freq`: Estimated center frequency (Hz)
- `psd`: Power spectral density
- `freqs`: Frequency array

---

#### `estimate_bandwidth(iq_data, fs, threshold_db=-3)`
Estimates occupied bandwidth.

**Parameters:**
- `iq_data`: Complex IQ samples
- `fs`: Sampling frequency
- `threshold_db`: Power threshold relative to peak (-3dB, -10dB, etc.)

**Returns:**
- `bandwidth`: Bandwidth in Hz
- `f_lower`: Lower frequency bound
- `f_upper`: Upper frequency bound

---

#### `detect_bursts(iq_data, fs, threshold_db=-10, min_duration=1e-6)`
Detects burst signals in time domain.

**Parameters:**
- `iq_data`: Complex IQ samples
- `fs`: Sampling frequency
- `threshold_db`: Detection threshold relative to median
- `min_duration`: Minimum burst duration (seconds)

**Returns:**
- `bursts`: List of (start_idx, end_idx, duration) tuples
- `envelope`: Signal envelope

---

#### `compute_duty_cycle(bursts, total_samples, fs)`
Computes duty cycle from detected bursts.

**Parameters:**
- `bursts`: Output from `detect_bursts()`
- `total_samples`: Total signal length
- `fs`: Sampling frequency

**Returns:**
- `duty_cycle`: Duty cycle (0 to 1)
- `active_time`: Total active time (seconds)
- `total_time`: Total duration (seconds)

---

#### `classify_modulation_simple(iq_data, fs)`
Simple rule-based modulation classification.

**Parameters:**
- `iq_data`: Complex IQ samples
- `fs`: Sampling frequency

**Returns:**
- `mod_type`: Estimated modulation type (string)
- `features`: Dictionary of extracted features

**Note:** For production use, consider CNN-based deep learning methods for better accuracy.

## Running the Demo

```bash
cd Codes
python demo_feature_extraction.py
```

This will:
1. Generate 4 synthetic test signals (QPSK, FM, Burst, AM)
2. Extract all features from each signal
3. Generate 16 visualizations (4 per signal type)
4. Create a summary report

## Test Results

Processed signal types:
- **QPSK**: Quadrature Phase Shift Keying
- **FM**: Frequency Modulation
- **Burst**: Pulsed carrier signal (40% duty cycle)
- **AM**: Amplitude Modulation

See `Report.txt` for detailed results and methodology.

## Methodology

### Spectrogram
- Uses Short-Time Fourier Transform (STFT)
- Hamming window with 50% overlap
- Power spectral density in dB scale

### IQ ↔ Waveform Conversion
- **Waveform → IQ**: Hilbert transform creates analytic signal
- **IQ → Waveform**: Interpolate (2×), frequency shift (fs/2), extract real part

### Center Frequency
- Peak detection: Finds maximum in FFT spectrum
- Weighted average: Power-weighted frequency mean (alternative)

### Bandwidth
- Threshold method: -10dB (or -3dB) below peak power
- Finds frequency bounds where power crosses threshold

### Burst Detection
- Envelope extraction using magnitude |IQ|
- Threshold detection in dB scale
- Minimum duration filtering

### Modulation Classification
- Heuristic approach using:
  - Amplitude variation coefficient
  - Phase standard deviation
- **Note**: Deep learning (CNN) methods achieve >85% accuracy

## Integration into Sensing Pipeline

All functions are modular and can be integrated independently:

```python
# Example pipeline
def rf_sensing_pipeline(iq_data, fs):
    # Step 1: Spectral analysis
    center_freq, _, _ = estimate_center_frequency(iq_data, fs)
    bandwidth, _, _ = estimate_bandwidth(iq_data, fs)

    # Step 2: Temporal analysis
    bursts, envelope = detect_bursts(iq_data, fs)
    duty_cycle, _, _ = compute_duty_cycle(bursts, len(iq_data), fs)

    # Step 3: Classification
    mod_type, features = classify_modulation_simple(iq_data, fs)

    # Step 4: Return results
    return {
        'center_freq': center_freq,
        'bandwidth': bandwidth,
        'duty_cycle': duty_cycle,
        'modulation': mod_type,
        'num_bursts': len(bursts)
    }
```

## Assumptions

1. **Sampling Rate**: Nyquist criterion is satisfied (fs ≥ 2 × signal_bandwidth)
2. **Signal Format**: IQ data is baseband or centered around DC
3. **Noise Model**: Additive White Gaussian Noise (AWGN)
4. **Modulation Classification**: Simple heuristics; use ML for production

## Future Enhancements

- [ ] CNN-based modulation classification
- [ ] Support for more modulation types (16-QAM, 64-QAM, OFDM)
- [ ] Advanced burst detection with adaptive thresholding
- [ ] Real-time processing support
- [ ] GNU Radio integration
- [ ] RTL-SDR / HackRF / USRP support

## References

1. PySDR - Python for Software Defined Radio: https://pysdr.org
2. SciPy Signal Processing Documentation
3. "Automatic Modulation Classification Using Deep Learning" (O'Shea et al., 2016)
4. "Fundamentals of FFT-Based Signal Analysis" - National Instruments

## License

MIT License - Free for educational and commercial use

## Contact

For questions or contributions, please open an issue on the repository.

---

**Happy Signal Processing!** 📡🔬
