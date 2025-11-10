"""
RF Signal Feature Extraction Module
====================================
This module provides functions for extracting various features from RF signals.
Supports both waveform (real) and IQ (complex) data.

Author: RF Signal Processing Pipeline
Date: October 2025
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import hilbert, spectrogram, welch, find_peaks
import matplotlib.pyplot as plt


def waveform_to_iq(waveform_data, fs):
    """
    Convert real-valued waveform to complex IQ samples.

    Uses Hilbert transform to create analytic signal.

    Parameters:
    -----------
    waveform_data : array_like
        Real-valued time-domain signal
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    iq_data : ndarray (complex)
        Complex IQ samples

    Methodology:
    ------------
    IQ(t) = real(t) + j * HilbertTransform(real(t))
    The Hilbert transform provides 90-degree phase shift.
    """
    # Apply Hilbert transform to get analytic signal
    iq_data = hilbert(waveform_data)

    return iq_data


def iq_to_waveform(iq_data, fs):
    """
    Convert complex IQ samples to real-valued waveform.

    Three-step process: interpolation, frequency shift, take real part.

    Parameters:
    -----------
    iq_data : array_like (complex)
        Complex IQ samples
    fs : float
        Original IQ sampling frequency in Hz

    Returns:
    --------
    waveform_data : ndarray (real)
        Real-valued time-domain signal
    fs_real : float
        Real signal sampling frequency (2x original)

    Methodology:
    ------------
    1. Interpolate by factor of 2 to avoid aliasing
    2. Frequency shift by fs/2 to move to positive frequencies
    3. Take real part to obtain real-valued signal
    """
    # Step 1: Interpolate by factor of 2
    from scipy.signal import resample_poly

    iq_interp_real = resample_poly(iq_data.real, up=2, down=1, padtype='line')
    iq_interp_imag = resample_poly(iq_data.imag, up=2, down=1, padtype='line')
    iq_interp = iq_interp_real + 1j * iq_interp_imag

    # Step 2: Frequency shift by fs/2
    fs_real = fs * 2
    freq_shift = fs / 2
    time_vector = np.arange(len(iq_interp)) / fs_real
    complex_sine = np.exp(1j * 2 * np.pi * freq_shift * time_vector)
    iq_shifted = iq_interp * complex_sine

    # Step 3: Take real part
    waveform_data = iq_shifted.real

    return waveform_data, fs_real


def compute_spectrogram(iq_data, fs, nperseg=256, noverlap=None):
    """
    Compute spectrogram of IQ signal.

    Parameters:
    -----------
    iq_data : array_like (complex)
        Complex IQ samples
    fs : float
        Sampling frequency in Hz
    nperseg : int
        Length of each segment (window size)
    noverlap : int
        Number of points to overlap between segments

    Returns:
    --------
    f : ndarray
        Array of sample frequencies
    t : ndarray
        Array of segment times
    Sxx : ndarray
        Spectrogram (power spectral density)

    Methodology:
    ------------
    Uses Short-Time Fourier Transform (STFT) to compute
    time-frequency representation. Window size determines
    time-frequency resolution tradeoff.
    """
    if noverlap is None:
        noverlap = nperseg // 2

    f, t, Sxx = spectrogram(iq_data, fs=fs, nperseg=nperseg, 
                            noverlap=noverlap, mode='psd')

    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-12)  # Add small value to avoid log(0)

    return f, t, Sxx_db


def estimate_center_frequency(iq_data, fs, method='peak'):
    """
    Estimate center frequency of the signal.

    Parameters:
    -----------
    iq_data : array_like (complex)
        Complex IQ samples
    fs : float
        Sampling frequency in Hz
    method : str
        'peak' - finds peak in PSD
        'weighted' - weighted average of frequency components

    Returns:
    --------
    center_freq : float
        Estimated center frequency in Hz
    psd : ndarray
        Power spectral density
    freqs : ndarray
        Frequency array

    Methodology:
    ------------
    Method 1 (peak): Find frequency bin with maximum power
    Method 2 (weighted): Compute power-weighted average frequency
    """
    # Compute FFT
    N = len(iq_data)
    yf = fft(iq_data)
    xf = fftfreq(N, 1/fs)

    # Shift to center zero frequency
    yf_shifted = fftshift(yf)
    xf_shifted = fftshift(xf)

    # Compute power spectral density
    psd = np.abs(yf_shifted)**2 / N

    if method == 'peak':
        # Find peak frequency
        peak_idx = np.argmax(psd)
        center_freq = xf_shifted[peak_idx]

    elif method == 'weighted':
        # Power-weighted average frequency
        # Only use positive power values
        psd_positive = np.maximum(psd, 0)
        total_power = np.sum(psd_positive)
        if total_power > 0:
            center_freq = np.sum(xf_shifted * psd_positive) / total_power
        else:
            center_freq = 0.0
    else:
        raise ValueError(f"Unknown method: {method}")

    return center_freq, psd, xf_shifted


def estimate_bandwidth(iq_data, fs, threshold_db=-3):
    """
    Estimate occupied bandwidth of signal.

    Parameters:
    -----------
    iq_data : array_like (complex)
        Complex IQ samples
    fs : float
        Sampling frequency in Hz
    threshold_db : float
        Power threshold in dB relative to peak (typically -3dB or -10dB)

    Returns:
    --------
    bandwidth : float
        Estimated bandwidth in Hz
    f_lower : float
        Lower frequency bound
    f_upper : float
        Upper frequency bound

    Methodology:
    ------------
    1. Compute power spectral density
    2. Find peak power
    3. Find frequencies where power drops below threshold
    4. Bandwidth = f_upper - f_lower
    """
    # Compute PSD
    center_freq, psd, freqs = estimate_center_frequency(iq_data, fs, method='peak')

    # Convert PSD to dB
    psd_db = 10 * np.log10(psd + 1e-12)

    # Find peak power
    peak_power_db = np.max(psd_db)
    threshold_power_db = peak_power_db + threshold_db

    # Find frequencies above threshold
    above_threshold = psd_db >= threshold_power_db
    indices = np.where(above_threshold)[0]

    if len(indices) > 0:
        f_lower = freqs[indices[0]]
        f_upper = freqs[indices[-1]]
        bandwidth = f_upper - f_lower
    else:
        bandwidth = 0.0
        f_lower = 0.0
        f_upper = 0.0

    return bandwidth, f_lower, f_upper


def detect_bursts(iq_data, fs, threshold_db=-10, min_duration=1e-6):
    """
    Detect burst signals in time-domain data.

    Parameters:
    -----------
    iq_data : array_like (complex)
        Complex IQ samples
    fs : float
        Sampling frequency in Hz
    threshold_db : float
        Detection threshold in dB relative to median power
    min_duration : float
        Minimum burst duration in seconds

    Returns:
    --------
    bursts : list of tuples
        Each tuple contains (start_idx, end_idx, duration_sec)
    envelope : ndarray
        Signal envelope (magnitude)

    Methodology:
    ------------
    1. Compute signal envelope using magnitude
    2. Convert to dB scale
    3. Threshold detection to find burst regions
    4. Filter out bursts shorter than min_duration
    """
    # Compute envelope (magnitude)
    envelope = np.abs(iq_data)

    # Convert to dB
    envelope_db = 20 * np.log10(envelope + 1e-12)

    # Compute threshold
    median_power = np.median(envelope_db)
    threshold = median_power + threshold_db

    # Find samples above threshold
    above_threshold = envelope_db > threshold

    # Find burst boundaries (transitions)
    diff = np.diff(above_threshold.astype(int))
    burst_starts = np.where(diff == 1)[0] + 1
    burst_ends = np.where(diff == -1)[0] + 1

    # Handle edge cases
    if len(above_threshold) > 0 and above_threshold[0]:
        burst_starts = np.concatenate([[0], burst_starts])
    if len(above_threshold) > 0 and above_threshold[-1]:
        burst_ends = np.concatenate([burst_ends, [len(above_threshold)]])

    # Create burst list with durations
    bursts = []
    min_samples = int(min_duration * fs)

    for start, end in zip(burst_starts, burst_ends):
        duration = (end - start) / fs
        if (end - start) >= min_samples:
            bursts.append((start, end, duration))

    return bursts, envelope


def compute_duty_cycle(bursts, total_samples, fs):
    """
    Compute duty cycle from detected bursts.

    Parameters:
    -----------
    bursts : list of tuples
        Output from detect_bursts()
    total_samples : int
        Total number of samples in signal
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    duty_cycle : float
        Duty cycle (0 to 1)
    active_time : float
        Total active time in seconds
    total_time : float
        Total signal duration in seconds

    Methodology:
    ------------
    Duty Cycle = (Total Active Time) / (Total Time)
    Active time is sum of all burst durations.
    """
    # Calculate total burst time
    active_samples = sum([end - start for start, end, _ in bursts])

    # Calculate times
    active_time = active_samples / fs
    total_time = total_samples / fs

    # Calculate duty cycle
    if total_time > 0:
        duty_cycle = active_time / total_time
    else:
        duty_cycle = 0.0

    return duty_cycle, active_time, total_time


def classify_modulation_simple(iq_data, fs):
    """
    Simple rule-based modulation type classification.

    NOTE: This is a simplified heuristic approach.
    Deep learning methods (CNN) provide better accuracy.

    Parameters:
    -----------
    iq_data : array_like (complex)
        Complex IQ samples
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    mod_type : str
        Estimated modulation type
    features : dict
        Extracted features used for classification

    Methodology:
    ------------
    Uses statistical features:
    - Amplitude variance (for AM detection)
    - Phase variance (for PM/FM detection)
    - Constellation spread (for digital mods)
    """
    # Compute basic features
    amplitude = np.abs(iq_data)
    phase = np.angle(iq_data)

    # Unwrap phase to avoid discontinuities
    phase_unwrapped = np.unwrap(phase)

    # Compute statistics
    amp_mean = np.mean(amplitude)
    amp_std = np.std(amplitude)
    amp_var_coef = amp_std / (amp_mean + 1e-12)  # Coefficient of variation

    phase_diff = np.diff(phase_unwrapped)
    phase_std = np.std(phase_diff)

    # Store features
    features = {
        'amplitude_var_coef': amp_var_coef,
        'phase_std': phase_std,
        'mean_amplitude': amp_mean,
        'std_amplitude': amp_std
    }

    # Simple classification rules (heuristic)
    if amp_var_coef > 0.3:
        if phase_std < 0.5:
            mod_type = "AM (Amplitude Modulation)"
        else:
            mod_type = "QAM-like (Amplitude and Phase)"
    else:
        if phase_std > 1.0:
            mod_type = "FM/PM (Frequency/Phase Modulation)"
        else:
            mod_type = "PSK-like (Phase Shift Keying)"

    return mod_type, features


# Visualization functions
def plot_spectrogram_result(f, t, Sxx_db, title="Signal Spectrogram"):
    """Create spectrogram visualization."""
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, fftshift(f), fftshift(Sxx_db, axes=0), 
                   shading='gouraud', cmap='viridis')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    plt.title(title)
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.tight_layout()
    return plt.gcf()


def plot_psd(freqs, psd, center_freq, title="Power Spectral Density"):
    """Create PSD visualization."""
    plt.figure(figsize=(10, 6))
    psd_db = 10 * np.log10(psd + 1e-12)
    plt.plot(freqs/1e6, psd_db, linewidth=1.5)
    plt.axvline(center_freq/1e6, color='r', linestyle='--', 
                label=f'Center: {center_freq/1e6:.3f} MHz')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


def plot_constellation(iq_data, title="IQ Constellation"):
    """Create constellation diagram."""
    plt.figure(figsize=(8, 8))
    plt.scatter(iq_data.real, iq_data.imag, alpha=0.5, s=1)
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    return plt.gcf()


def plot_time_domain(time, signal_data, envelope=None, title="Time Domain Signal"):
    """Create time-domain visualization."""
    plt.figure(figsize=(12, 6))

    if np.iscomplexobj(signal_data):
        plt.subplot(2, 1, 1)
        plt.plot(time*1e6, signal_data.real, linewidth=0.5, label='I (Real)')
        plt.ylabel('Amplitude (I)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(time*1e6, signal_data.imag, linewidth=0.5, label='Q (Imag)', color='orange')
        plt.ylabel('Amplitude (Q)')
        plt.xlabel('Time (μs)')
        plt.grid(True, alpha=0.3)
        plt.legend()
    else:
        plt.plot(time*1e6, signal_data, linewidth=0.5)
        plt.ylabel('Amplitude')
        plt.xlabel('Time (μs)')
        plt.title(title)
        plt.grid(True, alpha=0.3)

    if envelope is not None:
        plt.subplot(2, 1, 1)
        plt.plot(time*1e6, envelope, 'r--', linewidth=1, label='Envelope', alpha=0.7)
        plt.legend()

    plt.tight_layout()
    return plt.gcf()
