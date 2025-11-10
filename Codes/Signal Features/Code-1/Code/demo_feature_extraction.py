"""
RF Signal Feature Extraction - Demonstration Script
====================================================
This script demonstrates all feature extraction functions with various signal types.

Generates synthetic test signals and extracts:
- Spectrogram
- Center frequency and bandwidth
- IQ/Waveform conversions
- Burst detection and duty cycle
- Modulation classification

Author: RF Signal Processing Pipeline
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Import our feature extraction module
sys.path.append(os.path.dirname(__file__))
from signal_features import *

def generate_test_signals(fs=1e6, duration=0.001):
    """Generate various test RF signals."""
    num_samples = int(fs * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    signals = {}
    
    # 1. QPSK Signal
    print("\nGenerating QPSK signal...")
    symbol_rate = 50e3
    symbols_per_bit = int(fs / symbol_rate)
    num_symbols = num_samples // symbols_per_bit
    
    phases = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], num_symbols)
    qpsk_symbols = np.exp(1j * phases)
    qpsk_signal = np.repeat(qpsk_symbols, symbols_per_bit)[:num_samples]
    
    fc = 100e3
    carrier = np.exp(1j * 2 * np.pi * fc * t)
    qpsk_signal = qpsk_signal * carrier
    
    noise = (np.random.randn(num_samples) + 1j*np.random.randn(num_samples)) * 0.1
    signals['qpsk'] = qpsk_signal + noise
    
    # 2. FM Signal
    print("Generating FM signal...")
    fc = 150e3
    fm = 5e3
    beta = 5
    
    modulating = np.sin(2 * np.pi * fm * t)
    phase = 2 * np.pi * fc * t + beta * np.sin(2 * np.pi * fm * t)
    fm_signal = np.exp(1j * phase)
    
    noise = (np.random.randn(num_samples) + 1j*np.random.randn(num_samples)) * 0.05
    signals['fm'] = fm_signal + noise
    
    # 3. Burst Signal
    print("Generating burst signal...")
    fc = 200e3
    carrier = np.exp(1j * 2 * np.pi * fc * t)
    
    burst_envelope = np.zeros(num_samples)
    burst_duration = 0.0002
    burst_period = 0.0005
    
    num_bursts = int(duration / burst_period)
    for i in range(num_bursts):
        start_idx = int(i * burst_period * fs)
        end_idx = int((i * burst_period + burst_duration) * fs)
        if end_idx < num_samples:
            burst_envelope[start_idx:end_idx] = 1.0
    
    burst_signal = carrier * burst_envelope
    noise = (np.random.randn(num_samples) + 1j*np.random.randn(num_samples)) * 0.05
    signals['burst'] = burst_signal + noise
    
    # 4. AM Signal
    print("Generating AM signal...")
    fc = 120e3
    fm = 3e3
    m = 0.5
    
    modulating = np.sin(2 * np.pi * fm * t)
    am_envelope = 1 + m * modulating
    carrier = np.exp(1j * 2 * np.pi * fc * t)
    am_signal = am_envelope * carrier
    
    noise = (np.random.randn(num_samples) + 1j*np.random.randn(num_samples)) * 0.05
    signals['am'] = am_signal + noise
    
    return signals, t, fs


def test_all_features(signal_name, iq_signal, t, fs, output_dir='test_results'):
    """
    Test all feature extraction functions on a given signal.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {signal_name.upper()}")
    print(f"{'='*60}")

    results = {}

    # 1. Spectrogram
    print("\n1. Computing spectrogram...")
    f, t_spec, Sxx_db = compute_spectrogram(iq_signal, fs, nperseg=256)
    results['spectrogram'] = (f, t_spec, Sxx_db)

    fig = plot_spectrogram_result(f, t_spec, Sxx_db, 
                                   title=f"Spectrogram - {signal_name.upper()}")
    plt.savefig(f'{output_dir}/{signal_name}_spectrogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {signal_name}_spectrogram.png")


    # 2. Center Frequency
    print("\n2. Estimating center frequency...")
    center_freq, psd, freqs = estimate_center_frequency(iq_signal, fs, method='peak')
    results['center_freq'] = center_freq

    print(f"   Center Frequency: {center_freq/1e3:.3f} kHz")

    fig = plot_psd(freqs, psd, center_freq, 
                   title=f"Power Spectral Density - {signal_name.upper()}")
    plt.savefig(f'{output_dir}/{signal_name}_psd.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {signal_name}_psd.png")


    # 3. Bandwidth
    print("\n3. Estimating bandwidth...")
    bandwidth, f_lower, f_upper = estimate_bandwidth(iq_signal, fs, threshold_db=-10)
    results['bandwidth'] = bandwidth
    results['f_lower'] = f_lower
    results['f_upper'] = f_upper

    print(f"   Bandwidth: {bandwidth/1e3:.3f} kHz")
    print(f"   Frequency Range: {f_lower/1e3:.3f} to {f_upper/1e3:.3f} kHz")


    # 4. Constellation Diagram
    print("\n4. Creating constellation diagram...")
    fig = plot_constellation(iq_signal, title=f"IQ Constellation - {signal_name.upper()}")
    plt.savefig(f'{output_dir}/{signal_name}_constellation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {signal_name}_constellation.png")


    # 5. Burst Detection
    print("\n5. Detecting bursts...")
    bursts, envelope = detect_bursts(iq_signal, fs, threshold_db=-10, min_duration=50e-6)
    results['bursts'] = bursts
    results['num_bursts'] = len(bursts)

    print(f"   Number of bursts detected: {len(bursts)}")
    if len(bursts) > 0:
        for i, (start, end, duration) in enumerate(bursts[:5]):  # Show first 5
            print(f"      Burst {i+1}: {duration*1e6:.1f} μs")


    # 6. Duty Cycle
    print("\n6. Computing duty cycle...")
    duty_cycle, active_time, total_time = compute_duty_cycle(bursts, len(iq_signal), fs)
    results['duty_cycle'] = duty_cycle

    print(f"   Duty Cycle: {duty_cycle*100:.2f}%")
    print(f"   Active Time: {active_time*1e3:.3f} ms")
    print(f"   Total Time: {total_time*1e3:.3f} ms")


    # 7. Time Domain Plot with Envelope
    print("\n7. Creating time-domain plot...")
    fig = plot_time_domain(t[:len(iq_signal)], iq_signal, envelope=envelope,
                           title=f"Time Domain - {signal_name.upper()}")
    plt.savefig(f'{output_dir}/{signal_name}_timedomain.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {signal_name}_timedomain.png")


    # 8. Modulation Classification
    print("\n8. Classifying modulation type...")
    mod_type, features = classify_modulation_simple(iq_signal, fs)
    results['modulation_type'] = mod_type
    results['mod_features'] = features

    print(f"   Estimated Modulation: {mod_type}")
    print(f"   Amplitude Variation Coefficient: {features['amplitude_var_coef']:.4f}")
    print(f"   Phase Std Dev: {features['phase_std']:.4f}")


    # 9. IQ to Waveform Conversion
    print("\n9. Testing IQ to Waveform conversion...")
    waveform, fs_real = iq_to_waveform(iq_signal[:1000], fs)  # Use subset for speed
    results['waveform_conversion'] = len(waveform)
    print(f"   Converted {len(iq_signal[:1000])} IQ samples to {len(waveform)} real samples")
    print(f"   Real signal sampling rate: {fs_real/1e6:.3f} MHz")


    # 10. Waveform to IQ Conversion (test round-trip)
    print("\n10. Testing Waveform to IQ conversion...")
    iq_reconstructed = waveform_to_iq(waveform, fs_real)
    results['iq_reconstruction'] = len(iq_reconstructed)
    print(f"   Converted {len(waveform)} real samples to {len(iq_reconstructed)} IQ samples")

    return results


def main():
    """Main execution function."""
    print("="*60)
    print("RF SIGNAL FEATURE EXTRACTION - DEMONSTRATION")
    print("="*60)

    # Create output directory
    output_dir = 'test_results'
    os.makedirs(output_dir, exist_ok=True)

    # Generate test signals
    print("\nGenerating test signals...")
    signals, t, fs = generate_test_signals(fs=1e6, duration=0.001)

    print(f"\nSampling Rate: {fs/1e6:.3f} MHz")
    print(f"Signal Duration: {len(t)/fs*1e3:.3f} ms")
    print(f"Number of Samples: {len(t)}")
    print(f"Signal Types: {list(signals.keys())}")

    # Process each signal type
    all_results = {}
    for signal_name, iq_signal in signals.items():
        results = test_all_features(signal_name, iq_signal, t, fs, output_dir)
        all_results[signal_name] = results

    # Generate summary report
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)

    summary_lines = []
    summary_lines.append("RF Signal Feature Extraction - Test Results Summary\n")
    summary_lines.append("="*60 + "\n\n")

    for signal_name, results in all_results.items():
        summary_lines.append(f"Signal: {signal_name.upper()}\n")
        summary_lines.append("-"*40 + "\n")
        summary_lines.append(f"  Center Frequency: {results['center_freq']/1e3:.3f} kHz\n")
        summary_lines.append(f"  Bandwidth: {results['bandwidth']/1e3:.3f} kHz\n")
        summary_lines.append(f"  Frequency Range: {results['f_lower']/1e3:.3f} to {results['f_upper']/1e3:.3f} kHz\n")
        summary_lines.append(f"  Number of Bursts: {results['num_bursts']}\n")
        summary_lines.append(f"  Duty Cycle: {results['duty_cycle']*100:.2f}%\n")
        summary_lines.append(f"  Modulation Type: {results['modulation_type']}\n")
        summary_lines.append("\n")

    summary_lines.append("\nGenerated Files:\n")
    summary_lines.append("-"*40 + "\n")
    for signal_name in all_results.keys():
        summary_lines.append(f"  {signal_name}_spectrogram.png\n")
        summary_lines.append(f"  {signal_name}_psd.png\n")
        summary_lines.append(f"  {signal_name}_constellation.png\n")
        summary_lines.append(f"  {signal_name}_timedomain.png\n")

    summary_text = ''.join(summary_lines)

    # Save summary
    with open(f'{output_dir}/summary.txt', 'w') as f:
        f.write(summary_text)

    print(summary_text)
    print(f"\nSummary saved to: {output_dir}/summary.txt")
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
