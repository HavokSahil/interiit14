"""
RFScannerController - A comprehensive Python controller for HackRF One
Supports controlling center frequency, sample rate, gains, and streaming
"""

import numpy as np
from python_hackrf import pyhackrf
import threading
import time
from typing import Callable, Optional
import logging

class RFScannerController:
    """
    A controller class for HackRF One SDR that provides methods to control
    streaming, frequency, sample rate, gains, and other parameters.
    """

    def __init__(self, device_index: int = 0):
        """
        Initialize the RFScannerController.

        Args:
            device_index: Index of the HackRF device to open (default: 0)
        """
        self.device_index = device_index
        self.sdr = None
        self.is_initialized = False
        self.is_streaming = False

        # Default parameters
        self._center_freq = 100e6  # 100 MHz
        self._sample_rate = 10e6   # 10 MHz
        self._lna_gain = 16        # 16 dB (0-40 dB in 8 dB steps)
        self._vga_gain = 20        # 20 dB (0-62 dB in 2 dB steps)
        self._amp_enabled = False  # RF amp off by default
        self._antenna_enabled = False  # Antenna power off
        self._baseband_filter = 7.5e6  # 7.5 MHz

        # Streaming control
        self._rx_callback_func = None
        self._streaming_thread = None
        self._stop_streaming_event = threading.Event()

        # Sample buffer for callback
        self.samples_buffer = []
        self.buffer_lock = threading.Lock()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('RFScannerController')

    def initialize(self) -> bool:
        """
        Initialize the HackRF device and library.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            if self.is_initialized:
                self.logger.warning("Device already initialized")
                return True

            pyhackrf.pyhackrf_init()
            self.sdr = pyhackrf.pyhackrf_open()

            # Apply default settings
            self.set_sample_rate(self._sample_rate)
            self.set_center_frequency(self._center_freq)
            self.set_lna_gain(self._lna_gain)
            self.set_vga_gain(self._vga_gain)
            self.set_amp_enable(self._amp_enabled)
            self.set_antenna_enable(self._antenna_enabled)
            self.set_baseband_filter(self._baseband_filter)

            self.is_initialized = True
            self.logger.info(f"HackRF initialized successfully (Device {self.device_index})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize HackRF: {e}")
            return False

    def close(self) -> bool:
        """
        Close the HackRF device and clean up.

        Returns:
            bool: True if closed successfully, False otherwise
        """
        try:
            if self.is_streaming:
                self.stop_stream()

            if self.sdr:
                self.sdr.pyhackrf_close()
                pyhackrf.pyhackrf_exit()

            self.is_initialized = False
            self.sdr = None
            self.logger.info("HackRF closed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to close HackRF: {e}")
            return False

    # Frequency control methods
    def set_center_frequency(self, freq_hz: float) -> bool:
        """
        Set the center frequency of the HackRF.

        Args:
            freq_hz: Center frequency in Hz (1 MHz to 6 GHz)

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized:
            self.logger.error("Device not initialized")
            return False

        try:
            if freq_hz < 1e6 or freq_hz > 6e9:
                self.logger.error(f"Frequency {freq_hz/1e6:.2f} MHz out of range (1-6000 MHz)")
                return False

            self.sdr.pyhackrf_set_freq(int(freq_hz))
            self._center_freq = freq_hz
            self.logger.info(f"Center frequency set to {freq_hz/1e6:.2f} MHz")
            return True

        except Exception as e:
            self.logger.error(f"Failed to set center frequency: {e}")
            return False

    def get_center_frequency(self) -> float:
        """Get the current center frequency in Hz."""
        return self._center_freq

    # Sample rate control methods
    def set_sample_rate(self, rate_hz: float) -> bool:
        """
        Set the sample rate (bandwidth) of the HackRF.

        Args:
            rate_hz: Sample rate in Hz (2 MHz to 20 MHz)

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized:
            self.logger.error("Device not initialized")
            return False

        try:
            if rate_hz < 2e6 or rate_hz > 20e6:
                self.logger.error(f"Sample rate {rate_hz/1e6:.2f} MHz out of range (2-20 MHz)")
                return False

            self.sdr.pyhackrf_set_sample_rate(rate_hz)
            self._sample_rate = rate_hz
            self.logger.info(f"Sample rate set to {rate_hz/1e6:.2f} MHz")
            return True

        except Exception as e:
            self.logger.error(f"Failed to set sample rate: {e}")
            return False

    def get_sample_rate(self) -> float:
        """Get the current sample rate in Hz."""
        return self._sample_rate

    # Gain control methods
    def set_lna_gain(self, gain_db: int) -> bool:
        """
        Set the LNA (Low Noise Amplifier) gain.

        Args:
            gain_db: LNA gain in dB (0, 8, 16, 24, 32, 40)

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized:
            self.logger.error("Device not initialized")
            return False

        try:
            # Round to nearest valid value (0-40 in 8 dB steps)
            gain_db = int(round(gain_db / 8.0) * 8)
            gain_db = max(0, min(40, gain_db))

            self.sdr.pyhackrf_set_lna_gain(gain_db)
            self._lna_gain = gain_db
            self.logger.info(f"LNA gain set to {gain_db} dB")
            return True

        except Exception as e:
            self.logger.error(f"Failed to set LNA gain: {e}")
            return False

    def get_lna_gain(self) -> int:
        """Get the current LNA gain in dB."""
        return self._lna_gain

    def set_vga_gain(self, gain_db: int) -> bool:
        """
        Set the VGA (Variable Gain Amplifier) gain.

        Args:
            gain_db: VGA gain in dB (0-62 in 2 dB steps)

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized:
            self.logger.error("Device not initialized")
            return False

        try:
            # Round to nearest valid value (0-62 in 2 dB steps)
            gain_db = int(round(gain_db / 2.0) * 2)
            gain_db = max(0, min(62, gain_db))

            self.sdr.pyhackrf_set_vga_gain(gain_db)
            self._vga_gain = gain_db
            self.logger.info(f"VGA gain set to {gain_db} dB")
            return True

        except Exception as e:
            self.logger.error(f"Failed to set VGA gain: {e}")
            return False

    def get_vga_gain(self) -> int:
        """Get the current VGA gain in dB."""
        return self._vga_gain

    # Amplifier control methods
    def set_amp_enable(self, enable: bool) -> bool:
        """
        Enable or disable the RF amplifier (adds 14 dB).

        Args:
            enable: True to enable, False to disable

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized:
            self.logger.error("Device not initialized")
            return False

        try:
            self.sdr.pyhackrf_set_amp_enable(enable)
            self._amp_enabled = enable
            self.logger.info(f"RF amplifier {'enabled' if enable else 'disabled'}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to set amp enable: {e}")
            return False

    def get_amp_enable(self) -> bool:
        """Get the current RF amplifier state."""
        return self._amp_enabled

    # Antenna control methods
    def set_antenna_enable(self, enable: bool) -> bool:
        """
        Enable or disable antenna port power supply.

        Args:
            enable: True to enable, False to disable

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized:
            self.logger.error("Device not initialized")
            return False

        try:
            self.sdr.pyhackrf_set_antenna_enable(enable)
            self._antenna_enabled = enable
            self.logger.info(f"Antenna power {'enabled' if enable else 'disabled'}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to set antenna enable: {e}")
            return False

    def get_antenna_enable(self) -> bool:
        """Get the current antenna power state."""
        return self._antenna_enabled

    # Baseband filter control
    def set_baseband_filter(self, bandwidth_hz: float) -> bool:
        """
        Set the baseband filter bandwidth.

        Args:
            bandwidth_hz: Filter bandwidth in Hz

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized:
            self.logger.error("Device not initialized")
            return False

        try:
            # Calculate the supported bandwidth
            allowed_bw = pyhackrf.pyhackrf_compute_baseband_filter_bw_round_down_lt(bandwidth_hz)
            self.sdr.pyhackrf_set_baseband_filter_bandwidth(allowed_bw)
            self._baseband_filter = allowed_bw
            self.logger.info(f"Baseband filter set to {allowed_bw/1e6:.2f} MHz")
            return True

        except Exception as e:
            self.logger.error(f"Failed to set baseband filter: {e}")
            return False

    def get_baseband_filter(self) -> float:
        """Get the current baseband filter bandwidth in Hz."""
        return self._baseband_filter

    # Streaming control methods
    def start_stream(self, callback: Optional[Callable] = None) -> bool:
        """
        Start receiving IQ samples from the HackRF.

        Args:
            callback: Optional callback function that receives (samples_array) as argument

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized:
            self.logger.error("Device not initialized")
            return False

        if self.is_streaming:
            self.logger.warning("Already streaming")
            return True

        try:
            self._rx_callback_func = callback
            self._stop_streaming_event.clear()

            # Define internal callback for HackRF
            def internal_rx_callback(device, buffer, buffer_length, valid_length):
                accepted = valid_length // 2
                accepted_samples = buffer[:valid_length].astype(np.int8)
                accepted_samples = accepted_samples[0::2] + 1j * accepted_samples[1::2]
                accepted_samples = accepted_samples / 128.0  # Normalize to -1 to +1

                # Store in buffer
                with self.buffer_lock:
                    self.samples_buffer.append(accepted_samples)

                # Call user callback if provided
                if self._rx_callback_func:
                    try:
                        self._rx_callback_func(accepted_samples)
                    except Exception as e:
                        self.logger.error(f"Error in user callback: {e}")

                # Check if we should stop
                if self._stop_streaming_event.is_set():
                    return 1  # Return non-zero to stop
                return 0  # Continue streaming

            self.sdr.set_rx_callback(internal_rx_callback)
            self.sdr.pyhackrf_start_rx()
            self.is_streaming = True
            self.logger.info("Started streaming")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start stream: {e}")
            return False

    def stop_stream(self) -> bool:
        """
        Stop receiving IQ samples from the HackRF.

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_streaming:
            self.logger.warning("Not currently streaming")
            return True

        try:
            self._stop_streaming_event.set()
            time.sleep(0.1)  # Give callback time to finish
            self.sdr.pyhackrf_stop_rx()
            self.is_streaming = False
            self.logger.info("Stopped streaming")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop stream: {e}")
            return False

    def get_streaming_status(self) -> bool:
        """Check if the device is currently streaming."""
        return self.is_streaming

    def get_samples(self, clear_buffer: bool = True) -> np.ndarray:
        """
        Get accumulated samples from the buffer.

        Args:
            clear_buffer: If True, clear the buffer after reading

        Returns:
            numpy array of complex samples
        """
        with self.buffer_lock:
            if len(self.samples_buffer) == 0:
                return np.array([], dtype=np.complex64)

            samples = np.concatenate(self.samples_buffer)

            if clear_buffer:
                self.samples_buffer = []

            return samples

    # Device information methods
    def get_device_info(self) -> dict:
        """
        Get information about the HackRF device.

        Returns:
            Dictionary containing device information
        """
        info = {
            'initialized': self.is_initialized,
            'streaming': self.is_streaming,
            'center_freq_mhz': self._center_freq / 1e6,
            'sample_rate_mhz': self._sample_rate / 1e6,
            'lna_gain_db': self._lna_gain,
            'vga_gain_db': self._vga_gain,
            'amp_enabled': self._amp_enabled,
            'antenna_enabled': self._antenna_enabled,
            'baseband_filter_mhz': self._baseband_filter / 1e6,
        }
        return info

    def print_device_info(self):
        """Print device information in a formatted way."""
        info = self.get_device_info()
        print("\n" + "="*50)
        print("HackRF Device Information")
        print("="*50)
        print(f"Initialized: {info['initialized']}")
        print(f"Streaming: {info['streaming']}")
        print(f"Center Frequency: {info['center_freq_mhz']:.2f} MHz")
        print(f"Sample Rate: {info['sample_rate_mhz']:.2f} MHz")
        print(f"LNA Gain: {info['lna_gain_db']} dB")
        print(f"VGA Gain: {info['vga_gain_db']} dB")
        print(f"RF Amp: {'Enabled' if info['amp_enabled'] else 'Disabled'}")
        print(f"Antenna Power: {'Enabled' if info['antenna_enabled'] else 'Disabled'}")
        print(f"Baseband Filter: {info['baseband_filter_mhz']:.2f} MHz")
        print("="*50 + "\n")

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
