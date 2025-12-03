from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math
from datatype import AccessPoint, Interferer
from model import PropagationModel
from utils import compute_distance


@dataclass
class SensingResult:
    """Stores sensing results for a single access point."""
    ap_id: int
    major_interferer_type: str  # BLE, Bluetooth, or Microwave
    confidence: float  # 0.0 to 1.0
    center_frequency: float  # in GHz
    duty_cycle: float  # 0.0 to 1.0
    bandwidth: float  # in MHz


class SensingAPI:
    """
    Sensing API for scanning interferers from each access point.
    
    For each access point, this API:
    1. Scans all interferers and computes their distances
    2. Identifies the most major (dominant) interferer
    3. Outputs interferer characteristics with confidence
    """
    
    def __init__(self, access_points: List[AccessPoint], 
                 interferers: List[Interferer], 
                 prop_model: PropagationModel):
        """
        Initialize the Sensing API.
        
        Args:
            access_points: List of access points in the network
            interferers: List of interferers in the environment
            prop_model: Propagation model for computing received power
        """
        self.access_points = access_points
        self.interferers = interferers
        self.prop_model = prop_model
    
    @staticmethod
    def channel_to_frequency(channel: int) -> float:
        """
        Convert WiFi channel number to center frequency in GHz.
        
        WiFi 2.4 GHz band:
        - Channel 1: 2.412 GHz
        - Channel spacing: 5 MHz (0.005 GHz)
        
        Args:
            channel: WiFi channel number
            
        Returns:
            Center frequency in GHz
        """
        # 2.4 GHz band: Channel 1 starts at 2.412 GHz, with 5 MHz spacing
        return 2.407 + (channel * 0.005)
    
    def scan_interferers(self, ap: AccessPoint) -> List[Tuple[Interferer, float, float]]:
        """
        Scan all interferers from the perspective of a given access point.
        
        For each interferer, computes:
        - Distance from the AP
        - Received power at the AP (in dBm)
        
        Args:
            ap: Access point performing the scan
            
        Returns:
            List of tuples: (interferer, distance, received_power_dbm)
        """
        results = []
        
        for interferer in self.interferers:
            # Compute distance from AP to interferer
            dist = compute_distance(ap.x, ap.y, interferer.x, interferer.y)
            
            # Compute received power at the AP from this interferer
            rx_power_dbm = self.prop_model.compute_received_power(
                interferer.tx_power, dist
            )
            
            results.append((interferer, dist, rx_power_dbm))
        
        return results
    
    def identify_major_interferer(self, 
                                  interferer_data: List[Tuple[Interferer, float, float]]
                                  ) -> Tuple[Optional[Interferer], float]:
        """
        Identify the major (dominant) interferer and calculate confidence.
        
        The major interferer is the one with the highest effective received power,
        where effective power accounts for duty cycle:
            effective_power = rx_power_dbm + 10*log10(duty_cycle)
        
        Confidence is calculated based on the power difference between the
        strongest and second-strongest interferers:
            - If only one interferer: confidence = 1.0
            - Otherwise: confidence = min(1.0, power_diff_db / 10.0)
              (10 dB difference → 100% confidence)
        
        Args:
            interferer_data: List of (interferer, distance, rx_power_dbm) tuples
            
        Returns:
            Tuple of (major_interferer, confidence)
            Returns (None, 0.0) if no interferers present
        """
        if not interferer_data:
            return (None, 0.0)
        
        # Calculate effective power for each interferer (accounting for duty cycle)
        effective_powers = []
        for interferer, dist, rx_power_dbm in interferer_data:
            # Scale power by duty cycle (in dB domain)
            if interferer.duty_cycle > 0:
                effective_power = rx_power_dbm + 10 * math.log10(interferer.duty_cycle)
            else:
                effective_power = float('-inf')
            
            effective_powers.append((interferer, effective_power))
        
        # Sort by effective power (descending)
        effective_powers.sort(key=lambda x: x[1], reverse=True)
        
        # Strongest interferer is the major one
        major_interferer = effective_powers[0][0]
        strongest_power = effective_powers[0][1]
        
        # Calculate confidence
        if len(effective_powers) == 1:
            # Only one interferer → maximum confidence
            confidence = 1.0
        else:
            # Multiple interferers → confidence based on power difference
            second_strongest_power = effective_powers[1][1]
            power_diff_db = strongest_power - second_strongest_power
            
            # 10 dB difference gives 100% confidence
            # Linear scaling: confidence = power_diff / 10.0
            confidence = min(1.0, max(0.0, power_diff_db / 10.0))
        
        return (major_interferer, confidence)
    
    def compute_sensing_results(self) -> Dict[int, SensingResult]:
        """
        Compute sensing results for all access points.
        
        For each AP:
        1. Scans all interferers and computes distances/received powers
        2. Identifies the major interferer with confidence score
        3. Creates a SensingResult with all required outputs
        
        Returns:
            Dictionary mapping AP ID to SensingResult
            Returns empty dict if no APs or no interferers
        """
        results = {}
        
        if not self.interferers:
            # No interferers to sense
            return results
        
        for ap in self.access_points:
            # Step 1: Scan all interferers from this AP
            interferer_data = self.scan_interferers(ap)
            
            # Step 2: Identify the major interferer
            major_interferer, confidence = self.identify_major_interferer(interferer_data)
            
            # Step 3: Create sensing result
            if major_interferer is not None:
                result = SensingResult(
                    ap_id=ap.id,
                    major_interferer_type=major_interferer.type,
                    confidence=confidence,
                    center_frequency=self.channel_to_frequency(major_interferer.channel),
                    duty_cycle=major_interferer.duty_cycle,
                    bandwidth=major_interferer.bandwidth
                )
                
                results[ap.id] = result
        
        return results
    
    def print_sensing_results(self, results: Dict[int, SensingResult]) -> None:
        """
        Pretty-print sensing results for debugging and visualization.
        
        Args:
            results: Dictionary of sensing results from compute_sensing_results()
        """
        if not results:
            print("No sensing results available (no interferers detected)")
            return
        
        print("\n" + "="*80)
        print("SENSING API RESULTS")
        print("="*80)
        
        for ap_id, result in sorted(results.items()):
            print(f"\nAccess Point {ap_id}:")
            print(f"  Major Interferer Type: {result.major_interferer_type}")
            print(f"  Confidence:            {result.confidence:.2f}")
            print(f"  Center Frequency:      {result.center_frequency:.3f} GHz")
            print(f"  Duty Cycle:            {result.duty_cycle:.2%}")
            print(f"  Bandwidth:             {result.bandwidth} MHz")
        
        print("\n" + "="*80)
