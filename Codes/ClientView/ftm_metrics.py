from ftm_model import *
from ftm_datatype import *
from ftm_utils import *
import networkx as nx

# Class to manage client metrics
class ClientMetricsManager:
    def __init__(self, clients: List[Client], access_points: List[AccessPoint], prop_model: PropagationModel, interferers: List[Interferer] = []):
        self.clients = clients
        self.access_points = access_points
        self.interferers = interferers
        self.model = prop_model

    def update(self) -> None:
        self.compute_sinr_all()
        self.compute_maxrate_all()
        self.compute_retry_rate_all()


    # Function to compute througput of the maximum throughput of the client based on
    # the SINR information (Shannon's Capacity Theorem)
    @staticmethod
    def compute_maxrate(client: Client, access_points: List[AccessPoint], prop_model: PropagationModel, interferers: List[Interferer] = []) -> None:
        """Compute the maxrate for a client"""
        ClientMetricsManager.compute_sinr(client, access_points, prop_model, interferers)
        serving_ap = access_points[client.associated_ap]
        # compute the maxrate based on the SINR
        if client.sinr_db == float('inf'):
            client.max_rate_mbps = 0.0
        else:
            sinr_linear = 10** (client.sinr_db / 10.0)
            client.max_rate_mbps = serving_ap.bandwidth * math.log2(1 + sinr_linear)

    # Function to compute maxrate for all the clients
    def compute_maxrate_all(self) -> None:
        for client in self.clients:
            ClientMetricsManager.compute_maxrate(client, self.access_points, self.model, self.interferers)

    # Function to compute the SINR of the client
    @staticmethod
    def compute_sinr(client: Client, access_points: List[AccessPoint], prop_model: PropagationModel, interferers: List[Interferer] = []) -> None:
        """Compute the SINR for a client"""
        if client.associated_ap is None:
            client.sinr_db = float('-inf')
            return
        
        serving_ap = access_points[client.associated_ap]
        # the distance to the AP is required for the SINR power calculation
        dist_to_ap = compute_distance(client.x, client.y, serving_ap.x, serving_ap.y) 
        signal_power_dbm = prop_model.compute_received_power(serving_ap.tx_power, dist_to_ap)
        client.rssi_dbm = signal_power_dbm
        signal_power_mw = dbm_to_mw(signal_power_dbm)
        
        # interference power is the sum of the power from all other APs
        interference_power_mw = 0.0
        for ap in access_points:
            # accumulate the AP powers except the power of the associated AP
            if ap.id == client.associated_ap:
                continue

            channel_overlap = compute_channel_overlap(ap.channel, serving_ap.channel, ap.bandwidth, serving_ap.bandwidth)
            if channel_overlap > 0.01:
                dist_to_interf = compute_distance(client.x, client.y, ap.x, ap.y)
                interf_pw_dbm = prop_model.compute_received_power(ap.tx_power, dist_to_interf)
                interf_pw_mw = dbm_to_mw(interf_pw_dbm)
                interference_power_mw += interf_pw_mw * channel_overlap

        # Add interference from external interferers
        for interferer in interferers:
            channel_overlap = compute_channel_overlap(interferer.channel, serving_ap.channel, interferer.bandwidth, serving_ap.bandwidth)
            if channel_overlap > 0.01:
                dist_to_interf = compute_distance(client.x, client.y, interferer.x, interferer.y)
                interf_pw_dbm = prop_model.compute_received_power(interferer.tx_power, dist_to_interf)
                interf_pw_mw = dbm_to_mw(interf_pw_dbm)
                # Scale by duty cycle
                interference_power_mw += interf_pw_mw * channel_overlap * interferer.duty_cycle
        
        # preset noise model, accounting for the thermal noise
        noise_power_mw = dbm_to_mw(serving_ap.noise_floor)
        
        sinr_linear = signal_power_mw / (interference_power_mw + noise_power_mw)
        client.sinr_db = 10 * math.log10(sinr_linear) if sinr_linear > 0 else float('-inf')
    
    # Function to compute the SINR for all the clients
    def compute_sinr_all(self) -> None:
        for client in self.clients:
            ClientMetricsManager.compute_sinr(client, self.access_points, self.model, self.interferers)

    @staticmethod
    def compute_retry_rate(client: Client, access_points: List[AccessPoint]) -> None:
        """
        Compute retry rate for a client based on SINR using a smoother logistic model.
        Low SINR → higher retries, but not explosively like the exponential model.
        """
        # No AP or invalid SINR → worst-case
        if client.associated_ap is None or client.sinr_db == float('-inf'):
            client.retry_rate = 100.0
            return
    
        # Tunable parameters
        mid_db = 10.0     # SINR where retry_rate ≈ 50%
        scale_db = 4.0    # Slope; higher → smoother transition
        max_retry = 100.0
        min_retry = 0.0   # keep at 0 unless you want a small floor
    
        # Logistic mapping: retry_rate = max_retry / (1 + exp((SINR - mid_db)/scale_db))
        exp_term = math.exp((client.sinr_db - mid_db) / scale_db)
        retry_rate = max_retry / (1.0 + exp_term)
    
        # Clamp
        client.retry_rate = max(min_retry, min(max_retry, retry_rate))

    
    # Function to compute retry rate for all clients
    def compute_retry_rate_all(self) -> None:
        for client in self.clients:
            ClientMetricsManager.compute_retry_rate(client, self.access_points)


# Class to manage the AP metrics
class APMetricsManager:
    def __init__(self, access_points: List[AccessPoint], clients: List[Client], prop_model: PropagationModel, interferers: List[Interferer] = []):
        self.access_points = access_points
        self.clients = clients
        self.interferers = interferers
        self.model = prop_model 

    def update(self) -> None:
        self.allocate_airtime()
        self.compute_ap_incoming_energy()
        self.compute_cca_busy_percentage()
        self.compute_roaming_rates()
        self.compute_p95_throughput()
        self.compute_p95_retry_rate()

    @staticmethod
    def get_clients(access_point: AccessPoint, clients: List[Client]) -> List[Client]:
        """Get clients associated with an AP"""
        return [client for client in clients if client.associated_ap == access_point.id]

    @staticmethod
    def ap_duty(access_points: AccessPoint) -> float:
        """Compute the duty cycle of an AP"""
        return access_points.total_allocated_throughput / access_points.max_throughput

    def compute_ap_incoming_energy(self) -> None:
        """
        Compute the incoming received signal energy at each AP from all other APs and clients,
        separated by channel (1, 6, 11).
        This represents the energy detected by the AP's additional sensing radio on each channel.
        Energy is measured in dBm.
        """
        # Channels to track
        CHANNELS = [1, 6, 11]
        
        # Reset incoming energy for all APs and all channels
        for ap in self.access_points:
            ap.inc_energy_ch1 = float('-inf')
            ap.inc_energy_ch6 = float('-inf')
            ap.inc_energy_ch11 = float('-inf')
        
        for rx_ap in self.access_points:
            # Track energy per channel in milliwatts
            channel_energy_mw = {ch: 0.0 for ch in CHANNELS}
            
            # 1. Energy from all other APs (downlink transmissions)
            for tx_ap in self.access_points:
                if tx_ap.id == rx_ap.id:
                    continue  # Skip self
                
                # Compute distance between APs
                dist = compute_distance(rx_ap.x, rx_ap.y, tx_ap.x, tx_ap.y)
                
                # Compute received power at rx_ap from tx_ap using propagation model
                received_power_dbm = self.model.compute_received_power(tx_ap.tx_power, dist)
                received_power_mw = dbm_to_mw(received_power_dbm)
                
                # Scale by duty cycle of the transmitting AP
                tx_duty = APMetricsManager.ap_duty(tx_ap)
                
                # Add energy to each channel bucket based on overlap with tx_ap's channel
                for sense_ch in CHANNELS:
                    # Create temporary AP objects to compute overlap
                    # We check if tx_ap's transmission overlaps with the sensing channel
                    overlap = compute_channel_overlap(
                        tx_ap.channel, sense_ch, 
                        tx_ap.bandwidth, 20.0  # Assume 20 MHz sensing bandwidth
                    )
                    if overlap > 0.01:  # Only add if there's meaningful overlap
                        channel_energy_mw[sense_ch] += received_power_mw * overlap * tx_duty
            
            # NOTE: [assume for now there is no uplink transmission from the client]
            # 2. Energy from all clients (uplink transmissions)
            # # Assuming clients transmit with a typical uplink power (e.g., 15 dBm)
            # client_tx_power_dbm = 5.0  # Typical client uplink power
            
            # for client in self.clients:
            #     if client.associated_ap is None:
            #         continue
                
            #     # Get the AP the client is associated with to know which channel it's using
            #     client_ap = next((ap for ap in self.access_points if ap.id == client.associated_ap), None)
            #     if client_ap is None:
            #         continue
                
            #     # Compute distance between AP and client
            #     dist = compute_distance(rx_ap.x, rx_ap.y, client.x, client.y)
                
            #     # Compute received power at rx_ap from client
            #     received_power_dbm = self.model.compute_received_power(client_tx_power_dbm, dist)
            #     received_power_mw = dbm_to_mw(received_power_dbm)
                
            #     # Add energy to each channel bucket based on overlap
            #     for sense_ch in CHANNELS:
            #         overlap = compute_channel_overlap(
            #             client_ap.channel, sense_ch,
            #             client_ap.bandwidth, 20.0
            #         )
            #         if overlap > 0.01:
            #             channel_energy_mw[sense_ch] += received_power_mw * overlap
            
            # 3. Energy from external interferers
            for interferer in self.interferers:
                dist = compute_distance(rx_ap.x, rx_ap.y, interferer.x, interferer.y)
                received_power_dbm = self.model.compute_received_power(interferer.tx_power, dist)
                received_power_mw = dbm_to_mw(received_power_dbm)
                
                for sense_ch in CHANNELS:
                    overlap = compute_channel_overlap(
                        interferer.channel, sense_ch,
                        interferer.bandwidth, 20.0
                    )
                    if overlap > 0.01:
                        channel_energy_mw[sense_ch] += received_power_mw * overlap * interferer.duty_cycle

            # Convert channel energies from milliwatts to dBm
            if channel_energy_mw[1] > 0:
                rx_ap.inc_energy_ch1 = 10 * math.log10(channel_energy_mw[1])
            if channel_energy_mw[6] > 0:
                rx_ap.inc_energy_ch6 = 10 * math.log10(channel_energy_mw[6])
            if channel_energy_mw[11] > 0:
                rx_ap.inc_energy_ch11 = 10 * math.log10(channel_energy_mw[11])
    
    def compute_cca_busy_percentage(self) -> None:
        """
        Compute the CCA busy percentage for each AP based on OBSS PD threshold violations.
        CCA busy percentage is calculated as the fraction of time (over a sliding window)
        where the energy on the operating channel exceeded the OBSS PD threshold.
        """
        for ap in self.access_points:
            # Get the incident energy for this AP's operating channel
            if ap.channel == 1:
                inc_energy = ap.inc_energy_ch1
            elif ap.channel == 6:
                inc_energy = ap.inc_energy_ch6
            elif ap.channel == 11:
                inc_energy = ap.inc_energy_ch11
            else:
                # For non-standard channels, no violation tracking
                ap.cca_busy_percentage = 0.0
                continue
            
            # Record violation status for this step
            violation = False
            if inc_energy != float('-inf') and inc_energy > ap.obss_pd_threshold:
                violation = True
            
            # Add to history (deque automatically drops oldest if at maxlen)
            ap.obss_pd_violation_history.append(violation)
            
            # Calculate CCA busy percentage from history
            if len(ap.obss_pd_violation_history) > 0:
                violations = sum(ap.obss_pd_violation_history)
                total_samples = len(ap.obss_pd_violation_history)
                ap.cca_busy_percentage = (violations / total_samples) * 100.0
            else:
                ap.cca_busy_percentage = 0.0
            
    def allocate_airtime(self) -> None:
        """Allocate airtime and throughput to each associated client."""
        for ap in self.access_points:
            clients = APMetricsManager.get_clients(ap, self.clients)

            # Compute each client's effective requested throughput
            # limited by both demand and PHY/max_rate
            requests = []
            for c in clients:
                req = min(c.demand_mbps, c.max_rate_mbps)
                requests.append(req)

            total_req = sum(requests)

            # Edge case: no demand or no usable max-rate
            if total_req <= 0:
                ap.total_allocated_throughput = 0.0
                for c in clients:
                    c.throughput_mbps = 0.0
                    c.airtime_fraction = 0.0
                continue

            # Capacity scaling: proportional allocation
            capacity = ap.max_throughput
            scale = min(1.0, capacity / total_req)

            ap.total_allocated_throughput = 0.0

            for c, req in zip(clients, requests):
                allocated = req * scale
                c.throughput_mbps = allocated
                ap.total_allocated_throughput += allocated

            for c in clients:
                c.airtime_fraction = c.throughput_mbps / ap.total_allocated_throughput if ap.total_allocated_throughput > 0 else 0.0

    def compute_roaming_rates(self) -> None:
        """
        Compute roam-in and roam-out rates for each AP based on current roaming events.
        Rates are averaged over a sliding window.
        """
        for ap in self.access_points:
            # Add current roaming counts to history
            ap.roam_in_history.append(ap.roam_in)
            ap.roam_out_history.append(ap.roam_out)
            
            # Calculate average rates from history
            if len(ap.roam_in_history) > 0:
                ap.roam_in_rate = sum(ap.roam_in_history) / len(ap.roam_in_history)
            else:
                ap.roam_in_rate = 0.0
            
            if len(ap.roam_out_history) > 0:
                ap.roam_out_rate = sum(ap.roam_out_history) / len(ap.roam_out_history)
            else:
                ap.roam_out_rate = 0.0
    
    def compute_p95_throughput(self) -> None:
        """
        Compute p95 throughput for each AP.
        p95 throughput means 95% of clients get at least this throughput.
        This is the 5th percentile of client throughputs.
        """
        for ap in self.access_points:
            clients = APMetricsManager.get_clients(ap, self.clients)
            
            if len(clients) == 0:
                ap.p95_throughput = 0.0
                continue
            
            # Collect all client throughputs
            throughputs = [c.throughput_mbps for c in clients]
            
            # Calculate 5th percentile (95% of clients get at least this)
            throughputs_sorted = sorted(throughputs)
            percentile_idx = int(len(throughputs_sorted) * 0.05)
            ap.p95_throughput = throughputs_sorted[percentile_idx]
    
    def compute_p95_retry_rate(self) -> None:
        """
        Compute p95 retry rate for each AP.
        This is the 95th percentile of client retry rates.
        """
        for ap in self.access_points:
            clients = APMetricsManager.get_clients(ap, self.clients)
            
            if len(clients) == 0:
                ap.p95_retry_rate = 0.0
                continue
            
            # Collect all client retry rates
            retry_rates = [c.retry_rate for c in clients]
            
            # Calculate 95th percentile
            retry_rates_sorted = sorted(retry_rates)
            percentile_idx = int(len(retry_rates_sorted) * 0.95)
            if percentile_idx >= len(retry_rates_sorted):
                percentile_idx = len(retry_rates_sorted) - 1
            ap.p95_retry_rate = retry_rates_sorted[percentile_idx]


class InterferenceGraphBuilder:
    """Builds interference graph from access points."""
    
    def __init__(self, propagation_model: PropagationModel, 
                 interference_threshold_dbm: float = -80.0,
                 rssi_min_dbm: float = -90.0,
                 rssi_max_dbm: float = -50.0):
        self.model = propagation_model
        self.threshold = interference_threshold_dbm
        self.rssi_min = rssi_min_dbm
        self.rssi_max = rssi_max_dbm
    
    def _compute_distance(self, ap1: AccessPoint, ap2: AccessPoint) -> float:
        """Euclidean distance between two access points."""
        return math.sqrt((ap1.x - ap2.x)**2 + (ap1.y - ap2.y)**2)
    
    def _compute_interference(self, tx_ap: AccessPoint, rx_ap: AccessPoint) -> float:
        """Compute interference power from tx_ap at rx_ap location."""
        distance = self._compute_distance(tx_ap, rx_ap)
        return self.model.compute_received_power(tx_ap.tx_power, distance)
    
    def _normalize_rssi(self, rssi_dbm: float) -> float:
        """Normalize RSSI from dBm to 0-1 scale."""
        # Clamp to range
        rssi_clamped = max(self.rssi_min, min(rssi_dbm, self.rssi_max))
        # Normalize to 0-1 (higher RSSI = stronger interference = higher value)
        return (rssi_clamped - self.rssi_min) / (self.rssi_max - self.rssi_min)
    
    def _compute_channel_overlap(self, ap1: AccessPoint, ap2: AccessPoint) -> float:
        return compute_channel_overlap(ap1.channel, ap2.channel, ap1.bandwidth, ap2.bandwidth)
    
    def _compute_interference_weight(self, tx_ap: AccessPoint, rx_ap: AccessPoint, 
                                     rssi_weight: float = 0.5, 
                                     load_weight: float = 0.2,
                                     channel_weight: float = 0.3) -> float:
        """Compute normalized interference weight (0-1) from RSSI, load, and channel overlap."""
        # Get raw RSSI
        rssi_dbm = self._compute_interference(tx_ap, rx_ap)
        
        # Normalize RSSI and load
        rssi_normalized = self._normalize_rssi(rssi_dbm)
        load_normalized = APMetricsManager.ap_duty(tx_ap)
        
        # Compute channel overlap
        overlap = self._compute_channel_overlap(tx_ap, rx_ap)
        
        # Weighted combination
        interference = overlap * (rssi_weight * rssi_normalized + 
                       load_weight * load_normalized)
        
        return interference
    
    def build_graph(self, access_points: List[AccessPoint]) -> nx.DiGraph:
        """Build directed interference graph."""
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for ap in access_points:
            G.add_node(ap.id, x=ap.x, y=ap.y, tx_power=ap.tx_power, 
                      load=APMetricsManager.ap_duty(ap), num_clients=len(ap.connected_clients),
                      channel=ap.channel, max_throughput=ap.max_throughput,
                      airtime_utilization=APMetricsManager.ap_duty(ap))
        
        # Add edges based on interference
        for i, tx_ap in enumerate(access_points):
            for j, rx_ap in enumerate(access_points):
                if i == j:
                    continue
                
                interference_power = self._compute_interference(tx_ap, rx_ap)
                
                if interference_power > self.threshold:
                    # Compute normalized interference weight (0-1)
                    weight = self._compute_interference_weight(tx_ap, rx_ap)
                    
                    G.add_edge(tx_ap.id, rx_ap.id, 
                             weight=weight,
                             interference_dbm=interference_power,
                             distance=self._compute_distance(tx_ap, rx_ap))        
        return G
