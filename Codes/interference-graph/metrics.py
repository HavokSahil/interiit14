from model import *
from datatype import *
from utils import *
import networkx as nx

# Class to manage client metrics
class ClientMetricsManager:
    def __init__(self, clients: List[Client], access_points: List[AccessPoint], prop_model: PropagationModel):
        self.clients = clients
        self.access_points = access_points
        self.model = prop_model

    def update(self) -> None:
        self.compute_sinr_all()
        self.compute_maxrate_all()

    # Function to compute througput of the maximum throughput of the client based on
    # the SINR information (Shannon's Capacity Theorem)
    @staticmethod
    def compute_maxrate(client: Client, access_points: List[AccessPoint], prop_model: PropagationModel) -> None:
        """Compute the maxrate for a client"""
        ClientMetricsManager.compute_sinr(client, access_points, prop_model)
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
            ClientMetricsManager.compute_maxrate(client, self.access_points, self.model)

    # Function to compute the SINR of the client
    @staticmethod
    def compute_sinr(client: Client, access_points: List[AccessPoint], prop_model: PropagationModel) -> None:
        """Compute the SINR for a client"""
        if client.associated_ap is None:
            client.sinr_db = float('-inf')
            return
        
        serving_ap = access_points[client.associated_ap]
        # the distance to the AP is required for the SINR power calculation
        dist_to_ap = compute_distance(client.x, client.y, serving_ap.x, serving_ap.y) 
        signal_power_dbm = prop_model.compute_received_power(serving_ap.tx_power, dist_to_ap)
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
        
        # preset noise model, accounting for the thermal noise
        noise_power_mw = dbm_to_mw(serving_ap.noise_floor)
        
        sinr_linear = signal_power_mw / (interference_power_mw + noise_power_mw)
        client.sinr_db = 10 * math.log10(sinr_linear) if sinr_linear > 0 else float('-inf')
    
    # Function to compute the SINR for all the clients
    def compute_sinr_all(self) -> None:
        for client in self.clients:
            ClientMetricsManager.compute_sinr(client, self.access_points, self.model)


# Class to manage the AP metrics
class APMetricsManager:
    def __init__(self, access_points: List[AccessPoint], clients: List[Client], prop_model: PropagationModel):
        self.access_points = access_points
        self.clients = clients
        self.model = prop_model 

    def update(self) -> None:
        self.allocate_airtime()
        self.compute_ap_incoming_energy()

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
        Compute the incoming received signal energy at each AP from all other APs and clients.
        This represents the total energy detected by the AP's additional sensing radio.
        Energy is measured in dBm.
        """
        # Reset incoming energy for all APs
        for ap in self.access_points:
            ap.inc_energy = float('-inf')  # -inf dBm means no energy
        
        for rx_ap in self.access_points:
            total_energy_mw = 0.0
            
            # 1. Energy from all other APs (downlink transmissions)
            for tx_ap in self.access_points:
                if tx_ap.id == rx_ap.id:
                    continue  # Skip self
                
                # Compute distance between APs
                dist = compute_distance(rx_ap.x, rx_ap.y, tx_ap.x, tx_ap.y)
                
                # Compute received power at rx_ap from tx_ap using propagation model
                received_power_dbm = self.model.compute_received_power(tx_ap.tx_power, dist)
                
                # Convert to milliwatts and accumulate
                received_power_mw = dbm_to_mw(received_power_dbm)
                total_energy_mw += received_power_mw
            
            # 2. Energy from all clients (uplink transmissions)
            # Assuming clients transmit with a typical uplink power (e.g., 15 dBm)
            # You can make this configurable or add a tx_power field to Client dataclass
            client_tx_power_dbm = 15.0  # Typical client uplink power
            
            for client in self.clients:
                # Compute distance between AP and client
                dist = compute_distance(rx_ap.x, rx_ap.y, client.x, client.y)
                
                # Compute received power at rx_ap from client
                received_power_dbm = self.model.compute_received_power(client_tx_power_dbm, dist)
                
                # Convert to milliwatts and accumulate
                received_power_mw = dbm_to_mw(received_power_dbm)
                total_energy_mw += received_power_mw
            
            # Convert total energy from milliwatts to dBm
            if total_energy_mw > 0:
                rx_ap.inc_energy = 10 * math.log10(total_energy_mw)
            else:
                rx_ap.inc_energy = float('-inf')  # No energy detected
            
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
