from model import *
from datatype import *
from utils import *

class ClientAssociationManager:
    """Manages client-AP associations with SINR and airtime-fair throughput allocation."""
    
    def __init__(self, access_points: List[AccessPoint], clients: List[Client], propagation_model: PropagationModel, noise_floor_dbm: float = -95.0):
        self.access_points = access_points
        self.clients = clients
        self.model = propagation_model
        self.noise_floor = noise_floor_dbm

    @staticmethod
    def get_nearest_ap(client: Client, ap_list: List[AccessPoint]) -> AccessPoint:
        min_dist = float('inf')
        nearest_ap = None
        for ap in ap_list:
            dist = compute_distance(client.x, client.y, ap.x, ap.y)
            if dist < min_dist:
                min_dist = dist
                nearest_ap = ap
        return nearest_ap
    

    def voronoi_association(self) -> List[int]:
        """Initial association based on Voronoi diagram (nearest AP)."""
        roam_list = [0 for _ in range(len(self.clients))]

        for ap in self.access_points:
            ap.roam_in = 0
            ap.roam_out = 0

        ap_dict = {ap.id: ap for ap in self.access_points}

        for client in self.clients:
            nearest_ap = self.get_nearest_ap(client, self.access_points)
            if client.associated_ap != nearest_ap.id:
                client.last_assoc_ap = client.associated_ap
                roam_list[client.id] = 1

                # update roam_in and roam_out for the APs
                if client.associated_ap is not None:
                    ap_dict[client.associated_ap].roam_out += 1
                ap_dict[nearest_ap.id].roam_in += 1

            client.associated_ap = nearest_ap.id
            
        # update the associations on the AP side
        for ap in self.access_points:
            ap.connected_clients = []

        for client in self.clients:
            if client.associated_ap is not None:
                ap = ap_dict[client.associated_ap]
                ap.connected_clients.append(client.id)

        return roam_list
        
    def signal_strength_association(self) -> List[int]:
        """Associate clients to AP with best signal strength."""
        roam_list = [0 for _ in range(len(self.clients))]
        ap_dict = {ap.id: ap for ap in self.access_points}

        # reset roam_in and roam_out for all APs
        for ap in self.access_points:
            ap.roam_in = 0
            ap.roam_out = 0
        
        for client in self.clients:
            best_power = float('-inf')
            best_ap = None
            
            for ap in self.access_points:
                dist = compute_distance(client.x, client.y, ap.x, ap.y)
                rx_power = self.model.compute_received_power(ap.tx_power, dist)
                
                # Penalize APs with high CCA busy percentage
                # Effective power = Rx Power - Penalty
                # Penalty increases with CCA busy percentage
                cca_penalty = 0.1 * ap.cca_busy_percentage  # 0.1 dB penalty per 1% busy
                effective_power = rx_power - cca_penalty
                
                if effective_power > best_power:
                    best_power = effective_power
                    best_ap = ap.id
            
            if client.associated_ap != best_ap:
                client.last_assoc_ap = client.associated_ap
                roam_list[client.id] = 1

                # update roam_in and roam_out for the APs
                if client.associated_ap is not None:
                    ap_dict[client.associated_ap].roam_out += 1
                ap_dict[best_ap].roam_in += 1

            client.associated_ap = best_ap

        # update the associations on the AP side
        for ap in self.access_points:
            ap.connected_clients = []

        ap_dict = {ap.id: ap for ap in self.access_points}

        for client in self.clients:
            if client.associated_ap is not None:
                ap = ap_dict[client.associated_ap]
                ap.connected_clients.append(client.id)

        return roam_list