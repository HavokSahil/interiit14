from typing import Optional, List, Tuple, Dict
from datatype import AccessPoint, Client

class RRMEngine:
    def __init__(self) -> None:
        self.rrm_enabled = False
        # TODO: remote this Dummy Optional[int] later
        self.policy_engine: Optional[int] = None
        self.config_engine: Optional[int] = None # for changing the configuration of the APS
        self.slow_loop_engine: Optional[int] = None
        self.fast_loop_engine: Optional[int] = None

        # Stored COnfigurations
        self.slo_catalog: Optional[int] = None

        # Stored Entities
        self.aps: Dict[int, AccessPoint] = dict()
        self.stas: Dict[int, Client] = dict()


        self.sensing_api: Optional[int] = None
        self.client_view_api: Optional[int] = None
        


