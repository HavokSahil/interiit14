from ast import List
from command.bss_tm_cmd import BssTmRequestBuilder, ReqMode
from control.controller import Controller
from db.bsstm_db import BSSTransitionResponseDB
from db.nbrank_db import NeighborRankingDB
from db.neighbor_db import NeighborDB
from db.qoe_db import QoEDB
from db.station_db import StationDB
from model.neighbor import Neighbor
from model.station import Station


from logger import Logger

QOE_TRANSITION_THRESHOLD = 0.55

class TransitionManagementEngine:
    def __init__(self, ctrl: Controller):
        self.stdb = StationDB()
        self.nbdb = NeighborDB()
        self.btdb = BSSTransitionResponseDB()
        self.qoedb = QoEDB()
        self.rndb = NeighborRankingDB()
        self.ctrl = ctrl

    @staticmethod
    def qoe_quality_test(qoe: float) -> bool:
        Logger.log_info(f"Checking QoE quality: value={qoe}")
        return qoe > QOE_TRANSITION_THRESHOLD

    def _run_sta(self, station: Station):
        mac = station.mac
        Logger.log_info(f"Processing station: {mac}")
        qoe = self.qoedb.get(mac)
        Logger.log_info(f"Retrieved QoE for {mac}: {qoe}")
        if not qoe:
            Logger.log_info(f"No QoE available for {mac}; skipping BSS TM decision")
            return
        if TransitionManagementEngine.qoe_quality_test(qoe):
            Logger.log_info(f"QoE for {mac} is good enough ({qoe:.3f}); no transition management needed")
            return # NOTE: the qoe is good enough, keep serving
        
        Logger.log_info(f"QoE for {mac} is poor ({qoe:.3f}); considering transition")
        # get the ranked neighbors for the station
        nbs: List[Neighbor] = self.rndb.get_ranking(mac)
        Logger.log_info(f"Ranked neighbors for {mac}: count={len(nbs)}")
        # Build the BSS Transition command
        bsstmreq = BssTmRequestBuilder(
            sta_addr=mac,
            req_mode=ReqMode.PREFERRED_CAND_LIST_INCLUDED,
            validity_interval=1000,
            neighbors=nbs,
        )
        Logger.log_info(f"Constructed BSS TM Request for {mac}: {bsstmreq}")
        self.ctrl.req_bss_tm(bsstmreq)
        Logger.log_info(f"BSS TM Request sent for {mac}")

    def run(self):
        stations = self.stdb.all()
        Logger.log_info(f"Starting TM engine run for {len(stations)} stations")
        for station in stations:
            self._run_sta(station)
        Logger.log_info("TM engine run complete")
