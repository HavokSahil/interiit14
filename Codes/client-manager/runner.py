from math import pi
import threading
from time import sleep
from command.link_measurement_cmd import LinkMeasurementCommandBuilder
from command.request_beacon_cmd import MeasurementMode, OperatingClass, RequestBeaconCommandBuilder
from control.controller import Controller
from dashboard import bmrep_dashboard
from dashboard.nbrank_dashboard import NeighborRankingDashboard
from dashboard.qoe_dashboard import QoEDashboard
from db.nbrank_db import NeighborRankingDB
from db.qoe_db import QoEDB
from db.station_db import StationDB
from db.neighbor_db import NeighborDB
from db.bmrep_db import BeaconMeasurementDB
from db.bsstm_db import BSSTransitionResponseDB
from db.lmrep_db import LinkMeasurementDB
from dashboard.pipe_pool import PipePool
from dashboard.bmrep_dashboard import BeaconMeasurementDashboard
from dashboard.lmrep_dashboard import LinkMeasurementDashboard
from logger import Logger, LogLevel
from metrics.nbranking import NeighborRanking
from metrics.qoe import QoE
from protocol.rxmux import RxMux


def link_measurement_scheduler(ctrl, stationDB: StationDB, lmrepDB: LinkMeasurementDB, interval_sec):
    while True:
        for station in stationDB.all():
            if not lmrepDB.get(station.mac):
                lm_cmd = LinkMeasurementCommandBuilder(station.mac)
                ctrl.req_link_measurement(lm_cmd)
        sleep(interval_sec)


def beacon_measurement_scheduler(ctrl, stationDB: StationDB, neighborDB, bmrepDB: BeaconMeasurementDB, interval_sec):
    while True:
        for station in stationDB.all():
            if not bmrepDB.get(station.mac):
                bmrep = RequestBeaconCommandBuilder(dest_mac=station.mac)
                bmrep.set_measurement_params(
                    operating_class=OperatingClass.CLASS_2_4GHZ_20MHZ,
                    channel_number=0, # full active scan
                    randomization_interval=0,
                    measurement_duration=50,
                    measurement_mode=MeasurementMode.ACTIVE,
                    bssid="ff:ff:ff:ff:ff:ff"
                )
                ctrl.req_beacon_measurement(bmrep)
        sleep(interval_sec)


def qoe_scheduler(stdb: StationDB, stqoeDB: QoEDB, lmrepDB: LinkMeasurementDB,
                  interval_sec: float = 5.0):
    """
    Scheduler thread that computes and updates QoE for all stations periodically.
    """
    qoe_calc = QoE(stdb, stqoeDB, lmrepDB)

    while True:
        # Compute QoE for all stations
        qoe_calc.update()
        sleep(interval_sec)

def nbranking_scheduler(stdb: StationDB,
                        nrdb: NeighborRankingDB,
                        bmr: BeaconMeasurementDB,
                        ndb: NeighborDB,
                        interval_sec: float = 5.0):
    """
    Scheduler that recomputes neighbor ranking for all stations.
    Uses beacon measurements when available, otherwise falls back
    to static neighbor information.
    """

    ranker = NeighborRanking(nrdb=nrdb, bmr=bmr, ndb=ndb, stdb=stdb)

    while True:
        # Update the neighbor ranking table
        ranker.update()
        sleep(interval_sec)



def main():
    Logger.set_current_log_level(LogLevel.INFO)

    pipe_pool = PipePool()
    pipe_pool.create("bmrep", "bmrep.pipe")
    pipe_pool.create("lmrep", "lmrep.pipe")
    pipe_pool.create("nrank", "nrank.pipe")
    pipe_pool.create("stqoe", "stqoe.pipe")

    # Databases
    stationDB = StationDB()
    bmrepDB = BeaconMeasurementDB()
    bsstmDB = BSSTransitionResponseDB()
    lmrepDB = LinkMeasurementDB()
    neighborDB = NeighborDB()
    nbrankDB = NeighborRankingDB()
    stqoeDB = QoEDB()

    # Dashboards
    bmrep_dashboard = BeaconMeasurementDashboard(bmrepDB)
    lmrep_dashboard = LinkMeasurementDashboard(lmrepDB)
    nbrank_dashboard = NeighborRankingDashboard(nbrankDB)
    qoe_dashboard = QoEDashboard(stqoeDB)

    # Controller
    rxmux = RxMux()
    ctrl = Controller(rxmux=rxmux)
    ctrl.connect()

    # Initial station fetch
    ctrl.get_stations(stationDB)

    # Start scheduler threads
    lm_thread = threading.Thread(target=link_measurement_scheduler, args=(ctrl, stationDB, lmrepDB, 10), daemon=True)
    bm_thread = threading.Thread(target=beacon_measurement_scheduler, args=(ctrl, stationDB, neighborDB, bmrepDB, 15), daemon=True)
    qoe_thread = threading.Thread(target=qoe_scheduler,args=(stationDB, stqoeDB, lmrepDB),daemon=True)
    nbrank_thread = threading.Thread(target=nbranking_scheduler,args=(stationDB, nbrankDB, bmrepDB, neighborDB),daemon=True)

    lm_thread.start()
    bm_thread.start()
    qoe_thread.start()
    nbrank_thread.start()

    try:
        while True:
            ctrl.get_stations(stationDB)
            # Update dashboards
            bmrep_dashboard.show(pipe=pipe_pool.get("bmrep"), replace=True)
            lmrep_dashboard.show(pipe=pipe_pool.get("lmrep"), replace=True)
            qoe_dashboard.show(pipe=pipe_pool.get("stqoe"), replace=True)
            nbrank_dashboard.show(pipe=pipe_pool.get("nrank"), replace=True)        
            sleep(0.5)

    except KeyboardInterrupt:
        print("Exiting...")

    pipe_pool.destroy()
    ctrl.disconnect()


if __name__ == "__main__":
    main()
