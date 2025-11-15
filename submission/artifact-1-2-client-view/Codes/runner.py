from math import pi
import threading
from time import sleep
from api.stateapi import StateAPI
from command.bss_tm_cmd import BssTmRequestBuilder
from command.link_measurement_cmd import LinkMeasurementCommandBuilder
from command.request_beacon_cmd import MeasurementMode, OperatingClass, RequestBeaconCommandBuilder
from control.controller import Controller
from dashboard.bsstm_dashboard import BSSTransitionResponseDashboard
from dashboard.nbrank_dashboard import NeighborRankingDashboard
from dashboard.qoe_dashboard import QoEDashboard
from dashboard.socket_pool import SocketPool
from db.nbrank_db import NeighborRankingDB
from db.qoe_db import QoEDB
from db.station_db import StationDB
from db.neighbor_db import NeighborDB
from db.bmrep_db import BeaconMeasurementDB
from db.bsstm_db import BSSTransitionResponseDB
from db.lmrep_db import LinkMeasurementDB
from dashboard.bmrep_dashboard import BeaconMeasurementDashboard
from dashboard.lmrep_dashboard import LinkMeasurementDashboard
from dashboard.station_dashboard import StationDashboard
from logger import Logger, LogLevel
from metrics.nbranking import NeighborRanking
from metrics.qoe import QoE
from metrics.tm_engine import TransitionManagementEngine
from protocol.rxmux import RxMux
from store.routine import Routine

def link_measurement_scheduler(ctrl, stationDB: StationDB, lmrepDB: LinkMeasurementDB, interval_sec):
    while True:
        Logger.log_info(f"[Scheduler] Running link_measurement_scheduler")
        for station in stationDB.all():
            Logger.log_info(f"[LinkMeasurement] Checking station {station.mac}")
            if not lmrepDB.get(station.mac):
                Logger.log_info(f"[LinkMeasurement] Sending measurement command to {station.mac}")
                lm_cmd = LinkMeasurementCommandBuilder(station.mac)
                ctrl.req_link_measurement(lm_cmd)
        sleep(interval_sec)


def beacon_measurement_scheduler(ctrl, stationDB: StationDB, neighborDB, bmrepDB: BeaconMeasurementDB, interval_sec):
    while True:
        Logger.log_info(f"[Scheduler] Running beacon_measurement_scheduler")
        for station in stationDB.all():
            Logger.log_info(f"[BeaconMeasurement] Checking station {station.mac}")
            if not bmrepDB.get(station.mac):
                Logger.log_info(f"[BeaconMeasurement] Sending request beacon command to {station.mac}")
                bmrep = RequestBeaconCommandBuilder(dest_mac=station.mac)
                bmrep.set_measurement_params(
                    operating_class=OperatingClass.CLASS_2_4GHZ_20MHZ,
                    channel_number=0,
                    randomization_interval=0,
                    measurement_duration=50,
                    measurement_mode=MeasurementMode.ACTIVE,
                    bssid="ff:ff:ff:ff:ff:ff"
                )
                ctrl.req_beacon_measurement(bmrep)
        sleep(interval_sec)


def qoe_scheduler(qoe_calc: QoE, interval_sec: float = 5.0):
    while True:
        Logger.log_info(f"[Scheduler] Running qoe_scheduler")
        qoe_calc.update()
        sleep(interval_sec)


def nbranking_scheduler(stdb: StationDB,
                        nrdb: NeighborRankingDB,
                        bmr: BeaconMeasurementDB,
                        ndb: NeighborDB,
                        interval_sec: float = 5.0):
    ranker = NeighborRanking(nrdb=nrdb, bmr=bmr, ndb=ndb, stdb=stdb)
    while True:
        Logger.log_info(f"[Scheduler] Running nbranking_scheduler")
        ranker.update()
        sleep(interval_sec)


def bss_tm_scheduler(controller: Controller,
                    interval_sec: float = 5.0):
    bsstmreq = TransitionManagementEngine(controller)
    while True:
        Logger.log_info(f"[BssTMReq] Running bss_tm_scheduler")
        bsstmreq.run()
        sleep(interval_sec)


def accept_thread(pool: SocketPool, name: str):
    """Accept incoming connections and track them."""
    Logger.log_info(f"[SocketPool] Starting accept thread for {name}")
    while True:
        try:
            client = pool.accept(name)
            Logger.log_info(f"[SocketPool] Client connected to {name}: {client.name}")
        except Exception as e:
            Logger.log_info(f"[SocketPool] Accept error on {name}: {e}")
            sleep(1)  # Brief pause before retrying


def server_thread():
    stateapi = StateAPI()
    stateapi.serve()

def main():

    Logger.set_current_log_level(LogLevel.INFO)

    # --- Socket Pool instead of PipePool ---
    sock_pool = SocketPool()
    sock_pool.create("bmrep", "bmrep.sock", mode="server")
    sock_pool.create("lmrep", "lmrep.sock", mode="server")
    sock_pool.create("nrank", "nrank.sock", mode="server")
    sock_pool.create("stqoe", "stqoe.sock", mode="server")
    sock_pool.create("statn", "statn.sock", mode="server")
    sock_pool.create("bsstm", "bsstm.sock", mode="server")

    # Databases
    stationDB = StationDB()
    bmrepDB = BeaconMeasurementDB()
    bsstmDB = BSSTransitionResponseDB()
    lmrepDB = LinkMeasurementDB()
    neighborDB = NeighborDB()
    nbrankDB = NeighborRankingDB()
    qoeDB = QoEDB()

    # QoE calculator (needs to be created before dashboard)
    qoe_calc = QoE()
    
    # Dashboards
    bmrep_dashboard = BeaconMeasurementDashboard(bmrepDB)
    lmrep_dashboard = LinkMeasurementDashboard(lmrepDB)
    nbrank_dashboard = NeighborRankingDashboard(nbrankDB)
    qoe_dashboard = QoEDashboard(qoeDB, qoe_engine=qoe_calc)  # Pass engine for rich features
    station_dashboard = StationDashboard(stationDB)
    bss_dashboard = BSSTransitionResponseDashboard(bsstmDB)

    # Controller
    rxmux = RxMux()
    ctrl = Controller(rxmux=rxmux)
    ctrl.connect()

    # Initial station fetch
    ctrl.get_stations(stationDB)

    # Start scheduler threads
    lm_thread = threading.Thread(
        target=link_measurement_scheduler,
        args=(ctrl, stationDB, lmrepDB, 10),
        daemon=True
    )
    bm_thread = threading.Thread(
        target=beacon_measurement_scheduler,
        args=(ctrl, stationDB, neighborDB, bmrepDB, 15),
        daemon=True
    )
    qoe_thread = threading.Thread(
        target=qoe_scheduler,
        args=(qoe_calc, 2.0),
        daemon=True
    )
    nbrank_thread = threading.Thread(
        target=nbranking_scheduler,
        args=(stationDB, nbrankDB, bmrepDB, neighborDB),
        daemon=True
    )
    stateapi_thread = threading.Thread(
        target=server_thread,
        daemon=True
    )

    bss_tm_thread = threading.Thread(
        target=bss_tm_scheduler,
        args=(ctrl,),
        daemon=True
    )

    lm_thread.start()
    bm_thread.start()
    qoe_thread.start()
    nbrank_thread.start()
    stateapi_thread.start()
    bss_tm_thread.start()

    # Start accept threads for each server socket
    for name in ["bmrep", "lmrep", "nrank", "stqoe", "statn", "bsstm"]:
        threading.Thread(
            target=accept_thread, 
            args=(sock_pool, name), 
            daemon=True
    ).start()

    # Save all databases every 30 seconds, organized by date
    routine = Routine(
        output_dir="export",
        batch_interval=30.0,
        retention_days=7,   # automatically delete snapshots older than 7 days
        organize_by="hour"  # "date", "hour", or "flat"
    )

    # Maintain station_dashboard's instance inside the loop to update its db reference
    while True:
        ctrl.get_stations(stdb=stationDB)
        
        # Forcibly refresh the StationDashboard with updated StationDB
        station_dashboard.db = stationDB

        # Update each dashboard and send to connected clients
        for dash, name in [
            (bmrep_dashboard, "bmrep"),
            (lmrep_dashboard, "lmrep"),
            (qoe_dashboard, "stqoe"),
            (nbrank_dashboard, "nrank"),
            (station_dashboard, "statn"),
            (bss_dashboard, "bsstm")
        ]:
            clients = sock_pool.get_clients(name)
            if clients:
                Logger.log_info(f"[Dashboard] Updating {name} dashboard for {len(clients)} client(s)")
                for client in clients:
                    if isinstance(dash, QoEDashboard):
                        dash.show(pipe=client.conn, replace=True, mode="full")
                    elif isinstance(dash, StationDashboard):
                        # Always re-create dashboard output so it fetches latest data
                        dash.db = stationDB
                        dash.show(pipe_out=client.conn, replace=True)
                    else:
                        dash.show(pipe=client.conn, replace=True)

        sleep(0.5)

    # Cleanup (won't be reached)
    routine.stop()
    sock_pool.destroy()
    ctrl.disconnect()


if __name__ == "__main__":
    main()