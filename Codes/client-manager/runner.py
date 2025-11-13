from command.bss_tm_cmd import BssTmRequestBuilder, ReqMode
from command.link_measurement_cmd import LinkMeasurementCommandBuilder
from command.request_beacon_cmd import MeasurementMode, OperatingClass, RequestBeaconCommandBuilder
from control.controller import Controller

from dashboard import bmrep_dashboard
from db.station_db import StationDB
from db.neighbor_db import NeighborDB
from db.bmrep_db import BeaconMeasurementDB
from db.bsstm_db import BSSTransitionResponseDB
from db.lmrep_db import LinkMeasurementDB

from dashboard.pipe_pool import PipePool
from dashboard.station_dashboard import StationDashboard
from dashboard.bmrep_dashboard import BeaconMeasurementDashboard
from dashboard.lmrep_dashboard import LinkMeasurementDashboard
from dashboard.bsstm_dashboard import BSSTransitionResponseDashboard


from logger import Logger, LogLevel
from time import sleep

from model import measurement, neighbor
from protocol.rxmux import RxMux

def main():

    Logger.set_current_log_level(LogLevel.INFO)

    pipe_pool = PipePool()
    pipe_pool.create("bmrep", "bmrep.pipe")
    pipe_pool.create("bsstm", "bsstm.pipe")
    pipe_pool.create("lmrep", "lmrep.pipe")

    stationDB = StationDB()
    bmrepDB = BeaconMeasurementDB()
    bsstmDB = BSSTransitionResponseDB()
    lmrepDB = LinkMeasurementDB()
    neighborDB = NeighborDB()

    bmrep_dashboard = BeaconMeasurementDashboard(bmrepDB)
    lmrep_dashboard = LinkMeasurementDashboard(lmrepDB)
    bsstm_dashboard = BSSTransitionResponseDashboard(bsstmDB)

    rxmux = RxMux()
    ctrl = Controller(rxmux=rxmux)
    ctrl.connect()

    ctrl.get_stations(stationDB)

    flag = True

    while True:
        ctrl.get_stations(stationDB)
        if stationDB.count() > 0 and flag:
            for station in stationDB.all():
                # NOTE: send the link measurement report to each station
                #lm = LinkMeasurementCommandBuilder(station.mac)
                #ctrl.req_link_measurement(lm)

                #bmrep = RequestBeaconCommandBuilder(dest_mac=station.mac)
                # bmrep.set_measurement_params(
                #     operating_class=OperatingClass.CLASS_2_4GHZ_20MHZ,
                #     channel_number=1,
                #     randomization_interval=0,
                #     measurement_duration=50,
                #     measurement_mode=MeasurementMode.ACTIVE,
                #     bssid="ff:ff:ff:ff:ff:ff"
                # )

                #ctrl.req_beacon_measurement(bmrep)
                bsstm = BssTmRequestBuilder(
                    sta_addr=station.mac,
                    req_mode=ReqMode.PREFERRED_CAND_LIST_INCLUDED,
                    disassoc_timer=50,
                    validity_interval=100,
                    neighbors=neighborDB.all_for_sta(station)  
                )
                ctrl.req_bss_tm(bsstm)
            flag = False

        bmrep_dashboard.show(pipe=pipe_pool.get("bmrep"), replace=True)
        bsstm_dashboard.show(pipe=pipe_pool.get("bsstm"), replace=True)            
        lmrep_dashboard.show(pipe=pipe_pool.get("lmrep"), replace=True)            
        sleep(0.5)

    pipe_pool.destroy()
    ctrl.disconnect()

    


if __name__ == "__main__":
    main()
    print("exitting...")