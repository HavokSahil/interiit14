from command.bss_tm_cmd import *
from control.controller import Controller

from db.station_db import StationDB

from logger import Logger, LogLevel
from time import sleep

def test_bss_tm():

    Logger.set_current_log_level(LogLevel.INFO)

    stationDB = StationDB()

    ctrl = Controller()
    ctrl.connect()

    sleep(1.0)

    ctrl.get_stations(stationDB)

    req_mode =ReqMode.ABRUPT_TRANSITION

    stations = stationDB.all()
    for station in stations:
        bss_tm = BssTmRequestBuilder(station.mac, req_mode=req_mode, disassoc_timer=50)
        ctrl.req_bss_tm(bss_tm)

    ctrl.disconnect()