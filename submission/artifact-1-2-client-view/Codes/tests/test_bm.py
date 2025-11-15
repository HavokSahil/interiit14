from command.request_beacon_cmd import *
from control.controller import Controller

from db.station_db import StationDB

from logger import Logger, LogLevel
from time import sleep

def test_rbm():

    Logger.set_current_log_level(LogLevel.INFO)

    stationDB = StationDB()

    ctrl = Controller()
    ctrl.connect()

    sleep(1.0)

    ctrl.get_stations(stationDB)

    bm = RequestBeaconCommandBuilder("")
    bm.set_measurement_params(
        OperatingClass.CLASS_2_4GHZ_20MHZ,
        1,
        0,
        50,
        MeasurementMode.ACTIVE, "ff:ff:ff:ff:ff:ff")

    stations = stationDB.all()
    for station in stations:
        bm.dest_mac = station.mac
        ctrl.req_beacon_measurement(bm)

    ctrl.disconnect()