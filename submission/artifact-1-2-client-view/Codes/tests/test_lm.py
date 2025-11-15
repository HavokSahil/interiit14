from command.link_measurement_cmd import LinkMeasurementCommandBuilder
from control.controller import Controller

from db.station_db import StationDB

from logger import Logger, LogLevel
from time import sleep

def test_lm():

    Logger.set_current_log_level(LogLevel.ERROR)

    stationDB = StationDB()

    ctrl = Controller()
    ctrl.connect()

    sleep(1.0)

    ctrl.get_stations(stationDB)

    stations = stationDB.all()
    for station in stations:
        print(station)
        lm = LinkMeasurementCommandBuilder(station.mac)
        ctrl.req_link_measurement(lm)

    ctrl.disconnect()