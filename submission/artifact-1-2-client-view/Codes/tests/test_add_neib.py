from math import pi
from command.neighbor_cmd import NeighborCommandBuilder
from control.controller import Controller

from db.neighbor_db import NeighborDB

from dashboard.neighbor_dashboard import Neighbor, NeighborDashboard
from dashboard.pipe_pool import PipePool

from logger import Logger, LogLevel
from time import sleep

def test_add_neib():

    Logger.set_current_log_level(LogLevel.INFO)

    pipe_pool = PipePool()
    pipe_pool.create("neighbor", "neighbor.pipe")

    neighborDB = NeighborDB()
    nb_dashboard = NeighborDashboard(neighborDB)

    ctrl = Controller()
    ctrl.connect()

    sleep(1.0)

    for i in range(6):
        req = NeighborCommandBuilder(
            bssid=f"00:11:22:33:44:5{i}",
            ssid=f"MyNet{i}",
            nr={"bssid": f"00:11:22:33:44:5{i}", "bssid_info": 0x1234, "reg_class": 81, "channel": 6, "phy_type": 7},
            lci={"latitude": 37.4219, "longitude": -122.0840, "altitude": 10.0},
            civic={"country": "US", "city": "MountainView"},
            stationary=True,
            bss_parameter=5
        )
        ctrl.add_neighbor(req)
        ctrl.get_neighbors(neighborDB)
        nb_dashboard.show(pipe=pipe_pool.get("neighbor"), replace=True)
        sleep(2)



    ctrl.disconnect()