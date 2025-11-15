from control.hostapd import HostapdController
from command.basics import BasicCommand
from command.neighbor_cmd import NeighborCommandBuilder
from command.link_measurement_cmd import LinkMeasurementCommandBuilder
from command.request_beacon_cmd import RequestBeaconCommandBuilder
from command.bss_tm_cmd import BssTmRequestBuilder
from time import sleep
from model.mac_address import MacAddress
from model.neighbor import Neighbor
from parser.neighbor_parser import NeighborParser
from parser.station_parser import StationParser
from db.neighbor_db import NeighborDB
from db.station_db import StationDB
from logger import Logger
from protocol.rxmux import RxMux

CONTROLLER_WAIT_LOOP_SEC = 0.5

class Controller(HostapdController):
    def __init__(self, ctrl_path="/var/run/hostapd/wlan0", iface="wlan0", rxmux: RxMux = None):
        super().__init__(ctrl_path, iface=iface, rxmux=rxmux)

    @staticmethod
    def _event_enabled(e: str) -> bool:
        return "AP-ENABLED" in e

    @staticmethod
    def _event_disabled(e: str) -> bool:
        return "AP-DISABLED" in e

    def enable(self):
        if self.send_command(BasicCommand.enable()):
            print("waiting for AP's confirmation (enable)", end='')
            while True:
                msg = self.receive_event()
                if Controller._event_enabled(msg):
                    print()
                    break
                print(".", end='')
                sleep(CONTROLLER_WAIT_LOOP_SEC)

    def disable(self):
        if self.send_command(BasicCommand.disable()):
            print("waiting for AP's confirmation (disable)", end='')
            while True:
                msg = self.receive_event()
                if Controller._event_disabled():
                    print()
                    break
                print(".", end='')
                sleep(CONTROLLER_WAIT_LOOP_SEC)

    def restart(self):
        self.disable()
        self.enable()

    def get_stations(self, stdb: StationDB) -> int:

        # NOTE: keeping the station timeout as low, cause its local operation
        GET_STATION_TIMEOUT = 0.3

        stdb.clear()
        Logger.log_info("get_stations: cleared the StationDB")

        if self.send_command(BasicCommand.first_station(), timeout=GET_STATION_TIMEOUT):
            msg = self.receive(timeout=GET_STATION_TIMEOUT)
            while msg:
                first_line = msg.split("\n")
                if len(first_line) == 0: break
                if not MacAddress.is_valid(first_line[0].strip()):
                    break
                station = StationParser.from_content(msg)
                if not station:
                    break
                stdb.add(station)
                mac = station.mac
                if not self.send_command(BasicCommand.next_station(mac), timeout=GET_STATION_TIMEOUT):
                    break
                msg = self.receive(timeout=GET_STATION_TIMEOUT)

        return stdb.count()

    def get_neighbors(self, ndb: NeighborDB) -> int:
        ndb.clear()
        Logger.log_info("get_neighbors: cleared the NeighborDB")
        if self.send_command(BasicCommand.show_neighbor()):
            raw = self.receive()
            count = 0
            for line in raw.splitlines():
                # TODO: should add these neighbors to the database
                neib: Neighbor = NeighborParser.from_line(line)
                ndb.add(neib)
                count+=1
                Logger.log_debug(f"get_neighbors: added {neib.bssid} to the NeighborDB")
            return count            
        return 0

    def add_neighbor(self, neib: NeighborCommandBuilder):
        if self.send_command(f"{neib}"):
            return self.last_cmd_status
        return False

    def remove_neighbor(self, bssid: str) -> bool:
        if self.send_command(BasicCommand.remove_neighbor(bssid)):
            return self.last_cmd_status
        return False

    def req_link_measurement(self, lm: LinkMeasurementCommandBuilder):
        if self.send_command(f"{lm}"):
            return self.last_cmd_status
        return False

    @staticmethod
    def _check_beacon_req_ack(msg) -> bool:
        return msg and msg == "BEACON-REQ-TX-STATUS"

    def req_beacon_measurement(self, rbm: RequestBeaconCommandBuilder):
        if self.send_command(f"{rbm}"):
            return self.last_cmd_status
        return False

    def req_bss_tm(self, rbtm: BssTmRequestBuilder) -> bool:
        if self.send_command(f"{rbtm}"):
            return self.last_cmd_status
        return False