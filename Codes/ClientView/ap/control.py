from ap.models.builder import NeighborRequestBuilder
from ap.standard import WiFi24GHZChannels
from control.controller import Controller
from ap.models.status import ApStatus, ApStatusIndicator
from ap.models.info import NeighborInfo
from control.logger import *
from time import sleep
from client.models.datatype import *
from client.models.info import StationBasicInfo

class ApController(Controller):
    def __init__(self, ctrl_path="/var/run/hostapd/wlan0"):
        super().__init__(ctrl_path)
        self.sta = list[str]() # list of STA bssid
        self.nsta = 0          # number of connected STA
        self.status: ApStatus | None = None
        self.stations: list[MacAddress] = list[MacAddress]()
        self.neighbors: list[NeighborInfo] = list[NeighborInfo]()

        self.connect()
        self._load_status()

    def _load_status(self):
        if not self.send_command("STATUS"):
            return
        content: str | None = None
        while not content or not content.startswith("state="):
            content = self.receive()
        self.status = ApStatus.from_content(content)


    def enable(self):
        self._load_status()
        self.clear_events()
        if self.status and self.status.state == ApStatusIndicator.AP_ENABLED:
            return True
        if self.send_command("ENABLE"):
            event: str | None = None
            while event is None or event.find("AP-ENABLED") == -1:
                event = self.receive_event()
            
            print(f"Event was received: {event}")
            self._load_status()
            return self.status.state == ApStatusIndicator.AP_ENABLED
        return False


    def disable(self) -> bool:
        self._load_status()
        self.clear_events()
        if self.status and self.status.state == ApStatusIndicator.AP_DISABLED:
            return True
        if self.send_command("DISABLE"):
            event: str | None = None
            while event is None or event.find("AP-DISABLED") == -1:
                event = self.receive_event()

            print(f"Event was received: {event}")
            self._load_status()
            return self.status.state == ApStatusIndicator.AP_DISABLED
        return False


    def get_stations(self) -> int:

        self.nsta = 0
        self.stations.clear()

        timeout = 0.3 # keep low timeout, specifically for this function

        if self.send_command("STA-FIRST", timeout=timeout):
            content = self.receive()
            if not content: return 0
            stainfo: StationBasicInfo = StationBasicInfo.from_content(content)
            self.stations = [stainfo]
            while self.send_command(f"STA-NEXT {stainfo.mac}", timeout=timeout):
                content = self.receive(timeout=timeout)
                if not content:
                    break
                stainfo = StationBasicInfo.from_content(content)
                self.stations.append(stainfo)

        self.nsta = len(self.stations)
        return self.nsta

    def poll_station(self, mac: MacAddress) -> bool:
        if self.send_command(f"POLL_STA {mac.raw}"):
            return self.receive() == "OK"
        return False

    def restart(self, timeout: int = 3.0) -> bool:
        print("Trying to disable the AP while restarting")
        if not self.disable():
            Logger.log_err("Unable to disable while restarting the AP")
            return False

        sleep(timeout) # spin for the timeout

        print("Trying to enable the AP while restarting")
        if not self.enable():
            Logger.log_err("Unable to enable while restarting the AP")
            return False

        return True

    def switch_channel(self, freq: WiFi24GHZChannels, beacon_int: int = 10) -> bool:

        # TODO: fix this mess
        DEFAULT_RESTART_TIME = 1.0

        if not self.restart(DEFAULT_RESTART_TIME):
            Logger.log_debug("The restart failed, the AP may be in invalid state")
            return False

        # TODO: add the additional check for the valid frequency,
        # my guess is it will improve latency

        if self.send_command(f"CHAN_SWITCH {beacon_int} {freq.value}"):
            self._load_status()
            if self.status.freq == freq.value:
                return True
            else:
                Logger.log_err(f"The frequencies doesn't match after the command. ({freq.value} != {self.status.freq})")
        return False

    def add_neighbor(self, neib: NeighborRequestBuilder) -> bool:
        """Function to add neighbor report to the database"""
        cmd = neib.build()
        if self.send_command(f"{cmd}"):
            reply = self.receive().strip()
            print(f"REPLY_WAS: {reply}")
            return reply == "OK"
        return False

    def get_neighbor(self) -> int:
        print("Hello there")
        self.neighbors = []
        if not self.send_command("SHOW_NEIGHBOR"):
            return 0
        content = self.receive()
        print(f"Content was {content}")
        if not content: return 0
        for line in content.splitlines():
            self.neighbors.append(NeighborInfo.from_line(line))
        return len(self.neighbors)

    def reload_config(self):
        return self.send_command("RELOAD_CONFIG")

    def reload(self):
        return self.send_command("RELOAD")

    def __str__(self):
        if not self.status:
            # Try to refresh once for user feedback
            self._load_status()
        if not self.status:
            return "AP status not known"
        return self.status.__str__()