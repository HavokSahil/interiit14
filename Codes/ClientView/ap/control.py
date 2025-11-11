from ap.standard import WiFi24GHZChannels
from control.controller import Controller
from ap.models.status import ApStatus, ApStatusIndicator
from control.logger import *
from time import sleep

class ApController(Controller):
    def __init__(self, ctrl_path="/var/run/hostapd/wlan0"):
        super().__init__(ctrl_path)
        self.sta = list[str]() # list of STA bssid
        self.nsta = 0          # number of connected STA
        self.status: ApStatus | None = None
        self.

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