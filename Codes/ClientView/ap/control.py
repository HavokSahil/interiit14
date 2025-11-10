from control.controller import Controller
from ap.models.status import ApStatus, ApStatusIndicator

class ApController(Controller):
    def __init__(self, ctrl_path="/var/run/hostapd/wlan0"):
        super().__init__(ctrl_path)
        self.sta = list[str]() # list of STA bssid
        self.nsta = 0          # number of connected STA
        self.status: ApStatus | None = None

        self.connect()
        self._load_status()

    def _load_status(self):
        if not self.send_command("STATUS"):
            return
        content = self.receive(timeout=1.0)
        if content:
            self.status = ApStatus.from_content(content)

    def enable(self):
        if self.status and self.status.state != ApStatusIndicator.AP_ENABLED:            
            if self.send_command("ENABLE"):
                self._load_status()

    def disable(self):
        if self.status and self.status.state != ApStatusIndicator.AP_DISABLED:
            if self.send_command("DISABLE"):
                self._load_status()

    def __str__(self):
        if not self.status:
            # Try to refresh once for user feedback
            self._load_status()
        if not self.status:
            return "AP status not known"
        return self.status.__str__()