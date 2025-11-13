from enum import Enum

class APStatus(Enum):
    AP_ENABLED = 1
    AP_DISABLED = 0

class AP:
    def __init__(ap):
        ap.status_dict = dict()
        ap.state = APStatus.AP_DISABLED
        ap.phy = None
        ap.freq = None
        ap.hw_mode = None
        ap.country_code = None
        ap.channel = None
        ap.beacon_int = None
        ap.supported_rates = None
        ap.max_txpower = None
        ap.bss = None
        ap.bssid = None
        ap.ssid = None
        ap.num_sta = None
        ap.raw = None

    def __dict__(self):
        return self.status_dict

    def __str__(self):
        # Provide a readable string representation of the AP status
        attrs = [
            f"state={self.state.name}",
            f"phy={self.phy}",
            f"freq={self.freq}",
            f"hw_mode={self.hw_mode}",
            f"country_code={self.country_code}",
            f"channel={self.channel}",
            f"beacon_int={self.beacon_int}",
            f"supported_rates={self.supported_rates}",
            f"max_txpower={self.max_txpower}",
            f"bss={self.bss}",
            f"bssid={self.bssid}",
            f"ssid={self.ssid}",
            f"num_sta={self.num_sta}"
        ]
        return "<ApStatus " + " ".join(attrs) + ">"