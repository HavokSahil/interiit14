from enum import Enum

class APStatus(Enum):
    AP_ENABLED = 1
    AP_DISABLED = 0

class AP:
    def __init__(ap):
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

    def to_dict(self):
        """
        Dump the complete state of the AP.
        Returns a dictionary of all AP attributes.
        """
        # Add all instance attributes, convert Enum to its name
        d = dict(
            state=self.state.name if isinstance(self.state, Enum) else self.state,
            phy=self.phy,
            freq=self.freq,
            hw_mode=self.hw_mode,
            country_code=self.country_code,
            channel=self.channel,
            beacon_int=self.beacon_int,
            supported_rates=self.supported_rates,
            max_txpower=self.max_txpower,
            bss=self.bss,
            bssid=self.bssid,
            ssid=self.ssid,
            num_sta=self.num_sta,
            raw=self.raw
        )
        return d

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