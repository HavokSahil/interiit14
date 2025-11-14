class Neighbor:
    """Represents one neighbor entry from hostapd's SHOW_NEIGHBOR output."""

    def __init__(self):
        self.bssid: str | None = None
        self.ssid: str | None = None
        self.nr_raw: str | None = None
        self.bssid_info: int | None = None
        self.oper_class: int | None = None
        self.channel: int | None = None
        self.phy_type: int | None = None
        self.oper_class_desc: str | None = None
        self.phy_type_desc: str | None = None
        self.subelements: str | None = None

        # NOTE: these are optional fields and are not included in nr report
        self.rcpi: int | None = None
        self.rsni: int | None = None

    def to_dict(self) -> dict:
        """Return a dictionary representation (clean and JSON-serializable)."""
        return {
            "bssid": self.bssid,
            "ssid": self.ssid,
            "bssid_info": self.bssid_info,
            "oper_class": self.oper_class,
            "channel": self.channel,
            "phy_type": self.phy_type,
            "oper_class_desc": self.oper_class_desc,
            "phy_type_desc": self.phy_type_desc,
            "subelements": self.subelements,
            "nr_raw": self.nr_raw,
        }

    def __dict__(self):
        return self.to_dict()
    
    def __str__(self):
        return (f"<Neighbor bssid={self.bssid} ssid={self.ssid} "
                f"class={self.oper_class_desc} ch={self.channel} "
                f"phy={self.phy_type_desc}>")
