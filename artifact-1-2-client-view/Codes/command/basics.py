from defaults.enums import WiFi24GHZChannels

class BasicCommand:
    @staticmethod
    def enable() -> str:
        return "ENABLE"

    def status() -> str:
        return "STATUS"

    @staticmethod
    def disable() -> str:
        return "DISABLE"

    @staticmethod
    def station_info(mac: str) -> str:
        return f"STA {mac}"

    @staticmethod
    def first_station() -> str:
        return "STA-FIRST"
    
    @staticmethod
    def next_station(mac: str) -> str:
        return f"STA-NEXT {mac}"

    @staticmethod
    def chan_switch(beacon_interv: int, freq: WiFi24GHZChannels) -> str:
        return f"CHAN_SWITCH {beacon_interv} {freq.value}"

    @staticmethod
    def show_neighbor() -> str:
        return "SHOW_NEIGHBOR"

    @staticmethod
    def remove_neighbor(bssid: str) -> str:
        return f"REMOVE_NEIGHBOR {bssid}"

    @staticmethod
    def reload_config() -> str:
        return "RELOAD_CONFIG"

    @staticmethod
    def reload() -> str:
        return "RELOAD"