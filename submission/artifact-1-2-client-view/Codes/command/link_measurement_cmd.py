from model.mac_address import MacAddress
from logger import Logger

class LinkMeasurementCommandBuilder:
    def __init__(self, mac: str):
        self.mac = mac

    def build(self) -> str:
        if MacAddress.is_valid(self.mac):
            return f"REQ_LINK_MEASUREMENT {self.mac}"
        else:
            Logger.log_err("LinkMeasurementCommandBuilder.build(): invalid mac address")
            return ""
    
    def __str__(self) -> str:
        return self.build()