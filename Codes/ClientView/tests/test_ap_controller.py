from ap.control import *
from time import sleep
from ap.standard import *

def test_power() -> bool:
    ap = ApController()
    Logger.log_info("Disabling the AP")
    if not ap.disable():
        Logger.log_info("Failed to disable the AP")
    sleep(2)
    Logger.log_info("Enabling the AP")
    if not ap.enable():
        Logger.log_info("Failed to enable the AP")
    ap.disconnect()
    return True

def test_channel_switch() -> bool:
    ap = ApController()
    print(ap)

    if not ap.switch_channel(WiFi24GHZChannels.CHANNEL_1):
        Logger.log_err(f"Failed to switch to channel with ch. freq. : {WiFi24GHZChannels.CHANNEL_1}")
        return False

    if not ap.switch_channel(WiFi24GHZChannels.CHANNEL_6):
        Logger.log_err(f"Failed to switch to channel with ch. freq. : {WiFi24GHZChannels.CHANNEL_6}")
        return False

    return True
    


def test_all_ap_control():
    assert(test_power())
    assert(test_channel_switch())