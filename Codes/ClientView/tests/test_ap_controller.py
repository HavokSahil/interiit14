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

def test_sta():
    ap = ApController()
    ap.connect()
    print("Start sleeping")
    sleep(3)
    print("Done sleeping")
    ap.get_stations()

    for station in ap.stations:
        print("="*50)
        print(station)
        print("="*50)

    ap.disconnect()

def test_neighbors():
    ap = ApController()
    ap.connect()
    print("test_neighbors: sleep_begin")
    sleep(3)
    print("test_neighbors: sleep_end")
    nneib = ap.get_neighbor()
    print(f"Number of neighbors: {nneib}")
    for neib in ap.neighbors:
        print(neib)
    ap.disconnect()

def test_add_neighbor():
    ap = ApController()
    ap.connect()
    print("test_add_neighbor: sleep_begin")
    sleep(3)
    print("test_add_neighbor: sleep_end")
    
    # Create a NeighborInfo with required fields
    neibInfo = NeighborInfo()
    neibInfo.bssid = "00:11:22:33:44:55"
    neibInfo.ssid = "TestNeighbor"
    neibInfo.bssid_info = 0x00000000  # BSSID Information field
    neibInfo.oper_class = 81  # Operating class (e.g., 81 for 2.4 GHz)
    neibInfo.channel = 6  # Channel number
    neibInfo.phy_type = 7  # PHY type (e.g., 7 for HT)
    
    # Create NeighborRequestBuilder with the NeighborInfo
    neibBuilder = NeighborRequestBuilder(neighbor=neibInfo)
    
    # Build and print the command that will be sent
    cmd = neibBuilder.build()
    print(f"SET_NEIGHBOR command: {cmd}")
    
    # Add the neighbor
    if ap.add_neighbor(neibBuilder):
        print("Successfully added neighbor")
    else:
        print("Failed to add neighbor")
        Logger.log_err("Failed to add neighbor")

    ap.disconnect()
    
def test_all_ap_control():
    #test_add_neighbor()
    #test_neighbors()
    #assert(test_power())
    #assert(test_channel_switch())
    pass