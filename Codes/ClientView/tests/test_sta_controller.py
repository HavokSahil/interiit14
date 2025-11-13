
from ap.control import *
from ap.standard import WiFi24GHZChannelsNo
from client.control import *
from client.models.datatype import *

def test_sta_req_link_measurement():
    ap =  ApController()
    ap.connect()
    print("Sleep for sometime")
    sleep(2)
    print("Wake up from sleep")

    ap.get_stations()
    stations: list[StationBasicInfo] = ap.stations
    for station in stations:
        sta = StationController(ap, MacAddress(station.mac))
        if not sta.request_link_measurement():
            print(f"Link measurement request failed for {sta.mac}")

    ap.disconnect()

def test_sta_req_beacon():
    ap = ApController()
    ap.connect()
    print("test_sta_req_beacon: sleeping")
    sleep(2)
    print("test_sta_req_beacon: wake up")

    ap.get_stations()
    stations: list[StationBasicInfo] = ap.stations

    builder = (
        ReqBeaconBuilder()
        .set_req_mode(0x01)
        .set_measurement_params(
            operating_class=81,
            channel_number=WiFi24GHZChannelsNo.CHANNEL_1.value,
            randomization_interval=0,
            measurement_duration=50,
            measurement_mode=0,  # 0 = passive, 1 = active, 2 = table
            bssid="ff:ff:ff:ff:ff:ff"
        )
    )

    for station in stations:
        sta = StationController(ap, MacAddress(station.mac))
        if not sta.req_beacon(builder=builder):
            print(f"Req beacon failed for {sta.mac}")

        for value in sta.beacon_measurements:
            print(value)
    
    ap.disconnect()
            

def test_all_sta_control():
    # test_sta_req_link_measurement()
    test_sta_req_beacon()