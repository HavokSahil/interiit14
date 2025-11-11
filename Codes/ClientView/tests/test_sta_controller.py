
from ap.control import *
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


def test_all_sta_control():
    test_sta_req_link_measurement()