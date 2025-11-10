from ap.control import *
from time import sleep

def test_power():
    ap = ApController()
    print(ap)
    ap.disable()
    sleep(2)
    ap.enable()
    ap.disconnect()

def test_all_ap_control():
    test_power()