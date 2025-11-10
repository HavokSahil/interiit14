from ap.models.status import ApStatus
from client.models.info import StaInfo

def test_model_ApStatus():
    content = """state=ENABLED
phy=phy0
freq=2412
num_sta_non_erp=0
num_sta_no_short_slot_time=0
num_sta_no_short_preamble=0
olbc=0
num_sta_ht_no_gf=0
num_sta_no_ht=0
num_sta_ht_20_mhz=0
num_sta_ht40_intolerant=0
olbc_ht=0
ht_op_mode=0x0
hw_mode=g
country_code=IN
country3=0x20
cac_time_seconds=0
cac_time_left_seconds=N/A
channel=1
edmg_enable=0
edmg_channel=0
secondary_channel=0
ieee80211n=0
ieee80211ac=0
ieee80211ax=0
ieee80211be=0
beacon_int=100
dtim_period=2
supported_rates=02 04 0b 16 0c 12 18 24 30 48 60 6c
max_txpower=22
bss[0]=wlan0
bssid[0]=80:32:53:02:1d:cc
ssid[0]=test
num_sta[0]=0"""

    apstatus = ApStatus.from_content(content)
    print(apstatus)

def test_model_StaInfo():
    info_str = """0e:ed:c0:e3:59:0e
flags=[AUTH][ASSOC][AUTHORIZED][WMM]
aid=1
capability=0x1401
listen_interval=1
supported_rates=82 84 8b 96 0c 12 18 24 30 48 60 6c
timeout_next=NULLFUNC POLL
rx_packets=180
tx_packets=33
rx_bytes=18215
tx_bytes=3541
inactive_msec=2408
signal=-38
rx_rate_info=540
tx_rate_info=540
connected_time=16
mbo_cell_capa=1
supp_op_classes=51515354737475767778797a7b7c7d7e7f808182
min_txpower=8
max_txpower=19
ext_capab=040008020140
"""
    info = StaInfo.from_content(info_str)
    print(info)


def test_all_models():
    print("[+] Running Model Test for `ApStatus`")
    test_model_ApStatus()
    print("[+] Running Model Test for `StaInfo`")
    test_model_StaInfo()