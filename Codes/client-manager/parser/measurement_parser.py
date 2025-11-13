from scapy.all import Dot11, Dot11Elt, Dot11Action
from scapy.packet import Raw
from typing import Optional
from model.measurement import BeaconMeasurement, LinkMeasurement, BSSTransitionResponse

from typing import Optional
from logger import Logger

def parse_beacon_measurement(dot11) -> Optional[BeaconMeasurement]:
    """
    Convert a Scapy 802.11k Beacon Report frame into a BeaconMeasurement object.
    Returns None if the frame is not a Beacon Measurement Report.
    Logs each step for debugging.
    """
    Logger.log_info("Starting parse_beacon_measurement")

    # Make sure it's a Dot11 object
    if not dot11.haslayer(Dot11):
        Logger.log_info("No Dot11 layer found")
        return None

    # Only management action frames
    if dot11.type != 0 or dot11.subtype != 13:  # 13 = Action frame
        Logger.log_info(f"Not a management action frame: type={dot11.type}, subtype={dot11.subtype}")
        return None
    Logger.log_info("Confirmed management action frame")

    # Ensure it has Dot11Action layer
    action = dot11.getlayer(Dot11Action)
    if not action:
        Logger.log_info("No Dot11Action layer found")
        return None
    Logger.log_info("Found Dot11Action layer")

    category = getattr(action, "category", None)
    raw_frame = action.getlayer(Raw).load
    act = int(raw_frame[0])

    # Only Beacon Measurement Reports
    if category != 5 or act != 1:
        Logger.log_info(f"Not a Beacon Measurement Report: category={getattr(action, 'category', None)}, action={getattr(action, 'action', None)}")
        return None
    Logger.log_info("Confirmed Beacon Measurement Report")

    bm: BeaconMeasurement = BeaconMeasurement.from_bytes(raw_frame)
    Logger.log_info(f"Extracted fixed fields: measurement_token={bm.measurement_token}, dialog_token={bm.dialog_token}, sta_mac={bm.sta_mac}")
    return bm


def parse_link_measurement(dot11) -> Optional[LinkMeasurement]:
    """
    Convert a Scapy 802.11k Link Measurement Report frame into a LinkMeasurement object.
    Returns None if the frame is not a Link Measurement Report.
    Logs each step for debugging.
    """
    Logger.log_info("Starting parse_link_measurement")

    # Ensure it's a Dot11 object
    if not dot11.haslayer(Dot11):
        Logger.log_info("No Dot11 layer found")
        return None

    # Only management action frames
    if dot11.type != 0 or dot11.subtype != 13:
        Logger.log_info(f"Not a management action frame: type={dot11.type}, subtype={dot11.subtype}")
        return None
    Logger.log_info("Confirmed management action frame")

    # Ensure it has Dot11Action layer
    action = dot11.getlayer(Dot11Action)
    if not action:
        Logger.log_info("No Dot11Action layer found")
        return None
    Logger.log_info("Found Dot11Action layer")

    # Radio Measurement category (5) and Measurement Report action (1)
    category = getattr(action, "category", None)
    raw_frame = action.getlayer(Raw).load
    act = int(raw_frame[0])

    Logger.log_info(f"Action layer category={category}, action={act}")
    if category != 5 or act != 3:
        Logger.log_info("Not a Link Measurement Report")
        return None

    lm = LinkMeasurement.from_bytes(raw_frame)
    lm.bssid = dot11.addr2
    lm.operating_class
    Logger.log_info(f"parse_link_measurement: {lm}")
    Logger.log_info("Completed parse_link_measurement")
    return lm


def parse_bss_tm_response(dot11) -> Optional[BSSTransitionResponse]:
    """
    Convert a Scapy 802.11v BSS Transition Management Response into a BSSTransitionResponse object.
    Returns None if the frame is not a BSS TM Response.
    Logs each step for debugging.
    """
    Logger.log_info("Starting parse_bss_tm_response")

    # Check for Dot11 layer
    if not dot11.haslayer(Dot11):
        Logger.log_info("No Dot11 layer found")
        return None

    # Only management action frames
    if dot11.type != 0 or dot11.subtype != 13:
        Logger.log_info(f"Not a management action frame: type={dot11.type}, subtype={dot11.subtype}")
        return None
    Logger.log_info("Confirmed management action frame")

    # Ensure Dot11Action layer exists
    action = dot11.getlayer(Dot11Action)
    if not action:
        Logger.log_info("No Dot11Action layer found")
        return None
    Logger.log_info("Found Dot11Action layer")

    # Radio Management category (10) for WNM / BSS Transition Response
    category = getattr(action, "category", None)
    act = getattr(action, "action", None)
    Logger.log_info(f"Action layer category={category}, action={act}")
    if category != 10 or act != 1:
        Logger.log_info("Not a BSS Transition Management Response")
        return None

    # Extract fields
    dialog_token = getattr(action, "dialog_token", 0)
    status_code = getattr(action, "status_code", 1)  # default failure

    bss_termination_delay = getattr(action, "bss_term_delay", None)
    disassoc_timer = getattr(action, "disassoc_timer", None)
    target_bssid = getattr(action, "target_bssid", None)
    preference = getattr(action, "bss_preference", None)
    abridged = getattr(action, "abridged", None)
    recomm_bss_list = getattr(action, "recomm_bss_list", [])

    Logger.log_info(
        f"Extracted BSS TM Response fields: dialog_token={dialog_token}, status_code={status_code}, "
        f"bss_term_delay={bss_termination_delay}, disassoc_timer={disassoc_timer}, "
        f"target_bssid={target_bssid}, preference={preference}, abridged={abridged}, "
        f"recomm_bss_list={recomm_bss_list}"
    )

    Logger.log_info("Completed parse_bss_tm_response")
    return BSSTransitionResponse(
        dialog_token=dialog_token,
        status_code=status_code,
        bss_termination_delay=bss_termination_delay,
        disassoc_timer=disassoc_timer,
        target_bssid=target_bssid,
        preference=preference,
        abridged=abridged,
        recomm_bss_list=recomm_bss_list
    )
