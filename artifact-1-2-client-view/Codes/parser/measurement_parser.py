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
    Parse an 802.11v BSS Transition Management Response.
    Keeps Logger.log_info() calls for full traceability.
    """

    Logger.log_info("Starting parse_bss_tm_response")

    # -------------------------------------------------------------
    # Validate Action frame
    # -------------------------------------------------------------
    if not dot11.haslayer(Dot11):
        Logger.log_info("No Dot11 layer found → abort")
        return None

    if dot11.type != 0 or dot11.subtype != 13:
        Logger.log_info(f"Not mgmt-action: type={dot11.type}, subtype={dot11.subtype}")
        return None

    action = dot11.getlayer(Dot11Action)
    if not action:
        Logger.log_info("Dot11Action layer missing")
        return None

    category = getattr(action, "category", None)
    act = getattr(action, "action", None)
    Logger.log_info(f"Action category={category}, action={act}")

    # WNM = 10, BSS TM Response = 8
    if category != 10 or act != 8:
        Logger.log_info("Not a BSS Transition Management Response")
        return None

    # -------------------------------------------------------------
    # Extract raw payload
    # -------------------------------------------------------------
    raw = bytes(action.payload)
    Logger.log_info(f"Raw BSS TM Response payload: {raw.hex()}")

    if len(raw) < 3:
        Logger.log_info("Payload too short (<3 bytes)")
        return None

    dialog_token = raw[1]
    status_code = raw[2]
    bss_term_delay = raw[3]
    Logger.log_info(
        f"Parsed fixed fields → dialog_token={dialog_token}, "
        f"status_code={status_code}, bss_term_delay={bss_term_delay}"
    )

    idx = 4  # cursor

    # -------------------------------------------------------------
    # Optional Target BSSID (exists only if status_code == 0)
    # -------------------------------------------------------------
    target_bssid = None

    if status_code == 0:
        if len(raw) >= idx + 6:
            target_bssid = ":".join(f"{b:02x}" for b in raw[idx:idx+6])
            Logger.log_info(f"Found Target BSSID: {target_bssid}")
            idx += 6
        else:
            Logger.log_info("Status=0 but frame too short for target BSSID")

    # -------------------------------------------------------------
    # Parse remaining IEs
    # -------------------------------------------------------------
    neighbor_reports = []
    vendor_ies = []
    extra_ies = []

    Logger.log_info("Beginning IE parsing loop")

    while idx + 2 <= len(raw):
        eid = raw[idx]
        length = raw[idx+1]
        ie_data = raw[idx+2:idx+2+length]

        if len(ie_data) != length:
            Logger.log_info(f"Malformed IE at idx={idx}, break")
            break

        Logger.log_info(f"IE: eid={eid}, len={length}, data={ie_data.hex()}")

        if eid == 52:  # Neighbor Report IE
            Logger.log_info(" → Neighbor Report IE found")
            neighbor_reports.append({"raw": ie_data})

        elif eid == 221:  # Vendor Specific IE
            Logger.log_info(" → Vendor Specific IE found")
            vendor_ies.append({
                "oui": ie_data[:3].hex(),
                "subtype": ie_data[3] if len(ie_data) > 3 else None,
                "data": ie_data
            })

        else:
            extra_ies.append({"eid": eid, "data": ie_data})

        idx += 2 + length

    Logger.log_info("Completed IE parsing")

    result = BSSTransitionResponse(
        dialog_token=dialog_token,
        status_code=status_code,
        bss_termination_delay=bss_term_delay,
        target_bssid=target_bssid,
        neighbor_reports=neighbor_reports,
        vendor_ies=vendor_ies,
        extra_ies=extra_ies
    )

    Logger.log_info(f"Completed parse_bss_tm_response → {result}")

    return result
