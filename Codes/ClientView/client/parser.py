import re
from typing import Optional, Tuple


def extract_buf_from_event(event: str) -> Optional[str]:
    """
    Extract the buf hex string from AP-MGMT-FRAME-RECEIVED event.
    
    Args:
        event: The event string, e.g., "<3>AP-MGMT-FRAME-RECEIVED buf=..."
    
    Returns:
        The hex string if found, None otherwise.
    """
    match = re.search(r'buf=([0-9a-fA-F]+)', event)
    return match.group(1) if match else None


def extract_link_measurement_response_data(event: str) -> Optional[Tuple[str, str, str]]:
    """
    Extract data from LINK-MSR-RESP-RX event.
    
    Args:
        event: The event string, e.g., "<3>LINK-MSR-RESP-RX <mac> <number> <hex>"
    
    Returns:
        Tuple of (mac_address, number, hex_string) if successful, None otherwise.
    """
    parts = event.split()
    if len(parts) < 4:
        return None
    
    # parts[0] = "<3>LINK-MSR-RESP-RX"
    # parts[1] = MAC address
    # parts[2] = number
    # parts[3] = hex string
    return (parts[1], parts[2], parts[3])


def extract_beacon_response_data(event: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Extract data from BEACON-RESP-RX event.
    
    Args:
        event: The event string, e.g., "<3>BEACON-RESP-RX <mac> <number> <flags> <hex>"
    
    Returns:
        Tuple of (mac_address, number, flags, hex_string) if successful, None otherwise.
    """
    parts = event.split()
    if len(parts) < 5:
        return None
    
    # parts[0] = "<3>BEACON-RESP-RX"
    # parts[1] = MAC address
    # parts[2] = number
    # parts[3] = flags
    # parts[4] = hex string (may contain spaces, so join remaining parts)
    mac = parts[1]
    number = parts[2]
    flags = parts[3]
    hex_data = ''.join(parts[4:])  # Join remaining parts in case hex string has spaces
    
    return (mac, number, flags, hex_data)


def extract_beacon_req_tx_status(event: str) -> Optional[Tuple[str, str, bool]]:
    """
    Extract data from BEACON-REQ-TX-STATUS event.
    
    Args:
        event: The event string, e.g., "<3>BEACON-REQ-TX-STATUS <mac> <number> ack=1"
    
    Returns:
        Tuple of (mac_address, number, acknowledged) if successful, None otherwise.
    """
    parts = event.split()
    if len(parts) < 4:
        return None
    
    # parts[0] = "<3>BEACON-REQ-TX-STATUS"
    # parts[1] = MAC address
    # parts[2] = number (measurement token or dialog token)
    # parts[3] = "ack=1" or "ack=0"
    mac = parts[1]
    number = parts[2]
    
    # Extract ack value
    ack_str = parts[3]
    ack_match = re.search(r'ack=(\d+)', ack_str)
    if ack_match:
        ack_value = int(ack_match.group(1))
        acknowledged = (ack_value == 1)
    else:
        # Default to False if ack not found
        acknowledged = False
    
    return (mac, number, acknowledged)

