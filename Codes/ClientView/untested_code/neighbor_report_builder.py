#!/usr/bin/env python3

# Function to build the neighbor report hex
def build_nr(
    bssid: str,
    bssid_info: int,
    op_class: int,
    channel: int,
    phy: int,
    add_ssid=None,
    add_ht=False,
    add_vht=False,
    add_he=False
):
    """
    Build a Neighbor Report (NR) element with optional subelements.
    Returns hex string ready for hostapd SET_NEIGHBOR.
    """

    # Convert bssid into bytes
    bssid_bytes = bytes(int(x, 16) for x in bssid.split(":"))

    # mandatory 13 bytes body
    body = (
        bssid_bytes +
        bssid_info.to_bytes(2, "little") +
        bytes([op_class, channel, phy])
    )

    # optional subelements
    subelements = b""

    # SSID subelement
    if add_ssid:
        ssid_bytes = add_ssid.encode()
        subelements += bytes([0, len(ssid_bytes)]) + ssid_bytes

    # HT Capabilities (ID 45, 26 bytes)
    if add_ht:
        ht_caps = b"\x00" * 26   # generic placeholder
        subelements += bytes([45, len(ht_caps)]) + ht_caps

    # VHT Capabilities (ID 191, 12 bytes)
    if add_vht:
        vht_caps = b"\x00" * 12  # generic placeholder
        subelements += bytes([191, len(vht_caps)]) + vht_caps

    # HE Capabilities (ID 255 vendor-style, 5 bytes placeholder)
    if add_he:
        he_caps = b"\x01\x02\x03\x04\x05"
        subelements += bytes([255, len(he_caps)]) + he_caps

    # combine
    full_body = body + subelements

    # ie
    nr_ie = bytes([0x34, len(full_body)]) + full_body

    return nr_ie.hex()


# Function to build the neighbor for adding neighbors in bss tm request
def build_neighbor_text(
        bssid: str,
        bssid_info: int,
        op_class: int,
        channel: int,
        phy_type: int,
        subelements_hex: str = None
):
    """
    Build hostapd textual neighbor= entry.

    Format:
      neighbor=<BSSID>,<BSSID Info>,<op_class>,<channel>,<phy>[,<hexsubelems>]
    """
    base = f"neighbor={bssid},{hex(bssid_info)},{op_class},{channel},{phy_type}"
    if subelements_hex:
        base += f",{subelements_hex}"
    return base
