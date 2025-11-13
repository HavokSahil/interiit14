class ReqBeaconBuilder:
    """
    Build a REQ_BEACON command for hostapd from human-readable parameters.
    """
    def __init__(self, dest_mac: str = ""):
        self.dest_mac = dest_mac
        self.req_mode = 0
        self.operating_class = 0
        self.channel_number = 0
        self.randomization_interval = 0
        self.measurement_duration = 0
        self.measurement_mode = 0
        self.bssid = b'\xff\xff\xff\xff\xff\xff'  # default: wildcard BSSID
        self.subelements = b''

    def set_req_mode(self, mode: int):
        self.req_mode = mode
        return self

    def set_measurement_params(self, operating_class: int, channel_number: int,
                               randomization_interval: int, measurement_duration: int,
                               measurement_mode: int, bssid: str):
        """All standard fields from IEEE 802.11k Beacon Request body."""
        self.operating_class = operating_class
        self.channel_number = channel_number
        self.randomization_interval = randomization_interval
        self.measurement_duration = measurement_duration
        self.measurement_mode = measurement_mode
        self.bssid = bytes.fromhex(bssid.replace(':', ''))
        return self

    def add_subelements(self, data: bytes):
        """Optional variable-length subelements (raw bytes)."""
        self.subelements = data
        return self

    def _build_payload(self) -> bytes:
        """Assemble payload binary per IEEE 802.11k spec."""
        payload = bytearray()
        payload.append(self.operating_class & 0xFF)
        payload.append(self.channel_number & 0xFF)
        payload.extend(self.randomization_interval.to_bytes(2, 'little'))
        payload.extend(self.measurement_duration.to_bytes(2, 'little'))
        payload.append(self.measurement_mode & 0xFF)
        payload.extend(self.bssid)
        payload.extend(self.subelements)
        return bytes(payload)

    def build(self) -> str:
        """Return the final REQ_BEACON command string for hostapd."""
        payload_hex = self._build_payload().hex()
        if self.req_mode:
            return f"REQ_BEACON {self.dest_mac} req_mode={self.req_mode:02x} {payload_hex}"
        else:
            return f"REQ_BEACON {self.dest_mac} {payload_hex}"
