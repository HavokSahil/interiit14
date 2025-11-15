import re
import hashlib

class MacAddress:
    """Object for MAC address handling, parsing, and representation."""
    def __init__(self, mac: str):
        # Accepts colon, dash, or dot separators, or none, keep in hex
        pattern = re.compile(
            r"^(?P<mac>([0-9A-Fa-f]{2}[:\-\.]?){5,7}[0-9A-Fa-f]{2})$"
        )
        match = pattern.match(mac)
        if not match:
            raise ValueError(f"Invalid MAC address format: {mac}")
        self.raw = self._normalize(mac)
        self.octets = self.raw.split(":")

    @staticmethod
    def _normalize(mac: str) -> str:
        # Remove any separator, force lower, insert ":"
        normalized = re.sub(r'[^0-9A-Fa-f]', '', mac).lower()
        if len(normalized) != 12:
            raise ValueError(f"MAC address must be 12 hex digits: {mac}")
        return ':'.join(normalized[i:i+2] for i in range(0, 12, 2))

    def __str__(self):
        return self.raw

    def __repr__(self):
        return f"MacAddress('{self.raw}')"

    def is_multicast(self) -> bool:
        # Check least significant bit of first octet
        first_octet = int(self.octets[0], 16)
        return bool(first_octet & 0x01)

    def is_unicast(self) -> bool:
        return not self.is_multicast()

    def is_broadcast(self) -> bool:
        return all(o == 'ff' for o in self.octets)

    def is_local_administered(self) -> bool:
        # Second least significant bit of first octet is set
        first_octet = int(self.octets[0], 16)
        return bool(first_octet & 0x02)

    def oui(self) -> str:
        """Return the OUI (Organizationally Unique Identifier) as a string."""
        return ':'.join(self.octets[:3])

    def client_part(self) -> str:
        """Return the client-specific part (last three octets) as a string."""
        return ':'.join(self.octets[3:])

    def anonymized(self, hash_algorithm: str = "sha256", length: int = 12) -> str:
        """
        Return the MAC address as OUI:HASHED_CLIENT_PART.
        hash_algorithm: Hash function to use (e.g. 'sha256', 'md5').
        length: Number of hex characters from digest to use.
        """
        client = self.client_part()
        hasher = hashlib.new(hash_algorithm)
        hasher.update(client.encode('utf-8'))
        digest = hasher.hexdigest()[:length]
        # Reformat into MAC form: XX:XX:XX:xx:xx:xx
        hashed_client = ':'.join(digest[i:i+2] for i in range(0, min(6, length), 2))
        # If length < 6, fill with zeroes
        while hashed_client.count(':') < 2:
            hashed_client += ':00'
        return f"{self.oui()}:{hashed_client}"

    @staticmethod
    def is_valid(mac: str) -> bool:
        """
        Check if the given string is a valid MAC address format.
        Acceptable formats:
          - XX:XX:XX:XX:XX:XX
          - XX-XX-XX-XX-XX-XX
          - XXXXXXXXXXXX
        Returns True if valid, False otherwise.
        """
        if not isinstance(mac, str):
            return False
        mac = mac.strip()
        # 6 groups of 2 hex digits separated by : or -
        import re
        if re.fullmatch(r"([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})", mac):
            return True
        # 12 hex digits (no separators)
        if re.fullmatch(r"[0-9A-Fa-f]{12}", mac):
            return True
        return False
