from typing import Any

class Station:
    """Represents the status information of an associated station."""

    def __init__(self):
        self.info_dict: dict[str, Any] = {}
        self.raw: str | None = None

        # Primary identifiers
        self.mac: str | None = None
        self.flags: list[str] = []

        # Basic STA info
        self.aid: int | None = None
        self.capability: list[str] | None = None
        self.listen_interval: int | None = None
        self.supported_rates: list[float] | None = None
        self.timeout_next: str | None = None

        # Traffic counters
        self.rx_packets: int | None = None
        self.tx_packets: int | None = None
        self.rx_bytes: int | None = None
        self.tx_bytes: int | None = None
        self.rx_airtime: int | None = None
        self.tx_airtime: int | None = None
        self.beacons_count: int | None = None
        self.rx_drop_misc: int | None = None
        self.backlog_packets: int | None = None
        self.backlog_bytes: int | None = None
        self.fcs_error_count: int | None = None
        self.beacon_loss_count: int | None = None
        self.expected_throughput: int | None = None
        self.tx_retry_count: int | None = None
        self.tx_retry_failed: int | None = None

        # Rate / airtime info
        self.tx_bitrate: int | None = None
        self.rx_bitrate: int | None = None
        self.tx_duration: int | None = None
        self.rx_duration: int | None = None

        # PHY layer fields
        self.rx_mcs: int | None = None
        self.tx_mcs: int | None = None
        self.rx_vhtmcs: int | None = None
        self.tx_vhtmcs: int | None = None
        self.rx_he_nss: int | None = None
        self.tx_he_nss: int | None = None
        self.rx_vht_nss: int | None = None
        self.tx_vht_nss: int | None = None
        self.rx_dcm: int | None = None
        self.tx_dcm: int | None = None
        self.rx_guard_interval: int | None = None
        self.tx_guard_interval: int | None = None

        # Signal / timing
        self.signal: int | None = None
        self.avg_signal: int | None = None
        self.avg_beacon_signal: int | None = None
        self.avg_ack_signal: int | None = None
        self.inactive_msec: int | None = None
        self.connected_sec: int | None = None

        # Extra info
        self.rx_rate_info: int | None = None
        self.tx_rate_info: int | None = None
        self.connected_time: int | None = None
        self.mbo_cell_capa: int | None = None
        self.supp_op_classes: str | None = None
        self.min_txpower: int | None = None
        self.max_txpower: int | None = None
        self.ext_capab: list[str] | None = None

    

    def to_dict(self) -> dict[str, Any]:
        return self.info_dict

    def __str__(self) -> str:
        """String representation with all available details."""
        lines = []
        lines.append("=== StationBasicInfo ===")
        for attr in vars(self):
            # Skip info_dict itself if desired, it's a dict dump of all
            if attr == "info_dict":
                continue
            value = getattr(self, attr)
            lines.append(f"{attr}: {value!r}")
        # Optionally show info_dict at the end
        lines.append("info_dict: {")
        for k, v in self.info_dict.items():
            lines.append(f"    {k!r}: {v!r},")
        lines.append("}")
        return "\n".join(lines)
