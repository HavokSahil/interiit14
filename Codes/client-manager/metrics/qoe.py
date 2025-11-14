from db.lmrep_db import LinkMeasurementDB
from db.qoe_db import QoEDB
from db.station_db import StationDB
from model.measurement import LinkMeasurement
from model.station import Station

class QoE:
    """Compute and maintain QoE scores for stations."""

    def __init__(self, stdb: StationDB, qoedb: QoEDB, lmdb: LinkMeasurementDB):
        self.stdb = stdb
        self.qoedb = qoedb
        self.lmdb = lmdb

    # -----------------------------
    # Utility Computation Functions
    # -----------------------------
    @staticmethod
    def _qoe_link(station: Station, lm: LinkMeasurement) -> float:
        """Compute QoE using station + link measurement info."""
        # 1. Connection time utility
        uconn = min(station.connected_sec / 600, 1.0) if station.connected_sec else 0.0

        # 2. Inactivity utility
        uinact = max(0.0, 1.0 - min(station.inactive_msec / 10000, 1.0)) if station.inactive_msec else 1.0

        # 3. Supported rate utility
        rmax = max(station.supported_rates) if station.supported_rates else 0
        urate = min(rmax / 600, 1.0)

        # 4. Flags utility
        auth = 1.0 if "AUTHORIZED" in station.flags else 0.0
        assoc = 1.0 if "ASSOC" in station.flags else 0.0
        uflags = min(1.0, 0.5 + 0.3 * auth + 0.2 * assoc)

        # 5. Transmit power utility
        tx_power = getattr(station, "tx_bitrate", 0)  # fallback if tx power not set
        utx = min(tx_power / 30, 1.0) if tx_power else 0.0

        # Weighted QoE
        qoe_score = 0.25 * uconn + 0.25 * uinact + 0.20 * urate + 0.10 * uflags + 0.05 * utx
        return qoe_score

    @staticmethod
    def _qoe_nolink(station: Station) -> float:
        """Compute QoE without link measurement info."""
        # Same as _qoe_link but without LM-specific attributes
        uconn = min(station.connected_sec / 600, 1.0) if station.connected_sec else 0.0
        uinact = max(0.0, 1.0 - min(station.inactive_msec / 10000, 1.0)) if station.inactive_msec else 1.0
        rmax = max(station.supported_rates) if station.supported_rates else 0
        urate = min(rmax / 600, 1.0)
        auth = 1.0 if "AUTHORIZED" in station.flags else 0.0
        assoc = 1.0 if "ASSOC" in station.flags else 0.0
        uflags = min(1.0, 0.5 + 0.3 * auth + 0.2 * assoc)
        tx_power = getattr(station, "tx_bitrate", 0)
        utx = min(tx_power / 30, 1.0) if tx_power else 0.0

        qoe_score = 0.25 * uconn + 0.25 * uinact + 0.20 * urate + 0.10 * uflags + 0.05 * utx
        return qoe_score

    # -----------------------------
    # Update All Stations QoE
    # -----------------------------
    def update(self):
        """Iterate through all stations and update QoE database."""
        for station in self.stdb.all():
            mac = station.mac
            qoe = 0.0
            lm = self.lmdb.get(mac)
            if lm:
                qoe = QoE._qoe_link(station, lm)
            else:
                qoe = QoE._qoe_nolink(station)

            self.qoedb.set(mac, qoe)
