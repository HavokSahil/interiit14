from threading import RLock
from typing import Optional, Dict
from dataclasses import dataclass, field
from model.measurement import LinkMeasurement

DEFAULT_STA_MAC = "00:00:00:00:00:00"  # Default for AP-associated measurements

class LinkMeasurementDB:
    """Database for storing LinkMeasurement reports per station (STA MAC)."""

    _instance = None
    _lock = RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._db: Dict[str, "LinkMeasurement"] = {}  # sta_mac -> token -> LinkMeasurement
        self._initialized = True

    # -----------------------------
    # CRUD
    # -----------------------------
    def add(self, lm: "LinkMeasurement", sta_mac: Optional[str] = None):
        """Add a LinkMeasurement for a given station MAC."""
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC
        with self._lock:
            if sta_mac not in self._db:
                self._db[sta_mac] = {}
            self._db[sta_mac] = lm

    def remove(self, sta_mac: Optional[str] = None):
        """Remove a LinkMeasurement by token for a station."""
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC
        with self._lock:
            if sta_mac in self._db:
                self._db.pop(sta_mac)

    def get(self, sta_mac: Optional[str] = None) -> Optional["LinkMeasurement"]:
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC
        with self._lock:
            return self._db.get(sta_mac, {})

    def all(self) -> Dict[str, list["LinkMeasurement"]]:
        with self._lock:
            return {sta_mac: lms for sta_mac, lms in self._db.items()}

    def count(self) -> int:
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC
        with self._lock:
            return len(self._db.items())

    def clear(self, sta_mac: Optional[str] = None):
        if sta_mac is None:
            with self._lock:
                self._db.clear()
        else:
            with self._lock:
                self._db.pop(sta_mac, None)

    # -----------------------------
    # Utilities
    # -----------------------------
    def __contains__(self, sta_mac: str) -> bool:
        return sta_mac in self._db

    def __len__(self):
        with self._lock:
            return sum(len(lms) for lms in self._db.values())

    def __iter__(self):
        """Iterate over all LinkMeasurements (flattened)."""
        with self._lock:
            for lms in self._db.values():
                for lm in lms.values():
                    yield lm

    def to_dict(self):
        """Export as nested dict: sta_mac -> token -> dict."""
        with self._lock:
            return {
                sta_mac: {token: vars(lm) for token, lm in lms.items()}
                for sta_mac, lms in self._db.items()
            }

    def __repr__(self):
        with self._lock:
            return f"<LinkMeasurementDB stations={len(self._db)} total_measurements={len(self)}>"
