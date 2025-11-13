from threading import RLock
from typing import Optional, Dict
from dataclasses import dataclass, field
from model.measurement import BeaconMeasurement, BeaconReport

DEFAULT_STA_MAC = "00:00:00:00:00:00"

class BeaconMeasurementDB:
    """Thread-safe singleton database for BeaconMeasurement objects per station MAC."""

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
        self._db: Dict[str, list[BeaconMeasurement]] = {}  # sta_mac -> list of BeaconMeasurements
        self._initialized = True

    # -----------------------------
    # CRUD operations
    # -----------------------------
    def add(self, bm: BeaconMeasurement, sta_mac: Optional[str] = None):
        """Add a BeaconMeasurement for a given station MAC."""
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC
        with self._lock:
            if sta_mac not in self._db:
                self._db[sta_mac] = []
            self._db[sta_mac].append(bm)

    def remove(self, sta_mac: Optional[str] = None):
        """Remove all BeaconMeasurements for a given station MAC."""
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC
        with self._lock:
            self._db.pop(sta_mac, None)

    def get(self, sta_mac: Optional[str] = None) -> list[BeaconMeasurement]:
        """Get all BeaconMeasurements for a given STA MAC."""
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC
        with self._lock:
            return self._db.get(sta_mac, [])

    def all(self) -> Dict[str, list[BeaconMeasurement]]:
        """Return the entire database."""
        with self._lock:
            return {sta_mac: measurements[:] for sta_mac, measurements in self._db.items()}

    def count(self, sta_mac: Optional[str] = None) -> int:
        """Count measurements for a STA or total if None."""
        with self._lock:
            if sta_mac:
                return len(self._db.get(sta_mac, []))
            return sum(len(ms) for ms in self._db.values())

    def clear(self, sta_mac: Optional[str] = None):
        """Clear measurements for a STA or entire DB if None."""
        with self._lock:
            if sta_mac:
                self._db.pop(sta_mac, None)
            else:
                self._db.clear()

    # -----------------------------
    # Utilities
    # -----------------------------
    def __contains__(self, sta_mac: str) -> bool:
        with self._lock:
            return sta_mac in self._db

    def __len__(self):
        with self._lock:
            return sum(len(ms) for ms in self._db.values())

    def __iter__(self):
        """Iterate over all BeaconMeasurements (flattened)."""
        with self._lock:
            for measurements in self._db.values():
                for bm in measurements:
                    yield bm

    def __repr__(self):
        with self._lock:
            return f"<BeaconMeasurementDB stations={len(self._db)} total_measurements={len(self)}>"
