from threading import RLock
from typing import Optional, Dict
import time
from model.measurement import BeaconMeasurement

DEFAULT_STA_MAC = "00:00:00:00:00:00"
EXPIRE_SEC = 30  # configurable TTL


class BeaconMeasurementDB:
    """BeaconMeasurement DB with soft expiration.
    Internally stores timestamps, but all public access returns the old format.
    """

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

        # internal: sta_mac -> list of (timestamp, BeaconMeasurement)
        self._store: Dict[str, list[tuple[float, BeaconMeasurement]]] = {}
        self._initialized = True

    # ------------------------------------------------------------
    # Add
    # ------------------------------------------------------------
    def add(self, bm: BeaconMeasurement, sta_mac: Optional[str] = None):
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC

        with self._lock:
            if sta_mac not in self._store:
                self._store[sta_mac] = []

            self._store[sta_mac].append((time.time(), bm))

    # ------------------------------------------------------------
    # Get (returns only unexpired BeaconMeasurement objects)
    # ------------------------------------------------------------
    def get(self, sta_mac: Optional[str] = None) -> list[BeaconMeasurement]:
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC

        now = time.time()

        with self._lock:
            if sta_mac not in self._store:
                return []

            return [
                bm
                for ts, bm in self._store[sta_mac]
                if now - ts <= EXPIRE_SEC
            ]

    # ------------------------------------------------------------
    # All (returns old API format: dict[str, list[BeaconMeasurement]])
    # ------------------------------------------------------------
    def all(self) -> Dict[str, list[BeaconMeasurement]]:
        now = time.time()

        with self._lock:
            return {
                sta_mac: [
                    bm
                    for ts, bm in entries
                    if now - ts <= EXPIRE_SEC
                ]
                for sta_mac, entries in self._store.items()
            }

    # ------------------------------------------------------------
    # Raw — always return old structure (NO timestamps)
    #      used for backward compatibility / dashboards if needed
    # ------------------------------------------------------------
    def raw(self) -> Dict[str, list[BeaconMeasurement]]:
        with self._lock:
            return {
                sta_mac: [bm for _, bm in entries]
                for sta_mac, entries in self._store.items()
            }

    # ------------------------------------------------------------
    # Clear / remove
    # ------------------------------------------------------------
    def remove(self, sta_mac: Optional[str] = None):
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC
        with self._lock:
            self._store.pop(sta_mac, None)

    def clear(self, sta_mac: Optional[str] = None):
        with self._lock:
            if sta_mac:
                self._store.pop(sta_mac, None)
            else:
                self._store.clear()

    # ------------------------------------------------------------
    # Count
    # ------------------------------------------------------------
    def count(self, sta_mac: Optional[str] = None) -> int:
        now = time.time()

        with self._lock:
            if sta_mac:
                if sta_mac not in self._store:
                    return 0
                return sum(1 for ts, _ in self._store[sta_mac] if now - ts <= EXPIRE_SEC)

            return sum(
                1
                for entries in self._store.values()
                for ts, _ in entries
                if now - ts <= EXPIRE_SEC
            )

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    def __contains__(self, sta_mac: str) -> bool:
        return len(self.get(sta_mac)) > 0

    def __len__(self):
        return self.count()

    def __repr__(self):
        return f"<BeaconMeasurementDB stations={len(self._store)}>"
