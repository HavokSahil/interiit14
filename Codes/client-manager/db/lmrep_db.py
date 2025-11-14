import time
from threading import RLock
from typing import Optional, Dict
from model.measurement import LinkMeasurement

DEFAULT_STA_MAC = "00:00:00:00:00:00"


class LinkMeasurementDB:
    """Thread-safe DB storing exactly one LinkMeasurement per station,
    with soft TTL expiration that applies ONLY on reads.
    Internally stores timestamps but NEVER exposes them.
    """

    _instance = None
    _lock = RLock()

    expiration_sec = 30  # configurable TTL

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        # sta_mac -> (timestamp, LinkMeasurement)
        self._store: Dict[str, tuple[float, LinkMeasurement]] = {}
        self._initialized = True

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _is_expired(self, ts: float) -> bool:
        return (time.time() - ts) > self.expiration_sec

    # ------------------------------------------------------------
    # Add
    # ------------------------------------------------------------
    def add(self, lm: LinkMeasurement, sta_mac: Optional[str] = None):
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC

        with self._lock:
            self._store[sta_mac] = (time.time(), lm)

    # ------------------------------------------------------------
    # Get (filters expired)
    # ------------------------------------------------------------
    def get(self, sta_mac: Optional[str] = None) -> Optional[LinkMeasurement]:
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC

        with self._lock:
            item = self._store.get(sta_mac)
            if not item:
                return None

            ts, lm = item
            if self._is_expired(ts):
                return None

            return lm

    # ------------------------------------------------------------
    # All (filters expired)
    # ------------------------------------------------------------
    def all(self) -> Dict[str, LinkMeasurement]:
        now = time.time()
        ttl = self.expiration_sec

        with self._lock:
            return {
                mac: lm
                for mac, (ts, lm) in self._store.items()
                if now - ts <= ttl
            }

    # ------------------------------------------------------------
    # Raw (no filtering, no timestamps, backward compatible)
    # ------------------------------------------------------------
    def raw(self) -> Dict[str, LinkMeasurement]:
        with self._lock:
            return {mac: lm for mac, (_, lm) in self._store.items()}

    # ------------------------------------------------------------
    # Count (expired filtered)
    # ------------------------------------------------------------
    def count(self) -> int:
        now = time.time()
        ttl = self.expiration_sec

        with self._lock:
            return sum(1 for ts, _ in self._store.values() if now - ts <= ttl)

    # ------------------------------------------------------------
    # Clears
    # ------------------------------------------------------------
    def clear(self, sta_mac: Optional[str] = None):
        with self._lock:
            if sta_mac:
                self._store.pop(sta_mac, None)
            else:
                self._store.clear()

    def remove(self, sta_mac: Optional[str] = None):
        self.clear(sta_mac)

    # ------------------------------------------------------------
    # Magic methods (all must respect expiration)
    # ------------------------------------------------------------
    def __contains__(self, sta_mac: str) -> bool:
        return self.get(sta_mac) is not None

    def __len__(self):
        return self.count()

    def __iter__(self):
        with self._lock:
            now = time.time()
            ttl = self.expiration_sec
            for ts, lm in self._store.values():
                if now - ts <= ttl:
                    yield lm

    def __repr__(self):
        return f"<LinkMeasurementDB stations={len(self._store)}>"
