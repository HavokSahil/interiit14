from threading import RLock
from typing import Optional, Dict

class QoEDB:
    """Database for storing QoE (Quality of Experience) for each station."""

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
        self._db: Dict[str, float] = {}  # sta_mac -> qoe_value
        self._initialized = True

    # -----------------------------
    # CRUD Operations
    # -----------------------------
    def set(self, sta_mac: str, qoe_value: float):
        """Set or update QoE for a station."""
        with self._lock:
            self._db[sta_mac] = qoe_value

    def get(self, sta_mac: str) -> Optional[float]:
        """Get QoE for a station. Returns None if not set."""
        with self._lock:
            return self._db.get(sta_mac)

    def remove(self, sta_mac: str):
        """Remove QoE entry for a station."""
        with self._lock:
            self._db.pop(sta_mac, None)

    def all(self) -> Dict[str, float]:
        """Return all QoE entries."""
        with self._lock:
            return dict(self._db)

    def count(self) -> int:
        """Number of stations with QoE recorded."""
        with self._lock:
            return len(self._db)

    def clear(self):
        """Clear all QoE entries."""
        with self._lock:
            self._db.clear()

    # -----------------------------
    # Utilities
    # -----------------------------
    def to_dict(self) -> Dict[str, float]:
        """
        Dump the complete state of the database.
        Returns a dictionary mapping station MAC addresses to their QoE value.
        """
        with self._lock:
            return dict(self._db)

    def __contains__(self, sta_mac: str) -> bool:
        with self._lock:
            return sta_mac in self._db

    def __len__(self):
        return self.count()

    def __iter__(self):
        """Iterate over all (sta_mac, qoe) pairs."""
        with self._lock:
            for sta_mac, qoe in self._db.items():
                yield sta_mac, qoe

    def __repr__(self):
        with self._lock:
            return f"<QoEDB stations={len(self._db)}>"
