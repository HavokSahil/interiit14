from typing import Optional, Dict
from threading import RLock
from model.station import Station

class StationDB:
    """Singleton database for managing Station objects."""

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
        self._stations: Dict[str, Station] = {}
        self._initialized = True

    # ----------------------------
    # Core CRUD operations
    # ----------------------------
    def add(self, station: Station):
        """Add or update a Station entry by MAC address."""
        if not station.mac:
            raise ValueError("Station must have a valid MAC address.")
        with self._lock:
            self._stations[station.mac.lower()] = station

    def get(self, mac: str) -> Optional[Station]:
        """Retrieve a station by MAC address."""
        with self._lock:
            return self._stations.get(mac.lower())

    def remove(self, mac: str):
        """Remove a Station entry."""
        with self._lock:
            self._stations.pop(mac.lower(), None)

    def all(self) -> list[Station]:
        """Return all Station entries."""
        with self._lock:
            return list(self._stations.values())

    def list(self) -> list[str]:
        """Return list of MAC addresses for all stations."""
        with self._lock:
            return list(self._stations.keys())

    def count(self) -> int:
        """Return total number of stations."""
        with self._lock:
            return len(self._stations)

    def clear(self):
        """Remove all Station entries."""
        with self._lock:
            self._stations.clear()

    # ----------------------------
    # Update and export
    # ----------------------------
    def update(self, mac: str, info: dict):
        """Update an existing station's info_dict or attributes."""
        with self._lock:
            sta = self._stations.get(mac.lower())
            if not sta:
                sta = Station()
                sta.mac = mac
                self._stations[mac.lower()] = sta

            sta.info_dict.update(info)
            for k, v in info.items():
                if hasattr(sta, k):
                    setattr(sta, k, v)

    def to_dict(self) -> Dict[str, dict]:
        """
        Dump the complete state of the database.
        Returns a dictionary mapping lowercased station MAC addresses to
        their dictionary representations (calls Station.to_dict() if available, else vars()).
        """
        with self._lock:
            out = {}
            for mac, sta in self._stations.items():
                if hasattr(sta, "to_dict"):
                    out[mac] = sta.to_dict()
                else:
                    out[mac] = vars(sta)
            return out

    def __contains__(self, mac: str) -> bool:
        return mac.lower() in self._stations

    def __len__(self):
        return len(self._stations)

    def __iter__(self):
        return iter(self._stations.values())

    def __repr__(self):
        return f"<StationDB entries={len(self._stations)}>"
