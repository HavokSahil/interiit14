from threading import RLock
from typing import Optional, Dict
from model.neighbor import Neighbor

DEFAULT_STA_MAC = "00:00:00:00:00:00"  # Default for AP-associated neighbors

class NeighborDB:
    """Database of neighbors associated with stations (STA MACs)."""

    _instance = None
    _lock = RLock()

    def __new__(cls):
        """Singleton instantiation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        # Top-level dict: sta_mac -> dict of bssid -> Neighbor
        self._db: Dict[str, Dict[str, Neighbor]] = {}
        self._initialized = True

    # -----------------------------
    # Core CRUD operations
    # -----------------------------
    def add(self, neighbor: Neighbor, sta_mac: Optional[str] = None):
        """Add or update a neighbor for a given station MAC.

        If sta_mac is None, defaults to AP MAC (00:00:00:00:00:00)
        """
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC

        with self._lock:
            if sta_mac not in self._db:
                self._db[sta_mac] = {}
            self._db[sta_mac][neighbor.bssid] = neighbor

    def remove(self, bssid: str, sta_mac: Optional[str] = None):
        """Remove a neighbor for a given station MAC."""
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC

        with self._lock:
            if sta_mac in self._db:
                self._db[sta_mac].pop(bssid, None)
                if not self._db[sta_mac]:
                    del self._db[sta_mac]

    def get(self, bssid: str, sta_mac: Optional[str] = None) -> Optional[Neighbor]:
        """Retrieve a neighbor by station MAC and BSSID."""
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC
        with self._lock:
            return self._db.get(sta_mac, {}).get(bssid)

    def all_for_sta(self, sta_mac: Optional[str] = None) -> list[Neighbor]:
        """Return all neighbors for a specific station."""
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC
        with self._lock:
            return list(self._db.get(sta_mac, {}).values())

    def all(self) -> Dict[str, list[Neighbor]]:
        """Return all neighbors grouped by station MAC."""
        with self._lock:
            return {sta_mac: list(neighs.values()) for sta_mac, neighs in self._db.items()}

    def count(self, sta_mac: Optional[str] = None) -> int:
        """Count neighbors for a station or all neighbors if sta_mac is None."""
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC
        with self._lock:
            if sta_mac:
                return len(self._db.get(sta_mac, {}))
            return sum(len(neighs) for neighs in self._db.values())

    def clear(self, sta_mac: Optional[str] = None):
        """Clear neighbors for a station or all if sta_mac is None."""
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
            return sum(len(neighs) for neighs in self._db.values())

    def __iter__(self):
        """Iterate over all neighbors (flattened)."""
        with self._lock:
            for neighs in self._db.values():
                for n in neighs.values():
                    yield n

    def to_dict(self):
        """Export neighbors as nested dict: sta_mac -> bssid -> dict."""
        with self._lock:
            return {
                sta_mac: {bssid: n.to_dict() if hasattr(n, "to_dict") else vars(n)
                          for bssid, n in neighs.items()}
                for sta_mac, neighs in self._db.items()
            }

    def __repr__(self):
        with self._lock:
            return f"<NeighborDB stations={len(self._db)} total_neighbors={len(self)}>"
