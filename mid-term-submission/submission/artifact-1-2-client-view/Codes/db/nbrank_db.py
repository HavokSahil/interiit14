from threading import RLock
from typing import Optional, Dict, List
from model.neighbor import Neighbor

DEFAULT_STA_MAC = "00:00:00:00:00:00"

class NeighborRankingDB:
    """Thread-safe singleton database for neighbor rankings per station MAC."""

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
        # sta_mac -> ordered list of Neighbor objects (ranked)
        self._db: Dict[str, List[Neighbor]] = {}
        self._initialized = True

    # -----------------------------
    # CRUD operations
    # -----------------------------
    def set_ranking(self, sta_mac: str, neighbors: List[Neighbor]):
        """Set the ordered neighbor ranking for a station."""
        if not neighbors:
            return
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC
        with self._lock:
            self._db[sta_mac] = neighbors[:]

    def add_neighbor(self, sta_mac: str, neighbor: Neighbor):
        """Add or update a neighbor in the ranking for a station (appends to the end)."""
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC
        with self._lock:
            if sta_mac not in self._db:
                self._db[sta_mac] = []
            # Remove existing neighbor if present
            self._db[sta_mac] = [n for n in self._db[sta_mac] if n.bssid != neighbor.bssid]
            self._db[sta_mac].append(neighbor)

    def get_ranking(self, sta_mac: Optional[str] = None) -> List[Neighbor]:
        """Get the ordered neighbor ranking for a station."""
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC
        with self._lock:
            return self._db.get(sta_mac, [])[:]

    def remove(self, sta_mac: Optional[str] = None):
        """Remove the ranking for a station."""
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC
        with self._lock:
            self._db.pop(sta_mac, None)

    def all(self) -> Dict[str, List[Neighbor]]:
        """Return the entire database."""
        with self._lock:
            return {sta_mac: neighbors[:] for sta_mac, neighbors in self._db.items()}

    def count(self, sta_mac: Optional[str] = None) -> int:
        """Count neighbors for a station or total if None."""
        with self._lock:
            if sta_mac:
                return len(self._db.get(sta_mac, []))
            return sum(len(ns) for ns in self._db.values())

    def clear(self, sta_mac: Optional[str] = None):
        """Clear neighbor rankings for a station or all stations."""
        with self._lock:
            if sta_mac:
                self._db.pop(sta_mac, None)
            else:
                self._db.clear()

    # -----------------------------
    # Utilities
    # -----------------------------
    def to_dict(self) -> Dict[str, list]:
        """
        Dump the entire state of the database.
        For each station MAC, returns a list of dicts (one per Neighbor, using Neighbor.to_dict() if available).
        """
        with self._lock:
            result = {}
            for sta_mac, neighbors in self._db.items():
                nb_list = []
                for nb in neighbors:
                    if hasattr(nb, "to_dict"):
                        nb_list.append(nb.to_dict())
                    else:
                        nb_list.append(vars(nb))
                result[sta_mac] = nb_list
            return result

    def __contains__(self, sta_mac: str) -> bool:
        with self._lock:
            return sta_mac in self._db

    def __len__(self):
        with self._lock:
            return sum(len(ns) for ns in self._db.values())

    def __iter__(self):
        """Iterate over all neighbors (flattened)."""
        with self._lock:
            for neighbors in self._db.values():
                for n in neighbors:
                    yield n

    def __repr__(self):
        with self._lock:
            return f"<NeighborRankingDB stations={len(self._db)} total_neighbors={len(self)}>"
