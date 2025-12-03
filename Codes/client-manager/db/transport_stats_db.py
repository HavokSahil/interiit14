from threading import RLock
from typing import Optional, Dict
from model.transport_stats import TransportStats


class TransportStatsDB:
    """Database for storing Transport Layer statistics for each station.
    
    Unlike BeaconMeasurementDB, this does NOT use expiration.
    Stats are kept until explicitly cleared or updated.
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

        # sta_mac -> TransportStats (latest measurement)
        self._store: Dict[str, TransportStats] = {}
        self._initialized = True

    # ------------------------------------------------------------
    # Add / Update
    # ------------------------------------------------------------
    def add(self, stats: TransportStats, sta_mac: Optional[str] = None):
        """Add or update transport stats for a station.
        
        Args:
            stats: TransportStats object to store
            sta_mac: Optional MAC address (will use stats.sta_mac if not provided)
        """
        if sta_mac is None:
            sta_mac = stats.sta_mac

        with self._lock:
            self._store[sta_mac] = stats

    def set(self, sta_mac: str, stats: TransportStats):
        """Alias for add() to maintain consistency with other DBs."""
        self.add(stats, sta_mac)

    # ------------------------------------------------------------
    # Get
    # ------------------------------------------------------------
    def get(self, sta_mac: str) -> Optional[TransportStats]:
        """Get transport stats for a station.
        
        Args:
            sta_mac: Station MAC address
            
        Returns:
            TransportStats object or None if not found
        """
        with self._lock:
            return self._store.get(sta_mac)

    # ------------------------------------------------------------
    # All
    # ------------------------------------------------------------
    def all(self) -> Dict[str, TransportStats]:
        """Return all transport stats.
        
        Returns:
            Dictionary mapping MAC addresses to TransportStats
        """
        with self._lock:
            return dict(self._store)

    # ------------------------------------------------------------
    # Clear / remove
    # ------------------------------------------------------------
    def remove(self, sta_mac: str):
        """Remove transport stats for a specific station.
        
        Args:
            sta_mac: Station MAC address to remove
        """
        with self._lock:
            self._store.pop(sta_mac, None)

    def clear(self, sta_mac: Optional[str] = None):
        """Clear transport stats.
        
        Args:
            sta_mac: If provided, clear only this station. Otherwise clear all.
        """
        with self._lock:
            if sta_mac:
                self._store.pop(sta_mac, None)
            else:
                self._store.clear()

    # ------------------------------------------------------------
    # Export
    # ------------------------------------------------------------
    def to_dict(self) -> Dict[str, Dict]:
        """Export the complete state of the database.
        
        Returns:
            Dictionary mapping MAC addresses to transport stats dicts
        """
        with self._lock:
            return {
                sta_mac: stats.to_dict()
                for sta_mac, stats in self._store.items()
            }

    # ------------------------------------------------------------
    # Count
    # ------------------------------------------------------------
    def count(self) -> int:
        """Number of stations with transport stats.
        
        Returns:
            Count of stored stats
        """
        with self._lock:
            return len(self._store)

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    def __contains__(self, sta_mac: str) -> bool:
        with self._lock:
            return sta_mac in self._store

    def __len__(self):
        return self.count()

    def __iter__(self):
        """Iterate over all (sta_mac, stats) pairs."""
        with self._lock:
            for sta_mac, stats in self._store.items():
                yield sta_mac, stats

    def __repr__(self):
        with self._lock:
            return f"<TransportStatsDB stations={len(self._store)}>"
