from threading import RLock
from typing import Optional, Dict, List
from model.measurement import BSSTransitionResponse
from store.acceptance import BSSTransitionAcceptance

DEFAULT_STA_MAC = "00:00:00:00:00:00"  # Default if STA MAC is not provided

class BSSTransitionResponseDB:
    """Database for storing BSSTransitionResponse reports per station."""

    _instance = None
    _lock = RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, filename: Optional[str] = None):
        if getattr(self, "_initialized", False):
            return
        # sta_mac -> dialog_token -> BSSTransitionResponse
        self._db: Dict[str, Dict[int, "BSSTransitionResponse"]] = {}
        from datetime import datetime
        import time
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filename is None:
            filename = f"export/bssTmStat_{now_str}"
        self.filename = filename
        self._initialized = True
        self.acceptance = BSSTransitionAcceptance(self.filename)

    # -----------------------------
    # CRUD
    # -----------------------------
    def add(self, resp: "BSSTransitionResponse", sta_mac: Optional[str] = None):
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC
        self.acceptance.add(sta_mac, resp.status_code)
        with self._lock:
            if sta_mac not in self._db:
                self._db[sta_mac] = {}
            self._db[sta_mac][resp.dialog_token] = resp

    def remove(self, dialog_token: int, sta_mac: Optional[str] = None):
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC
        with self._lock:
            if sta_mac in self._db:
                self._db[sta_mac].pop(dialog_token, None)
                if not self._db[sta_mac]:
                    del self._db[sta_mac]

    def get(self, dialog_token: int, sta_mac: Optional[str] = None) -> Optional["BSSTransitionResponse"]:
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC
        with self._lock:
            return self._db.get(sta_mac, {}).get(dialog_token)

    def all_for_sta(self, sta_mac: Optional[str] = None) -> List["BSSTransitionResponse"]:
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC
        with self._lock:
            return list(self._db.get(sta_mac, {}).values())

    def all(self) -> Dict[str, List["BSSTransitionResponse"]]:
        with self._lock:
            return {sta_mac: list(resps.values()) for sta_mac, resps in self._db.items()}

    def count(self, sta_mac: Optional[str] = None) -> int:
        if sta_mac is None:
            sta_mac = DEFAULT_STA_MAC
        with self._lock:
            if sta_mac:
                return len(self._db.get(sta_mac, {}))
            return sum(len(resps) for resps in self._db.values())

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
            return sum(len(resps) for resps in self._db.values())

    def __iter__(self):
        """Iterate over all BSS Transition responses (flattened)."""
        with self._lock:
            for resps in self._db.values():
                for r in resps.values():
                    yield r

    def to_dict(self):
        """Export as nested dict: sta_mac -> dialog_token -> dict."""
        with self._lock:
            return {
                sta_mac: {token: vars(resp) for token, resp in resps.items()}
                for sta_mac, resps in self._db.items()
            }

    def __repr__(self):
        with self._lock:
            return f"<BSSTransitionResponseDB stations={len(self._db)} total_responses={len(self)}>"
