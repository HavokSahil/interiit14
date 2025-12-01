from model.mac_address import MacAddress
from typing import Dict, List, Optional, Any
from enum import IntEnum
import json
import threading
import os

class BSSTransitionResponseStatus(IntEnum):
    ACCEPTED = 0
    REJECT_UNSPECIFIED = 1
    REJECT_TS_DELAY_TOO_SHORT = 2
    REJECT_STA_POLICY = 3
    REJECT_AP_POLICY = 4
    REJECT_STA_BUSY = 5
    REJECT_INSUFFICIENT_RESOURCES = 6
    REJECT_OTHER = 7

    def description(self) -> str:
        """Return a human-readable description of the status."""
        descriptions = {
            self.ACCEPTED: "Accepted ... STA agrees to transition",
            self.REJECT_UNSPECIFIED: "Rejected ... Unspecified reason",
            self.REJECT_TS_DELAY_TOO_SHORT: "Rejected ... Transition delay too short",
            self.REJECT_STA_POLICY: "Rejected ... STA policy prevents transition",
            self.REJECT_AP_POLICY: "Rejected ... AP policy prevents transition",
            self.REJECT_STA_BUSY: "Rejected ... STA currently busy",
            self.REJECT_INSUFFICIENT_RESOURCES: "Rejected ... Insufficient resources on target BSS",
            self.REJECT_OTHER: "Rejected ... Other reason",
        }
        return descriptions.get(self, "Unknown status")


class BSSTransitionAcceptance:
    """
    Singleton class that accepts and persists BSS Transition response statuses for each station.
    """
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, persist_path: Optional[str] = None):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super(BSSTransitionAcceptance, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, persist_path: Optional[str] = None) -> None:
        # Initialize only once in singleton
        if getattr(self, "_initialized", False):
            return
        self._lock = threading.Lock()
        self.info: Dict[str, List[int]] = dict()  # mac as string, status as int for JSONability
        self.persist_path = persist_path or "bss_transition_acceptance.json"
        self._load()
        self._initialized = True

    def add(self, mac: MacAddress, status: int) -> None:
        """
        Add a response status for a MAC. Will also persist immediately (incremental persistence).
        Args:
            mac: MacAddress (object or string)
            status: BSSTransitionResponseStatus
        """
        mac_str = str(mac)
        with self._lock:
            if mac_str not in self.info:
                self.info[mac_str] = []
            self.info[mac_str].append(int(status))
            self._save()

    def get(self, mac: MacAddress) -> List[int]:
        """
        Get the list of response statuses for a MAC address.
        Returns: List of BSSTransitionResponseStatus
        """
        mac_str = str(mac)
        with self._lock:
            return [BSSTransitionResponseStatus(s) for s in self.info.get(mac_str, [])]

    def to_dict(self) -> Dict[str, List[int]]:
        """Return the full mapping mac->list-of-status as JSON-serializable dict."""
        with self._lock:
            return dict(self.info)

    def save(self, path: Optional[str] = None) -> None:
        """Save the current info dict to file, overwriting it."""
        with self._lock:
            self._save(path)

    def _save(self, path: Optional[str] = None) -> None:
        """Internal, thread-safe save-to-path."""
        path = path or self.persist_path
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.info, f, indent=2)
            os.replace(tmp_path, path)
        except Exception as e:
            # Logging or error handling can be added here as needed
            pass

    def load(self, path: Optional[str] = None) -> None:
        """Manually reload from file."""
        with self._lock:
            self._load(path)

    def _load(self, path: Optional[str] = None) -> None:
        """Internal, thread-safe load-from-path."""
        path = path or self.persist_path
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    # Defensive: clean up to ensure int-list
                    self.info = {mac: [int(s) for s in statuses] for mac, statuses in raw.items()}
            except Exception as e:
                self.info = dict()
        else:
            self.info = dict()

    def __len__(self):
        with self._lock:
            return sum(len(v) for v in self.info.values())

    def __contains__(self, mac: MacAddress) -> bool:
        mac_str = str(mac)
        with self._lock:
            return mac_str in self.info

    def __getitem__(self, mac: MacAddress) -> List[int]:
        return self.get(mac)

    def __repr__(self):
        with self._lock:
            return f"<BSSTransitionAcceptance size={len(self)}>"
