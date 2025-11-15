import os
import threading
from typing import Optional, Dict, IO

class PipePool:
    """Manages multiple named pipes (FIFOs) for simple IPC streaming."""

    def __init__(self):
        self._pipes: Dict[str, IO] = {}
        self._paths: Dict[str, str] = {}
        self._lock = threading.RLock()

    def create(self, name: str, path: str, mode: str = "w") -> IO:
        """
        Create or open a named pipe (FIFO).

        Args:
            name: logical name for the pipe
            path: filesystem path to the FIFO
            mode: 'w' for writer, 'r' for reader
        """
        with self._lock:
            if name in self._pipes:
                raise ValueError(f"Pipe '{name}' already exists")

            # Create FIFO if not present
            if not os.path.exists(path):
                os.mkfifo(path)

            # Open in the requested mode
            # Note: open() blocks until the other end opens the pipe
            try:
                if mode == "w":
                    f = open(path, "w", buffering=1)  # line-buffered
                elif mode == "r":
                    f = open(path, "r")
                else:
                    raise ValueError("mode must be 'r' or 'w'")
            except Exception as e:
                raise ConnectionError(f"Pipe creation failed at {path}: {e}")

            self._pipes[name] = f
            self._paths[name] = path
            return f

    def get(self, name: str) -> Optional[IO]:
        """Retrieve a pipe by name."""
        with self._lock:
            return self._pipes.get(name)

    def close(self, name: str):
        """Close a pipe and optionally remove its file."""
        with self._lock:
            f = self._pipes.pop(name, None)
            path = self._paths.pop(name, None)
            if f:
                try:
                    f.close()
                except Exception:
                    pass
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    def destroy(self):
        """Close and clean up all pipes."""
        with self._lock:
            for name, f in list(self._pipes.items()):
                try:
                    f.close()
                except Exception:
                    pass
                path = self._paths.get(name)
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
            self._pipes.clear()
            self._paths.clear()

    def list(self) -> list[str]:
        """List all active pipe names."""
        with self._lock:
            return list(self._pipes.keys())

    def __len__(self):
        return len(self._pipes)

    def __repr__(self):
        return f"<PipePool fifos={len(self._pipes)}>"
