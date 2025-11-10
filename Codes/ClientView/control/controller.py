import socket
import struct
import os
import binascii
import select
import subprocess
import time
import re
import threading
import queue

class Controller:
    def __init__(self, ctrl_path="/var/run/hostapd/wlan0"):
        self.ctrl_path = ctrl_path
        self.sock = None
        self.local_path = f"/tmp/hostapd_ctrl_{os.getpid()}"
        self._recv_queue: "queue.Queue[str]" = queue.Queue()
        self._reader_thread: "threading.Thread | None" = None
        self._running = False

    def connect(self):
        """Connect to the controller."""
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        if os.path.exists(self.local_path):
            os.unlink(self.local_path)
        self.sock.bind(self.local_path)
        self.sock.connect(self.ctrl_path)
        # Attach and start background reader
        self.send_command("ATTACH")
        self.start_read()
        print(f"[+] Connected to hostapd: {self.ctrl_path}")

    def start_read(self):
        self._running = True
        self._reader_thread = threading.Thread(target=self._reader_loop, name="ControllerReader", daemon=True)
        self._reader_thread.start()

    def stop_read(self):
        # Stop reader thread
        self._running = False
        if self._reader_thread and self._reader_thread.is_alive():
            try:
                self._reader_thread.join(timeout=1.0)
            except Exception:
                pass

    def disconnect(self):
        """Disconnect from the controller."""
        if self.sock:
            try:
                self.send_command("DETACH")
            except Exception:
                pass
            
            self.stop_read()
            self.sock.close()
            if os.path.exists(self.local_path):
                os.unlink(self.local_path)
        print("[+] Disconnected from hostapd")

    def send_command(self, cmd, timeout=3.0) -> bool:
        """Send a command to the controller without reading a response."""
        if not self.sock:
            raise RuntimeError("Controller is not connected")
        self.sock.send(cmd.encode())
        print(f"SENT: {cmd}")
        resp = self.receive(timeout=1.0)
        print(f"Response: {resp}")
        return resp == "OK"

    def receive(self, timeout: float | None = None) -> str | None:
        """Receive the next message from the background reader queue."""
        try:
            return self._recv_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _reader_loop(self):
        """Continuously read from socket and enqueue messages."""
        while self._running and self.sock:
            try:
                ready = select.select([self.sock], [], [], 0.5)
                if not ready[0]:
                    continue
                resp = self.sock.recv(8192).decode("utf-8", errors="ignore")
                if not resp:
                    continue
                message = resp.strip()
                if message:
                    self._recv_queue.put(message)
                    #print(f"INCOMING: {message}")
            except (OSError, ValueError):
                # Socket might be closed during shutdown; exit loop
                break

    def shell(self):
        """ Run the Controller as a shell. """
        print("Welcome to the Controller shell. Type 'exit' or 'quit' to quit.")
        while True:
            cmd = input("> ")
            if cmd == "exit" or cmd == "quit":
                break
            self.send_command(cmd)
            print(self.receive(timeout=3.0))


# The main function to run the Controller as a shell.
if __name__ == "__main__":
    controller = Controller()
    controller.connect()
    controller.shell()
    controller.disconnect()