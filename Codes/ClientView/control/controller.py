import socket
import os
import select
import time
import threading
import queue
from control.logger import Logger
from time import sleep

class Controller:
    def __init__(self, ctrl_path="/var/run/hostapd/wlan0", iface="wlan0"):
        self.ctrl_path = ctrl_path
        self.iface = iface # by default the interface is `wlan0`
        self.sock = None
        self.local_path = f"/tmp/hostapd_ctrl_{os.getpid()}"
        self._event_queue: "queue.Queue[str]" = queue.Queue()
        self._reply_queue: "queue.Queue[str]" = queue.Queue()
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

        sleep(1) # wait for stablization: TODO: see if its required

        # just as a safety measure
        self.clear_events()
        self.clear_reply()

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
        # by default clear the reply queue
        self.clear_reply()

        try:
            # TODO: remember this debug message
            print(f"SENT: {cmd}")
            sent = self.sock.send(cmd.encode())
            if sent == 0:
                Logger.log_err("Socket connection broken")
                return False
            return True
        except Exception as e:
            Logger.log_err(f"Error sending command: {e}")
            return False

    def receive(self, timeout=3.0) -> str | None:
        """Retrieve a reply from _reply_queue, waiting up to timeout seconds."""
        try:
            return self._reply_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def receive_event(self, timeout=3.0) -> str | None:
        """Retrieve an event from _event_queue, waiting up to timeout seconds."""
        try:
            return self._event_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _reader_loop(self):
        """Background thread to read from the socket and separate replies and events."""
        while self._running:
            try:
                ready = select.select([self.sock], [], [], 0.2)
                if ready[0]:
                    data = self.sock.recv(4096)
                    if not data:
                        continue
                    msg = data.decode(errors="ignore")
                    if msg.startswith('<'):
                        # NOTE: remember this debug
                        print(f"EVENT: {" ".join(msg.splitlines())}")
                        self._event_queue.put(msg)
                    else:
                        # NOTE: remember this debug
                        print(f"REPLY: {" ".join(msg.splitlines())}")
                        self._reply_queue.put(msg)
            except Exception as e:
                Logger.log_err(f"Error in controller reader loop: {e}")
                time.sleep(0.1)

    def clear_reply(self):
        while not self._reply_queue.empty():
            try:
                self._reply_queue.get_nowait()
            except queue.Empty:
                break

    def clear_events(self):
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                break

    def repl(self):
        """
        Simple REPL shell for interacting with the controller.
        Type 'exit' or 'quit' to exit.
        """
        print("Controller REPL started. Type 'exit' or 'quit' to exit.")
        while True:
            try:
                cmd = input("controller> ").strip()
                print(f"Command: `{cmd}`")
                if not cmd:
                    continue
                if cmd.lower() in ("exit", "quit"):
                    print("Exiting controller REPL.")
                    break
                if cmd.startswith("#"):  # allow shell comments
                    continue
                sent = self.send_command(cmd)
                if sent:
                    reply = self.receive()
                    if reply is not None:
                        print(f"Reply: {reply}")
                    else:
                        print("No reply received (timeout).")
                else:
                    print("Failed to send command or command not accepted.")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting controller REPL.")
                break
            except Exception as e:
                Logger.log_err(f"REPL error: {e}")


if __name__ == "__main__":
    controller = Controller()
    controller.connect()
    controller.repl()
    controller.disconnect()