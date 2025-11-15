import os
import socket
import threading
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ClientConnection:
    """Represents a connected client."""
    conn: socket.socket
    addr: str
    connected_at: datetime
    name: str  # unique identifier for this connection
    
    def close(self):
        try:
            self.conn.close()
        except:
            pass


class SocketPool:
    """Manages multiple UNIX domain sockets for IPC (server + client) with client tracking."""
    
    def __init__(self):
        # name → (socket-object, path, mode)
        self._sockets: Dict[str, Tuple[socket.socket, str, str]] = {}
        
        # server_name → list of ClientConnection objects
        self._clients: Dict[str, List[ClientConnection]] = {}
        
        # client_name → ClientConnection (for easy lookup)
        self._client_lookup: Dict[str, ClientConnection] = {}
        
        self._lock = threading.RLock()
        self._client_counter = 0

    # ------------------------------------------------------------ 
    # Create socket (server or client)
    # ------------------------------------------------------------ 
    def create(self, name: str, path: str, mode: str = "server") -> socket.socket:
        """
        Create a UNIX domain socket.
        
        Args:
            name: logical handle
            path: path on filesystem  
            mode: "server" or "client"
        """
        with self._lock:
            if name in self._sockets:
                raise ValueError(f"Socket '{name}' already exists")

            if mode == "server":
                # Clear any stale file
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.bind(path)
                sock.listen(5)
                
                # Fix permissions so non-root can connect
                try:
                    os.chmod(path, 0o666)
                except Exception as e:
                    print(f"[SocketPool] chmod failed for {path}: {e}")
                
                # Initialize client list for this server
                self._clients[name] = []
                
            elif mode == "client":
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(path)
            else:
                raise ValueError("Mode must be 'server' or 'client'")

            self._sockets[name] = (sock, path, mode)
            return sock

    def accept(self, name: str, client_name: Optional[str] = None) -> Optional[ClientConnection]:
        """
        Block until a client connects to a server socket.
        Returns a ClientConnection object that can be used to communicate with the client.
        
        Args:
            name: server socket name
            client_name: optional name for the client connection (auto-generated if None)
        """
        with self._lock:
            entry = self._sockets.get(name)
            if not entry:
                raise ValueError(f"Socket '{name}' not found")
            
            sock, path, mode = entry
            if mode != "server":
                raise ValueError(f"Socket '{name}' is not a server")
        
        # Accept outside the lock to avoid blocking other operations
        conn, addr = sock.accept()
        
        with self._lock:
            # Generate unique client name if not provided
            if client_name is None:
                self._client_counter += 1
                client_name = f"{name}_client_{self._client_counter}"
            elif client_name in self._client_lookup:
                raise ValueError(f"Client name '{client_name}' already exists")
            
            # Create client connection object
            client = ClientConnection(
                conn=conn,
                addr=str(addr),
                connected_at=datetime.now(),
                name=client_name
            )
            
            # Track the client
            self._clients[name].append(client)
            self._client_lookup[client_name] = client
            
            return client

    # ------------------------------------------------------------ 
    # Client management
    # ------------------------------------------------------------ 
    def get_client(self, client_name: str) -> Optional[ClientConnection]:
        """Get a client connection by name."""
        with self._lock:
            return self._client_lookup.get(client_name)
    
    def get_clients(self, server_name: str) -> List[ClientConnection]:
        """Get all clients connected to a server."""
        with self._lock:
            return list(self._clients.get(server_name, []))
    
    def close_client(self, client_name: str):
        """Close and remove a specific client connection."""
        with self._lock:
            client = self._client_lookup.pop(client_name, None)
            if not client:
                return
            
            # Remove from server's client list
            for server_clients in self._clients.values():
                if client in server_clients:
                    server_clients.remove(client)
                    break
            
            client.close()
    
    def broadcast(self, server_name: str, data: bytes):
        """Send data to all clients connected to a server."""
        with self._lock:
            clients = self._clients.get(server_name, [])
            dead_clients = []
            
            for client in clients:
                try:
                    client.conn.sendall(data)
                except Exception as e:
                    print(f"[SocketPool] Failed to send to {client.name}: {e}")
                    dead_clients.append(client.name)
            
            # Clean up dead connections
            for client_name in dead_clients:
                self.close_client(client_name)

    # ------------------------------------------------------------ 
    # Retrieve a socket
    # ------------------------------------------------------------ 
    def get(self, name: str) -> Optional[socket.socket]:
        with self._lock:
            entry = self._sockets.get(name)
            return entry[0] if entry else None

    # ------------------------------------------------------------ 
    # Close one socket
    # ------------------------------------------------------------ 
    def close(self, name: str):
        with self._lock:
            entry = self._sockets.pop(name, None)
            if not entry:
                return
            
            sock, path, mode = entry
            
            # Close all connected clients if this is a server
            if mode == "server" and name in self._clients:
                for client in self._clients[name]:
                    client.close()
                    self._client_lookup.pop(client.name, None)
                del self._clients[name]
            
            try:
                sock.close()
            except:
                pass

            # Only remove socket file for **server** sockets
            if mode == "server" and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

    # ------------------------------------------------------------ 
    # Destroy all sockets
    # ------------------------------------------------------------ 
    def destroy(self):
        with self._lock:
            # Close all client connections
            for client in list(self._client_lookup.values()):
                client.close()
            self._client_lookup.clear()
            self._clients.clear()
            
            # Close all sockets
            for name, (sock, path, mode) in list(self._sockets.items()):
                try:
                    sock.close()
                except:
                    pass
                if mode == "server" and os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            
            self._sockets.clear()

    # ------------------------------------------------------------ 
    # Utils
    # ------------------------------------------------------------ 
    def list(self) -> List[str]:
        with self._lock:
            return list(self._sockets.keys())
    
    def list_clients(self, server_name: Optional[str] = None) -> List[str]:
        """List all client connection names, optionally filtered by server."""
        with self._lock:
            if server_name:
                return [c.name for c in self._clients.get(server_name, [])]
            return list(self._client_lookup.keys())
    
    def stats(self) -> Dict:
        """Get statistics about the socket pool."""
        with self._lock:
            return {
                "sockets": len(self._sockets),
                "total_clients": len(self._client_lookup),
                "servers": {
                    name: len(clients) 
                    for name, clients in self._clients.items()
                }
            }

    def __len__(self):
        return len(self._sockets)

    def __repr__(self):
        return f"<SocketPool: {len(self._sockets)} sockets, {len(self._client_lookup)} clients>"