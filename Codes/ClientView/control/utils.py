"""Utility functions for converting between different data types."""

def bytes_to_hex(b: bytes):
    """Convert bytes to hex string."""
    return b.hex()

def hex_to_bytes(h: str):
    """Convert hex string to bytes."""
    return bytes.fromhex(h)

def str_to_hex(s: str):
    """Convert string to hex string."""
    return s.encode("utf-8").hex()

def hex_to_str(h: str):
    """Convert hex string to string."""
    return bytes.fromhex(h).decode("utf-8", errors="ignore")

def bytes_to_str(b: bytes):
    """Convert bytes to string."""
    return b.decode("utf-8", errors="ignore")

def str_to_bytes(s: str):
    """Convert string to bytes."""
    return s.encode("utf-8")    