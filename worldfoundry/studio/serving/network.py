from __future__ import annotations

import socket


def get_external_ip() -> str:
    """Return the outward-facing local IP address for this host."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def public_url_for_bind(host: str, port: int) -> str:
    """Return a useful browser URL for a service bind address."""

    visible_host = get_external_ip() if host in {"", "0.0.0.0", "::"} else host
    return f"http://{visible_host}:{int(port)}"
