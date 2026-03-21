"""Small network helpers (keep import-light for tests)."""

from __future__ import annotations

import errno
import socket


def assert_tcp_port_available(host: str, port: int) -> None:
    """Raise SystemExit if host:port cannot be bound for TCP listen."""
    bind_host = host.strip() or "0.0.0.0"
    family = (
        socket.AF_INET6
        if (":" in bind_host and not bind_host.startswith("["))
        else socket.AF_INET
    )
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((bind_host, port))
    except OSError as e:
        if e.errno in (errno.EADDRINUSE, 10048):
            raise SystemExit(
                f"Port {port} on {bind_host!r} is already in use. "
                f"Use another port (e.g. --port 8011) or free it: "
                f"ss -ltnp | grep :{port}   (or sudo ss -ltnp)"
            ) from e
        raise
