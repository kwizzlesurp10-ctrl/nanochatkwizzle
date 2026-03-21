import socket

import pytest

from nanochat.netutil import assert_tcp_port_available


def test_assert_tcp_port_available_succeeds_when_port_is_free():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert_tcp_port_available("127.0.0.1", port)


def test_assert_tcp_port_available_exits_when_port_is_held():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as hold:
        hold.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        hold.bind(("127.0.0.1", 0))
        hold.listen(1)
        port = hold.getsockname()[1]
        with pytest.raises(SystemExit) as exc_info:
            assert_tcp_port_available("127.0.0.1", port)
    assert "already in use" in str(exc_info.value)
