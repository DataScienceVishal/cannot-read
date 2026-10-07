"""The no-model-calls rule, as a test rather than as a sentence in the README.

`--disable-socket` is in `addopts`, so this passes because pytest-socket is
switched on and not because nothing here happens to open a connection. Without
the flag, a suite that makes no network calls and a suite that cannot make them
would look the same.
It matters more here than in most repositories, because the code still to come
fetches benchmark files and scorers, and no test may be what fetches them.
"""

from __future__ import annotations

import socket

import pytest


def test_opening_a_socket_is_a_test_failure() -> None:
    with pytest.raises(BaseException) as raised:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    assert "SocketBlocked" in type(raised.value).__name__
