#!/usr/bin/env python3
"""Ask the running creamWafes what's going on.

    tools/creamwafes-inspect.py state              JSON: tracks, pads, audio health, focus, recent log
    tools/creamwafes-inspect.py screenshot [path]  the app saves a picture of its own window
    tools/creamwafes-inspect.py log [n]            last n log lines
"""

import os
import socket
import sys
from pathlib import Path

SOCKET = Path(os.environ.get("CREAMWAFES_SOCKET") or Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "creamwafes.sock")


def ask(command: str) -> str:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(5)
        s.connect(str(SOCKET))
        s.sendall((command + "\n").encode())
        data = b""
        while not data.endswith(b"\x04\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    return data.decode().removesuffix("\n\x04\n")


if __name__ == "__main__":
    try:
        print(ask(" ".join(sys.argv[1:]) or "state"))
    except (FileNotFoundError, ConnectionRefusedError):
        sys.exit("creamWafes isn't running (no inspect socket)")
