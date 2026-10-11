"""creamWafes: make sounds with local AI, play them on pads, build songs from them."""

from pathlib import Path

try:
    from importlib.metadata import version as _version
    __version__ = _version("creamwafes")
except Exception:  # noqa: BLE001 - running from a plain checkout
    __version__ = "0.1"

ASSETS = Path(__file__).resolve().parent / "assets"
