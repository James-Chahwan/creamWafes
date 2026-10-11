"""Per-track sound shaping (warmth, tone, cuts, pitch, punch, ducking, width, delay, reverb).

Non-destructive: a track's settings live in ``Track.fx`` and are applied when the track is mixed;
its audio files never change. Results are cached per track, so only the track you're shaping is
reprocessed. Order: pitch → filters/tone → warmth → punch → duck → width → delay → reverb.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .model import SR

DEFAULTS = {"warmth": 0.0, "tone": 0.0, "low_cut": 20.0, "high_cut": 20000.0, "pitch": 0, "punch": 0.0,
            "duck": 0.0, "duck_to": "", "width": 1.0, "delay": 0.0, "delay_time": "1/8", "reverb": 0.0,
            "size": 0.5, "bypass": False}
DELAY_BEATS = {"1/16": 0.25, "1/8": 0.5, "dotted 1/8": 0.75, "1/4": 1.0, "1/2": 2.0}
PRESETS = {
    "Clean (reset)": {},
    "Warm & dusty": {"warmth": 0.55, "tone": -0.35, "high_cut": 9000.0, "reverb": 0.12, "size": 0.3},
    "Lo-fi radio": {"warmth": 0.7, "low_cut": 350.0, "high_cut": 3800.0, "width": 0.3},
    "Underwater": {"high_cut": 900.0, "reverb": 0.3, "size": 0.6, "tone": -0.5},
    "Big room": {"reverb": 0.38, "size": 0.75, "punch": 0.3},
    "Wide & airy": {"tone": 0.35, "width": 1.6, "reverb": 0.22, "size": 0.55, "low_cut": 120.0},
    "Punchy drums": {"punch": 0.7, "warmth": 0.25, "low_cut": 30.0},
    "Pumping (duck to kick)": {"duck": 0.75},
    "Echo throw": {"delay": 0.45, "delay_time": "dotted 1/8", "reverb": 0.15},
    "Dark & distant": {"tone": -0.6, "high_cut": 5000.0, "reverb": 0.45, "size": 0.85, "width": 1.3},
}


def settings(fx: dict) -> dict:
    return {**DEFAULTS, **(fx or {})}


def is_active(fx: dict) -> bool:
    s = settings(fx)
    return not s["bypass"] and any(s[k] != v for k, v in DEFAULTS.items() if k not in ("bypass", "duck_to", "size",
                                                                                         "delay_time"))


def describe(fx: dict) -> str:
    """'warm 40% · reverb 30% · +2 st' — what's on a track, in a few words."""
    s = settings(fx)
    bits = []
    for key, text in (("warmth", "warm"), ("punch", "punch"), ("duck", "duck"), ("delay", "echo"),
                      ("reverb", "reverb")):
        if s[key]:
            bits.append(f"{text} {s[key]:.0%}")
    if s["tone"]:
        bits.append(f"{'brighter' if s['tone'] > 0 else 'darker'} {abs(s['tone']):.0%}")
    if s["low_cut"] > DEFAULTS["low_cut"]:
        bits.append(f"low cut {s['low_cut']:.0f} Hz")
    if s["high_cut"] < DEFAULTS["high_cut"]:
        bits.append(f"high cut {s['high_cut'] / 1000:.1f} kHz")
    if s["pitch"]:
        bits.append(f"{s['pitch']:+d} st")
    if s["width"] != 1.0:
        bits.append("mono" if s["width"] == 0 else f"width {s['width']:.0%}")
    return ("(bypassed) " if s["bypass"] else "") + " · ".join(bits)


def pitch_available() -> bool:
    return shutil.which("rubberband") is not None


# --- the building blocks --------------------------------------------------------------------------

def _shelf(f0: float, gain_db: float, high: bool) -> np.ndarray:
    """RBJ shelving biquad as one SOS section."""
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    cos, alpha = np.cos(w0), np.sin(w0) / 2 * np.sqrt(2)
    sa = 2 * np.sqrt(a) * alpha
    sign = -1 if high else 1
    b0 = a * ((a + 1) - sign * (a - 1) * cos + sa)
    b1 = 2 * sign * a * ((a - 1) - sign * (a + 1) * cos)
    b2 = a * ((a + 1) - sign * (a - 1) * cos - sa)
    a0 = (a + 1) + sign * (a - 1) * cos + sa
    a1 = -2 * sign * ((a - 1) + sign * (a + 1) * cos)
    a2 = (a + 1) + sign * (a - 1) * cos - sa
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _filter(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return signal.sosfilt(sos, x, axis=0).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def _match(y: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Same loudness as ``like``, so turning a knob compares the character, not the level."""
    ry, rx = _rms(y), _rms(like)
    return y * (rx / ry) if ry > 1e-9 and rx > 1e-9 else y


BLOCK = 64


def _envelope(mono: np.ndarray, attack: float, release: float) -> np.ndarray:
    """Peak follower per BLOCK samples (attack/release in seconds)."""
    n = len(mono) // BLOCK
    peaks = np.abs(mono[: n * BLOCK]).reshape(n, BLOCK).max(axis=1) if n else np.zeros(0)
    ga, gr = np.exp(-BLOCK / (attack * SR)), np.exp(-BLOCK / (release * SR))
    env, e = np.empty(n), 0.0
    for k, p in enumerate(peaks.tolist()):
        g = ga if p > e else gr
        e = g * e + (1 - g) * p
        env[k] = e
    return env


def _per_sample(block_gain: np.ndarray, frames: int) -> np.ndarray:
    if not len(block_gain):
        return np.ones(frames, np.float32)
    centres = np.arange(len(block_gain)) * BLOCK + BLOCK / 2
    return np.interp(np.arange(frames), centres, block_gain).astype(np.float32)


def _pitch(x: np.ndarray, semitones: int) -> np.ndarray:
    """Shift pitch, keep the length (Rubber Band; without it, the audio is returned unchanged)."""
    exe = shutil.which("rubberband")
    if not exe or not semitones or len(x) < 2048:
        return x
    with tempfile.TemporaryDirectory(prefix="layerdaw-fx-") as tmp:
        src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
        sf.write(src, x, SR, subtype="FLOAT")
        subprocess.run([exe, "-q", "-p", str(semitones), str(src), str(dst)], check=True, timeout=600,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        y, _ = sf.read(dst, dtype="float32", always_2d=True)
    out = np.zeros_like(x)
    out[: min(len(x), len(y))] = y[: len(x)]
    return out


def _warmth(x: np.ndarray, amount: float) -> np.ndarray:
    """Tape/tube-ish: soft asymmetric saturation (even harmonics), a bit more low end, softer top."""
    drive, bias = 1 + 5 * amount, 0.15 * amount
    y = np.tanh(drive * x + bias) - np.tanh(bias)
    y = _filter(y, np.vstack([_shelf(140, 2.5 * amount, False), _shelf(7000, -4 * amount, True)]))
    return _match(y, x)


def _punch(x: np.ndarray, amount: float) -> np.ndarray:
    """Transient shaper (attacks up) + gentle compression of the body, level-matched."""
    mono = x.mean(axis=1)
    fast, slow = _envelope(mono, 0.001, 0.03), _envelope(mono, 0.02, 0.25)
    transient = np.clip(fast / (slow + 1e-6) - 1, 0, 3)
    level = 20 * np.log10(slow + 1e-6)
    threshold = np.percentile(level[level > -60], 85) - 6 if np.any(level > -60) else -20
    ratio = 1 + 3 * amount
    gain_db = np.minimum(9.0, 6 * amount * transient) - np.maximum(0, level - threshold) * (1 - 1 / ratio)
    y = x * _per_sample(10 ** (gain_db / 20), len(x))[:, None]
    return _match(y, x)


def _duck(x: np.ndarray, side: np.ndarray, amount: float, beat_s: float) -> np.ndarray:
    """Pump: dip this track each time the sidechain (the kick) hits, recovering within the beat.
    Keyed off the sidechain's low end, so hats and snares in the same track don't trigger it."""
    low = signal.sosfilt(signal.butter(2, 150, "lowpass", fs=SR, output="sos"), side.mean(axis=1))
    env = _envelope(low, 0.002, max(0.05, 0.2 * beat_s))
    top = np.percentile(env, 99.5) if len(env) else 0
    if top <= 1e-6:
        return x
    gain = 1 - 0.85 * amount * np.clip(env / top, 0, 1)
    ahead = int(0.01 * SR / BLOCK)  # look 10 ms ahead (we're offline): the dip lands on the kick, not after it
    gain = np.r_[gain[ahead:], np.ones(ahead)]
    return x * _per_sample(gain, len(x))[:, None]


def _width(x: np.ndarray, width: float) -> np.ndarray:
    mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2 * width
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)


def _delay(x: np.ndarray, amount: float, beats: float, beat_s: float) -> np.ndarray:
    """Tempo-synced ping-pong echoes, darkened as they repeat."""
    d = max(1, int(beats * beat_s * SR))
    fb = 0.3 + 0.3 * amount
    taps = int(min(12, max(1, np.ceil(np.log(0.02) / np.log(fb)))))
    out = np.zeros((len(x) + d * taps, 2), np.float32)
    out[: len(x)] = x
    wet = np.zeros_like(out)
    for k in range(1, taps + 1):
        pan = np.array([1.0, 0.35] if k % 2 else [0.35, 1.0], np.float32)
        wet[k * d: k * d + len(x)] += x * (fb ** (k - 1)) * pan
    wet = signal.sosfilt(signal.butter(2, 4500, "lowpass", fs=SR, output="sos"), wet, axis=0)
    return (out + wet * 0.65 * amount).astype(np.float32)


def _reverb(x: np.ndarray, amount: float, size: float) -> np.ndarray:
    """Convolution with a synthetic stereo room (decaying noise, darker as it grows)."""
    rt = 0.5 + 3.0 * size
    n = int(rt * SR)
    t = np.arange(n) / SR
    ir = np.random.default_rng(7).standard_normal((n, 2)) * np.exp(-6.9 * t / rt)[:, None]
    ir = signal.sosfilt(signal.butter(1, 7000 - 4000 * size, "lowpass", fs=SR, output="sos"), ir, axis=0)
    ir = np.vstack([np.zeros((int(0.012 * SR), 2)), ir])
    ir /= np.sqrt((ir ** 2).sum(axis=0))
    wet = np.stack([signal.oaconvolve(x[:, c], ir[:, c]) for c in range(2)], axis=1)
    out = np.zeros_like(wet, dtype=np.float32)
    out[: len(x)] = x * (1 - 0.3 * amount)
    return (out + wet * 0.55 * amount).astype(np.float32)


def process(x: np.ndarray, fx: dict, *, beat_s: float, sidechain: np.ndarray | None = None,
            pitched: np.ndarray | None = None) -> np.ndarray:
    """Run the whole chain on ``x`` (frames × 2). May return a longer buffer (echo/reverb tails).
    ``pitched``: ``x`` already pitch-shifted (the slow step, cached by the caller)."""
    s = settings(fx)
    y = pitched if pitched is not None else _pitch(x, int(s["pitch"]))
    sos = []
    if s["low_cut"] > DEFAULTS["low_cut"]:
        sos.append(signal.butter(2, s["low_cut"], "highpass", fs=SR, output="sos"))
    if s["high_cut"] < DEFAULTS["high_cut"]:
        sos.append(signal.butter(2, s["high_cut"], "lowpass", fs=SR, output="sos"))
    if s["tone"]:
        sos += [_shelf(250, -6 * s["tone"], False), _shelf(3000, 6 * s["tone"], True)]
    if sos:
        y = _filter(y, np.vstack(sos))
    if s["warmth"]:
        y = _warmth(y, s["warmth"])
    if s["punch"]:
        y = _punch(y, s["punch"])
    if s["duck"] and sidechain is not None:
        y = _duck(y, sidechain, s["duck"], beat_s)
    if s["width"] != 1.0:
        y = _width(y, s["width"])
    if s["delay"]:
        y = _delay(y, s["delay"], DELAY_BEATS.get(s["delay_time"], 0.5), beat_s)
    if s["reverb"]:
        y = _reverb(y, s["reverb"], s["size"])
    return np.asarray(y, np.float32)


# --- per-track rendering with a cache ----------------------------------------------------------------

_cache: OrderedDict = OrderedDict()   # key → (start frame, processed audio)
_pitch_cache: OrderedDict = OrderedDict()
_lock = threading.Lock()              # the UI renders in the background too


def _remember(store: OrderedDict, key, value, keep: int = 24):
    with _lock:
        store[key] = value
        store.move_to_end(key)
        while len(store) > keep:
            store.popitem(last=False)
    return value


def _recall(store: OrderedDict, key):
    with _lock:
        if key in store:
            store.move_to_end(key)
            return store[key]
    return None


def clips_key(track) -> tuple:
    return tuple((str(c.path), round(c.start, 6), round(c.offset, 6), round(c.length, 6)) for c in track.clips)


def raw(track) -> tuple[int, np.ndarray]:
    """The track's clips summed, starting at its first clip (start frame, audio); no volume/pan/fx."""
    if not track.clips:
        return 0, np.zeros((0, 2), np.float32)
    first = int(round(min(c.start for c in track.clips) * SR))
    end = int(round(max(c.end for c in track.clips) * SR))
    out = np.zeros((max(0, end - first), 2), np.float32)
    for c in track.clips:
        a = int(round(c.start * SR)) - first
        audio = c.audio
        n = min(len(audio), len(out) - a)
        if n > 0:
            out[a:a + n] += audio[:n]
    return first, out


def sidechain_source(project, track, name: str | None = None):
    """The track to duck under: the one picked by name, else the kick (by name), else the drums.
    ``name`` overrides the track's own pick ("" = what Auto would choose)."""
    name = settings(track.fx)["duck_to"] if name is None else name
    others = [t for t in project.tracks if t is not track and t.clips]
    if name:
        return next((t for t in others if t.name == name), None)
    return (next((t for t in others if "kick" in t.name.lower()), None)
            or next((t for t in others if t.kind in ("drums", "kick")), None))


def _key(project, track) -> tuple:
    s = settings(track.fx)
    side_track = sidechain_source(project, track) if s["duck"] else None
    return (id(track), clips_key(track), tuple(sorted(s.items())), project.bpm,
            clips_key(side_track) if side_track else None)


def ready(project, track) -> bool:
    """Nothing to work out for this track (no fx, or already done)."""
    return not is_active(track.fx) or _recall(_cache, _key(project, track)) is not None


def output(project, track, compute: bool = True) -> tuple[int, np.ndarray] | None:
    """The track as heard: (start frame, audio) with its fx (cached). Volume/pan are applied by the mix.
    With ``compute=False`` a track whose fx aren't worked out yet gives None instead of waiting."""
    if not is_active(track.fx):
        return raw(track)
    key = _key(project, track)
    done = _recall(_cache, key)
    if done is not None or not compute:
        return done
    s = settings(track.fx)
    side_track = sidechain_source(project, track) if s["duck"] else None
    start, audio = raw(track)
    side = None
    if side_track is not None:
        s_start, s_audio = raw(side_track)
        side = np.zeros_like(audio)  # the sidechain, lined up with this track
        a, b = max(start, s_start), min(start + len(audio), s_start + len(s_audio))
        if b > a:
            side[a - start:b - start] = s_audio[a - s_start:b - s_start]
    pkey = (clips_key(track), int(s["pitch"]))
    pitched = _recall(_pitch_cache, pkey)
    if pitched is None and s["pitch"]:
        pitched = _remember(_pitch_cache, pkey, _pitch(audio, int(s["pitch"])), keep=8)
    result = process(audio, track.fx, beat_s=project.beat_seconds, sidechain=side, pitched=pitched)
    return _remember(_cache, key, (start, result))
