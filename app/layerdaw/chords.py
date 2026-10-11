"""The instrument behind chord and note pads: a pad's own sound, pitched to every note it needs.

The sound is whatever is on the pad (its pitch is detected so chords land in key) or, if there
isn't one, a built-in synth tone.
"""

from __future__ import annotations

import numpy as np

from .model import SR, Clip, Pad, declick, seamless_loop

A4 = 440.0


def detect_pitch(audio: np.ndarray) -> float | None:
    """Fundamental of a held sound as a (fractional) MIDI note, via YIN on the steady middle part.
    Returns None when there's no clear single pitch (drums, full mixes)."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    n = len(mono)
    frame = 4096
    if n < frame:
        return None
    lo, hi = int(SR / 1000), int(SR / 50)  # 50 Hz – 1 kHz
    estimates = []
    for start in np.linspace(int(n * 0.2), max(int(n * 0.2), n - frame - 1), 5).astype(int):
        x = mono[start:start + frame].astype(np.float64)
        x -= x.mean()
        if np.sqrt(np.mean(x ** 2)) < 1e-3:
            continue
        w = frame // 2
        d = np.array([np.sum((x[:w] - x[tau:tau + w]) ** 2) for tau in range(hi + 2)])
        cmnd = d.copy()
        cmnd[1:] = d[1:] * np.arange(1, len(d)) / np.maximum(np.cumsum(d[1:]), 1e-12)
        cmnd[0] = 1
        below = np.nonzero(cmnd[lo:hi] < 0.15)[0]
        if not len(below):
            continue
        tau = lo + below[0]
        while tau + 1 < hi and cmnd[tau + 1] < cmnd[tau]:
            tau += 1
        a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
        shift = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c else 0.0
        estimates.append(SR / (tau + shift))
    if len(estimates) < 2:
        return None
    notes = 69 + 12 * np.log2(np.array(estimates) / A4)
    ref = float(np.median(notes))
    folded = (notes - ref + 6) % 12 - 6  # octave slips (sub-harmonics) count as the same note
    if np.std(folded) > 0.5:  # pitch really wanders: not a held note
        return None
    return ref + float(np.median(folded))


def synth_note(midi: float, seconds: float = 2.0) -> np.ndarray:
    """A warm, slightly detuned saw pad: the chord sound when nothing has been loaded."""
    t = np.arange(int(seconds * SR)) / SR
    f = A4 * 2 ** ((midi - 69) / 12)
    out = np.zeros_like(t)
    for detune in (-0.06, 0.06):
        fd = f * 2 ** (detune / 12)
        for k in range(1, int(min(9000, SR / 2 - 100) // fd) + 1):
            out += np.sin(2 * np.pi * k * fd * t) / k * (1 / (1 + (k * fd / 2200) ** 2))
    out += 0.5 * np.sin(2 * np.pi * f / 2 * t)  # a little sub
    env = np.minimum(t / 0.02, 1.0) * (0.75 + 0.25 * np.exp(-t / 0.4))
    out = out * env / np.abs(out).max() * 0.5
    stereo = np.stack([out, np.roll(out, 7)], axis=1).astype(np.float32)
    return declick(stereo)


class ChordInstrument:
    def __init__(self):
        self.source: Pad | None = None
        self.root: float | None = None  # detected MIDI pitch of the source
        self._notes: dict[float, tuple[np.ndarray, tuple[int, int] | None]] = {}

    @property
    def label(self) -> str:
        from .theory import note_name
        if self.source is None:
            return "built-in synth"
        return f"{self.source.name} ({note_name(self.root)})" if self.root is not None else \
            f"{self.source.name} (pitch unclear — treated as the key's root)"

    def set_source(self, clip: Clip | None, name: str = "", root: float | None = None) -> None:
        self._notes.clear()
        if clip is None:
            self.source, self.root = None, None
            return
        self.source = Pad(clip=clip, name=name, mode="hold")
        self.root = root if root is not None else detect_pitch(clip.audio)

    def note(self, midi: int, key_root_midi: int) -> tuple[np.ndarray, tuple[int, int] | None]:
        """(buffer, loop) sounding ``midi``. A source with no clear pitch is treated as the key root."""
        if midi in self._notes:
            return self._notes[midi]
        if self.source is None:
            result = seamless_loop(synth_note(midi))
        else:
            root = self.root if self.root is not None else key_root_midi
            root += 12 * round((key_root_midi - root) / 12)  # nearest octave: small shifts sound natural
            voice = Pad(clip=self.source.clip, mode="hold", pitch=midi - root)
            result = voice.hold_buffer()
        if len(self._notes) > 64:
            self._notes.clear()
        self._notes[midi] = result
        return result


class PadInstrument:
    """Plays any pad's sound at any pitch: the engine behind chord and note pads.

    A pad's detected root is shifted to each wanted pitch class by at most ±6 semitones, so the
    sound keeps its own register (an 808 stays low, a Rhodes stays mid) instead of being dragged
    octaves away; ``pad.pitch`` moves it from there. No sound → the built-in synth, at the exact
    pitch asked for."""

    def __init__(self):
        self._cache: dict[tuple, tuple[np.ndarray, tuple[int, int] | None]] = {}

    def root_of(self, pad: Pad, key_root: int) -> float:
        if pad.clip is not None and pad.root is None and not pad.__dict__.get("_root_checked"):
            pad.root = detect_pitch(pad.clip.audio)
            pad.__dict__["_root_checked"] = True
        return pad.root if pad.root is not None else float(key_root)

    def note(self, pad: Pad, midi: int, key_root: int,
             sustain: bool | None = None) -> tuple[np.ndarray, tuple[int, int] | None]:
        sustain = pad.mode != "oneshot" if sustain is None else sustain
        if pad.clip is None:
            key = ("synth", midi + pad.pitch, sustain)
            if key not in self._cache:
                buf = synth_note(midi + pad.pitch)
                self._cache[key] = seamless_loop(buf) if sustain else (buf, None)
            return self._cache[key]
        shift = ((midi - self.root_of(pad, key_root) + 6) % 12) - 6 + pad.pitch
        key = (id(pad.clip.audio), round(float(shift), 3), sustain)
        if key not in self._cache:
            if len(self._cache) > 256:
                self._cache.clear()
            voice = Pad(clip=pad.clip, pitch=shift)
            self._cache[key] = voice.freeze_buffer() if sustain else (voice.buffer(), None)
        return self._cache[key]
