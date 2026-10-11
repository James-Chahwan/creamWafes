"""Playback: the song is pre-mixed into one buffer and streamed out through PortAudio, with
live one-shot voices (pads, chords) mixed on top so they sound instantly, even when stopped."""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from .model import SR

MAX_VOICES = 48
RELEASE = int(0.12 * SR)  # fade-out after letting go of a held pad
ATTACK = int(0.001 * SR)
# 256 frames + latency="low" stalls PortAudio→ALSA→PipeWire (measured: 0.13 s of audio per 1.5 s).
# 512 + "low" runs clean at ~11 ms; the fallback is for devices that still struggle.
STREAM_SETTINGS = [dict(blocksize=512, latency="low"), dict(blocksize=1024, latency="high")]


@dataclass
class Note:
    """One sound to start: a one-shot, a held ``gate`` note (plays straight on through ``buf`` until
    released), or a ``loop`` note (repeats a region until released). ``delay`` staggers strums."""

    buf: np.ndarray
    gain: float = 1.0
    loop: tuple[int, int] | None = None
    delay: float = 0.0
    gate: bool = False

    @property
    def sustains(self) -> bool:
        return self.gate or self.loop is not None


class Voice:
    """A sounding Note. Key-up fades sustaining notes over 120 ms; one-shots play out."""

    __slots__ = ("buf", "pos", "gain", "tag", "loop", "gate", "release_left")

    def __init__(self, note: Note, tag=None):
        self.buf, self.gain, self.tag, self.loop, self.gate = note.buf, note.gain, tag, note.loop, note.gate
        self.pos = -int(note.delay * SR)
        self.release_left: int | None = None

    def release(self) -> None:
        if (self.loop is not None or self.gate) and self.release_left is None:
            self.release_left = RELEASE

    def render(self, frames: int) -> tuple[np.ndarray, bool]:
        """Next ``frames`` of audio, and whether the voice has finished."""
        block = np.zeros((frames, 2), np.float32)
        i = 0
        if self.pos < 0:  # still waiting out a strum delay
            i = min(frames, -self.pos)
            self.pos += i
        holding = self.loop is not None and self.release_left is None
        while i < frames:
            if holding and self.pos >= self.loop[1]:
                self.pos = self.loop[0]
            end = self.loop[1] if holding else len(self.buf)
            n = min(frames - i, end - self.pos)
            if n <= 0:
                break
            seg = self.buf[self.pos:self.pos + n]
            if self.pos < ATTACK:  # gate notes read straight from the file: soften the very first ms
                ramp = np.minimum(1.0, (np.arange(self.pos, self.pos + n) + 1) / ATTACK).astype(np.float32)
                seg = seg * ramp[:, None]
            block[i:i + n] = seg
            self.pos += n
            i += n
        block *= self.gain
        if self.release_left is not None:
            ramp = np.clip((self.release_left - np.arange(frames)) / RELEASE, 0.0, 1.0).astype(np.float32)
            block *= ramp[:, None]
            self.release_left -= frames
        finished = (i < frames and not holding) or (self.release_left is not None and self.release_left <= 0)
        return block, finished


def render_note(note: Note, held_frames: int) -> np.ndarray:
    """Offline version of a pad press held for ``held_frames`` (used when recording a take)."""
    voice, chunks, done = Voice(Note(note.buf, note.gain, note.loop, 0.0, note.gate)), [], False
    remaining = max(1, held_frames)
    while remaining > 0 and not done:
        block, done = voice.render(min(4096, remaining))
        chunks.append(block)
        remaining -= len(block)
    voice.release()
    while not done:  # one-shots just play out; held notes fade
        block, done = voice.render(4096)
        chunks.append(block)
    out = np.concatenate(chunks)
    nz = np.flatnonzero(np.abs(out).max(axis=1) > 0)
    return out[:nz[-1] + 1] if len(nz) else out[:1]


class _SilentStream:
    """Stand-in for tests (CREAMWAFES_NO_AUDIO=1): nothing reaches the speakers."""

    latency, blocksize, active = 0.0, 0, True

    def __init__(self, **_kw):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    abort = stop

    def close(self):
        pass


class Player:
    """Plays a stereo buffer with an optional loop region, plus triggered voices.

    The audio callback only reads ``self._mix`` / ``self._loop``; the UI thread swaps them
    wholesale (atomic under the GIL). New voices go through a deque, which is thread-safe for
    append/popleft, so the callback owns its voice list outright.
    """

    def __init__(self):
        self._mix = np.zeros((0, 2), dtype=np.float32)
        self._loop: tuple[int, int] | None = None
        self._pos = 0
        self.playing = False
        self._peaks = np.zeros(2, np.float32)  # max |sample| per channel since last read_peaks()
        self._incoming: deque = deque()
        self._voices: list[Voice] = []
        self._cut: deque = deque()       # tags whose voices stop dead (choke)
        self._released: deque = deque()  # tags whose held voices should fade out
        self.callbacks = 0              # bumped by every audio callback; the UI watches it for stalls
        self._setting = 0
        self._open()

    def _open(self) -> None:
        stream = _SilentStream if os.environ.get("CREAMWAFES_NO_AUDIO") else sd.OutputStream
        self._stream = stream(samplerate=SR, channels=2, dtype="float32", callback=self._callback,
                              **STREAM_SETTINGS[self._setting])
        self._stream.start()
        self._stream_settings = STREAM_SETTINGS[self._setting]
        self._last_seen = (self.callbacks, time.monotonic())

    def check_alive(self, now: float) -> str | None:
        """Call ~every UI tick. If the audio callback hasn't run for 0.5 s, reopen the stream
        (falling back to safer settings). Returns a message when it had to recover."""
        if os.environ.get("CREAMWAFES_NO_AUDIO"):
            return None
        count, since = self._last_seen
        if self.callbacks != count:
            self._last_seen = (self.callbacks, now)
            return None
        if now - since < 0.5:
            return None
        try:
            self._stream.abort()
            self._stream.close()
        except Exception:  # noqa: BLE001 - already dead
            pass
        # A one-off hiccup (headset reconnect) retries the fast setting; stalling again within
        # 10 s means this device can't take it, so step down to the safer one.
        if now - getattr(self, "_last_restart", -1e9) < 10.0:
            self._setting = min(self._setting + 1, len(STREAM_SETTINGS) - 1)
        self._last_restart = now
        self._open()
        return f"Audio stalled — restarted the output ({STREAM_SETTINGS[self._setting]})"

    def set_mix(self, mix: np.ndarray) -> None:
        self._mix = mix

    def set_loop(self, start: float | None, end: float | None) -> None:
        self._loop = None if start is None or end is None or end <= start else (int(start * SR), int(end * SR))

    @property
    def position(self) -> float:
        return self._pos / SR

    def seek(self, seconds: float) -> None:
        self._pos = max(0, int(seconds * SR))

    def play(self) -> None:
        if self._loop and not (self._loop[0] <= self._pos < self._loop[1]):
            self._pos = self._loop[0]
        self.playing = True

    def stop(self) -> None:
        self.playing = False

    def play_note(self, note: Note, tag=None, choke: bool = False) -> None:
        """Start ``note`` now. ``choke`` first stops any voice already playing with the same tag."""
        if choke and tag is not None:
            self._cut.append(tag)
        self._incoming.append(Voice(note, tag))

    def trigger(self, buffer: np.ndarray, gain: float = 1.0, tag=None, choke: bool = False) -> None:
        """Start a plain one-shot."""
        self.play_note(Note(buffer, gain), tag, choke)

    def release(self, tag) -> None:
        """Let go of a held voice: it fades out (one-shots just finish on their own)."""
        self._released.append(tag)

    @property
    def voice_count(self) -> int:
        return len(self._voices) + len(self._incoming)

    def read_peaks(self) -> np.ndarray:
        """Peak level per channel since the last call (pre-clip, so >1.0 means clipping)."""
        peaks, self._peaks = self._peaks, np.zeros(2, np.float32)
        return peaks

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()

    def _callback(self, out, frames, _time, _status) -> None:
        self.callbacks += 1
        out.fill(0)
        if self.playing:
            self._mix_song(out, frames)
        self._mix_voices(out, frames)
        self._peaks = np.maximum(self._peaks, np.abs(out).max(axis=0))
        np.clip(out, -1.0, 1.0, out=out)

    def _mix_song(self, out, frames) -> None:
        mix, loop = self._mix, self._loop
        written = 0
        while written < frames:
            end = loop[1] if loop else len(mix)
            if self._pos >= end:
                if not loop:
                    self.playing = False
                    return
                self._pos = loop[0]
            n = min(frames - written, end - self._pos, max(0, len(mix) - self._pos))
            if n <= 0:  # loop region extends past the audio: play silence up to the loop end
                n = min(frames - written, end - self._pos)
            else:
                out[written:written + n] = mix[self._pos:self._pos + n]
            written += n
            self._pos += n

    def _mix_voices(self, out, frames) -> None:
        while self._cut:
            tag = self._cut.popleft()
            self._voices = [v for v in self._voices if v.tag != tag]
        while self._incoming:
            self._voices.append(self._incoming.popleft())
        while self._released:
            tag = self._released.popleft()
            for v in self._voices:
                if v.tag == tag:
                    v.release()
        del self._voices[:-MAX_VOICES]
        alive = []
        for voice in self._voices:
            block, finished = voice.render(frames)
            out += block
            if not finished:
                alive.append(voice)
        self._voices = alive
