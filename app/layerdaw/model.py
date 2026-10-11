"""Project data: tracks of audio clips on one timeline, plus mixdown and save/load."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SR = 48000  # ACE-Step renders at 48 kHz, so the whole project runs at that rate.

# Track types ACE-Step's add-layer ("lego") task understands.
LAYER_TYPES = ["vocals", "backing_vocals", "drums", "bass", "guitar", "keyboard",
               "percussion", "strings", "synth", "fx", "brass", "woodwinds"]
VOCAL_TYPES = {"vocals", "backing_vocals"}
# parts the engine can separate out of a generated mix (Demucs stems), and the kind a stem becomes
STEM_OF = {"vocals": "vocals", "backing_vocals": "vocals", "drums": "drums", "percussion": "drums", "bass": "bass"}
STEM_KIND = {"vocals": "vocals", "drums": "drums", "bass": "bass", "other": "audio"}
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}

_file_cache: dict[Path, np.ndarray] = {}


def load_audio(path: Path) -> np.ndarray:
    """Read any soundfile-supported file as float32 stereo at the project rate (cached per file)."""
    path = Path(path)
    if path not in _file_cache:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2]
        if sr != SR:
            audio = soxr.resample(audio, sr, SR).astype(np.float32)
        if len(_file_cache) > 64:
            _file_cache.clear()
        _file_cache[path] = np.ascontiguousarray(audio)
    return _file_cache[path]


@dataclass(eq=False)
class Clip:
    """A window onto an audio file: plays file[offset : offset+length] starting at ``start``."""

    path: Path
    start: float = 0.0          # seconds on the timeline
    offset: float = 0.0         # seconds into the file
    length: float | None = None  # seconds; None = to the end of the file
    audio: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.audio is None:
            full = load_audio(self.path)
            a = int(round(self.offset * SR))
            b = len(full) if self.length is None else a + int(round(self.length * SR))
            self.audio = full[a:b]
        self.length = len(self.audio) / SR

    @property
    def end(self) -> float:
        return self.start + self.length

    def copy(self, start: float | None = None) -> Clip:
        return replace(self, start=self.start if start is None else start)

    def split(self, t: float) -> tuple[Clip, Clip] | None:
        """Cut at timeline time ``t``; None if ``t`` isn't strictly inside the clip."""
        k = int(round((t - self.start) * SR))
        if not 0 < k < len(self.audio):
            return None
        left = Clip(self.path, self.start, self.offset, k / SR, self.audio[:k])
        right = Clip(self.path, t, self.offset + k / SR, (len(self.audio) - k) / SR, self.audio[k:])
        return left, right


@dataclass(eq=False)
class Track:
    name: str
    kind: str = "audio"
    clips: list[Clip] = field(default_factory=list)
    volume: float = 1.0
    pan: float = 0.0  # -1 left .. +1 right
    muted: bool = False
    solo: bool = False
    prompt: str = ""
    gen: dict = field(default_factory=dict)  # generation settings, so the layer can be regenerated
    fx: dict = field(default_factory=dict)   # sound shaping applied in the mix (see fx.py); never baked in

    @property
    def end(self) -> float:
        return max((c.end for c in self.clips), default=0.0)

    def copy(self) -> Track:
        return replace(self, name=f"{self.name} copy", clips=[c.copy() for c in self.clips], gen=dict(self.gen),
                       fx=dict(self.fx))

    def gains(self) -> np.ndarray:
        """Volume with a balance-style pan applied, per channel."""
        return np.array([min(1.0, 1.0 - self.pan), min(1.0, 1.0 + self.pan)], np.float32) * self.volume


PAD_COUNT = 16


def freeze(base: np.ndarray, seconds: float = 4.0) -> tuple[np.ndarray, tuple[int, int]]:
    """Attack of ``base`` followed by a sustain grown from its own body, plus loop points over the
    sustain (a 4 s period with a crossfaded seam, so even long holds don't sound repetitive).

    The sustain is WSOLA: grains wander slowly back and forth through the body, and each one is
    placed where its waveform lines up with the previous grain's, so even deep tones (a kick's
    body) stay smooth instead of beating. Only this sound sustains; nothing that came after it."""
    n = len(base)
    env = np.abs(base).max(axis=1)
    peak = int(np.argmax(env))
    attack_end = min(n - 1, peak + int(0.02 * SR))  # keep the real attack, freeze what follows
    tail = int(0.006 * SR) + 1
    body = base[attack_end:max(attack_end + 1, n - tail)]
    # only the part that's still really sounding: stop where it has faded to ~-16 dB
    hop5 = int(0.005 * SR)
    if len(body) > 4 * hop5:
        rms = np.sqrt((body[: len(body) // hop5 * hop5].mean(axis=1).reshape(-1, hop5) ** 2).mean(axis=1))
        faded = np.flatnonzero(rms < rms[:2].mean() * 0.15)
        if len(faded):
            body = body[: max(int(0.03 * SR), faded[0] * hop5)]
    if len(body) < int(0.03 * SR):  # nearly nothing after the attack: use the whole slice
        body = base[: max(1, n - tail)]
    mono = body.mean(axis=1)
    level = np.sqrt(np.mean(body[: int(0.05 * SR)] ** 2)) + 1e-9  # every grain is brought to this
    grain = int(np.clip(len(body) * 0.5, 0.02 * SR, 0.06 * SR))
    grain = max(64, min(grain, len(body)))
    hop = grain // 2
    search = min(int(0.01 * SR), max(0, (len(body) - grain) // 2))
    window = np.hanning(grain).astype(np.float32)
    length = int(seconds * SR)
    out = np.zeros((length + grain, 2), np.float32)
    weight = np.zeros(length + grain, np.float32)
    span = max(1, len(body) - grain - 2 * search)
    pos, speed, prev = float(search), 0.3, None  # read pointer drifts at 0.3× real time, ping-pong
    for at in range(0, length, hop):
        target = int(pos)
        src = target
        if prev is not None and search > 0:
            want = mono[prev + hop: prev + hop + hop]  # how the last grain would naturally go on
            lo, hi = max(0, target - search), min(len(mono) - grain, target + search)
            if len(want) == hop and hi > lo:
                region = mono[lo:hi + hop]
                size = 1 << int(np.ceil(np.log2(len(region) + hop)))
                xc = np.fft.irfft(np.fft.rfft(region, size) * np.conj(np.fft.rfft(want, size)), size)
                src = lo + int(np.argmax(xc[: hi - lo + 1]))
        piece = body[src:src + grain]
        gain = np.clip(level / (np.sqrt(np.mean(piece ** 2)) + 1e-9), 0.25, 4.0)  # no natural decay in a hold
        out[at:at + len(piece)] += piece * (window[: len(piece), None] * gain)
        weight[at:at + len(piece)] += window[: len(piece)]
        prev = src
        pos += hop * speed
        if pos >= search + span or pos < search:
            speed, pos = -speed, min(max(pos, search), search + span - 1)
    sustain = out[:length] / np.maximum(weight[:length], 1e-3)[:, None]
    # match the level of the body right after the attack, so the join is seamless
    target_rms = np.sqrt(np.mean(base[attack_end:attack_end + int(0.05 * SR)] ** 2)) + 1e-9
    sustain *= target_rms / (np.sqrt(np.mean(sustain ** 2)) + 1e-9)
    head = base[:attack_end].astype(np.float32)
    x = min(int(0.015 * SR), len(head), len(sustain))
    if x > 0:  # crossfade the attack into the sustain
        fade = np.linspace(0, 1, x, dtype=np.float32)[:, None]
        joined = head[-x:] * (1 - fade) + sustain[:x] * fade
        result = np.concatenate([head[:-x], joined, sustain[x:]])
    else:
        result = np.concatenate([head, sustain])
    return seamless_loop_region(result, len(head), len(result) - int(0.01 * SR))


def seamless_loop_region(buf: np.ndarray, start: int, end: int, xfade: float = 0.05) -> tuple[np.ndarray, tuple[int, int]]:
    """Copy of ``buf`` whose [start, end) region loops without a click (crossfaded seam)."""
    out = buf.copy()
    x = min(int(xfade * SR), (end - start) // 3, start)
    if x > 0:
        fade = np.linspace(0, 1, x, dtype=np.float32)[:, None]
        out[end - x:end] = out[end - x:end] * (1 - fade) + out[start - x:start] * fade
    return out, (start, end)


def seamless_loop(base: np.ndarray) -> tuple[np.ndarray, tuple[int, int] | None]:
    """A copy of ``base`` plus loop points for sustaining it. The loop sits in the body (after the
    attack, before the declick tail); the last few ms before the loop end are blended with the
    audio just before the loop start, so jumping end→start is seamless."""
    buf, n = base.copy(), len(base)
    tail = int(0.006 * SR) + 1
    start, end = max(int(0.25 * n), int(0.03 * SR)), n - tail
    if end - start < int(0.02 * SR):  # short slice: loop (almost) all of it
        start = int(0.003 * SR)
    if end - start < int(0.01 * SR):
        return buf, None
    x = min(int(0.03 * SR), (end - start) // 3, start)
    if x > 0:
        fade = np.linspace(0, 1, x, dtype=np.float32)[:, None]
        buf[end - x:end] = buf[end - x:end] * (1 - fade) + buf[start - x:start] * fade
    return buf, (start, end)


def declick(audio: np.ndarray) -> np.ndarray:
    """Copy with 1 ms fade-in / 6 ms fade-out so a slice never clicks."""
    out = np.array(audio, dtype=np.float32, copy=True)
    for n, ramp in ((min(len(out), int(0.001 * SR)), (0, 1)), (min(len(out), int(0.006 * SR)), (1, 0))):
        if n:
            fade = np.linspace(*ramp, n, dtype=np.float32)[:, None]
            if ramp[0] == 0:
                out[:n] *= fade
            else:
                out[-n:] *= fade
    return out


@dataclass(eq=False)
class Pad:
    """One sampler pad: a slice of some audio file, played as a one-shot."""

    clip: Clip | None = None
    name: str = ""
    gain: float = 1.0
    pitch: int = 0  # semitones, resampled like a classic sampler (higher = shorter)
    kind: str = "drums"  # colour on the pad grid
    # "oneshot" plays the slice · "hold" sustains this sound while held (freeze) ·
    # "loop" repeats the slice while held · "playon" keeps playing the original audio while held
    mode: str = "oneshot"
    # what a hit plays: the "sample" itself, a "chord" of the key built from this sound, or one
    # "note" of the key (an 808/bass/lead line). Chord/note pads without a sound use a synth.
    play: str = "sample"
    degree: int = 0          # scale degree for chord/note pads (0 = the key's root, 7 = an octave up)
    flavour: str = "triad"   # chord pads: triad, 7th, sus2, sus4, add9
    strum: bool = False      # chord pads: roll the notes in
    root: float | None = None  # detected pitch of this pad's sound (MIDI), for chord/note pads

    def freeze_buffer(self) -> tuple[np.ndarray, tuple[int, int]] | None:
        """Hold mode: (buffer, loop) = this pad's attack, then a smooth sustain grown from grains of
        its own body. Only this sound sustains: no loop rhythm, nothing that came after it."""
        base = self.buffer()
        if base is None:
            return None
        cache = self.__dict__.get("_freeze")
        if cache is None or cache[0] is not base:
            cache = self.__dict__["_freeze"] = (base, *freeze(base))
        return cache[1], cache[2]

    def continue_buffer(self) -> np.ndarray | None:
        """Hold mode: the source audio from this pad's start onward (up to a minute), so holding
        a pad keeps playing the original instead of repeating the slice."""
        if self.clip is None:
            return None
        full = load_audio(self.clip.path)
        key = (self.clip.offset, self.pitch)
        cache = self.__dict__.get("_cont")
        if cache is None or cache[0] is not full or cache[1] != key:
            a = int(round(self.clip.offset * SR))
            view = full[a:a + int((20.0 if self.pitch else 60.0) * SR)]  # a view: no copy unless pitched
            if self.pitch:
                view = soxr.resample(view, SR * 2 ** (self.pitch / 12), SR).astype(np.float32)
            cache = self.__dict__["_cont"] = (full, key, view)
        return cache[2]

    def hold_buffer(self) -> tuple[np.ndarray, tuple[int, int] | None] | None:
        """Loop mode: (buffer, loop points); see ``seamless_loop``."""
        base = self.buffer()
        if base is None:
            return None
        cache = self.__dict__.get("_hold")
        if cache is None or cache[0] is not base:
            cache = self.__dict__["_hold"] = (base, *seamless_loop(base))
        return cache[1], cache[2]

    def buffer(self) -> np.ndarray | None:
        if self.clip is None:
            return None
        cache = self.__dict__.get("_buf")
        if cache is None or cache[0] is not self.clip.audio or cache[1] != self.pitch:
            audio = self.clip.audio
            if self.pitch:
                audio = soxr.resample(audio, SR * 2 ** (self.pitch / 12), SR).astype(np.float32)
            cache = self.__dict__["_buf"] = (self.clip.audio, self.pitch, declick(audio))
        return cache[2]

    def clear(self) -> None:
        self.clip, self.name, self.gain, self.pitch, self.kind, self.mode = None, "", 1.0, 0, "drums", "oneshot"
        self.play, self.degree, self.flavour, self.strum, self.root = "sample", 0, "triad", False, None


@dataclass
class Project:
    folder: Path
    bpm: int = 90
    key: str = "A minor"
    beats_per_bar: int = 4
    tracks: list[Track] = field(default_factory=list)
    pads: list[Pad] = field(default_factory=lambda: [Pad() for _ in range(PAD_COUNT)])
    chords: dict = field(default_factory=lambda: {"migrated": True})  # legacy chord-row settings (see load)

    @property
    def name(self) -> str:
        return self.folder.name

    @property
    def audio_dir(self) -> Path:
        return self.folder / "audio"

    @property
    def bar_seconds(self) -> float:
        return 60.0 / self.bpm * self.beats_per_bar

    @property
    def beat_seconds(self) -> float:
        return 60.0 / self.bpm

    @property
    def length(self) -> float:
        return max((t.end for t in self.tracks), default=0.0)

    def audible(self, track: Track) -> bool:
        any_solo = any(t.solo for t in self.tracks)
        return not track.muted and (track.solo or not any_solo)

    def render(self, tracks: list[Track] | None = None, length: float | None = None,
               ready_only: bool = False, dry: bool = False) -> np.ndarray:
        """Mix the given tracks (default: everything audible) into one stereo buffer, with each
        track's fx. ``ready_only``: a track whose fx are still being worked out plays dry for now
        (the UI re-mixes when they're ready) instead of holding everything up. ``dry``: no fx at all
        (what a layer sounds like as recorded, e.g. for regenerating part of it)."""
        from . import fx
        tracks = [t for t in self.tracks if self.audible(t)] if tracks is None else tracks
        frames = int(round((self.length if length is None else length) * SR))
        mix = np.zeros((frames, 2), dtype=np.float32)
        for track in tracks:
            gains = track.gains()
            shaped = fx.output(self, track, compute=not ready_only) if fx.is_active(track.fx) and not dry else None
            parts = [shaped] if shaped else [(int(round(c.start * SR)), c.audio) for c in track.clips]
            for start, audio in parts:
                n = min(len(audio), frames - start)
                if n > 0:
                    mix[start:start + n] += audio[:n] * gains
        return mix

    def write_wav(self, audio: np.ndarray, name: str) -> Path:
        stem = self.folder / "renders" / f"{name}-{time.strftime('%H%M%S')}"
        path, n = stem.with_suffix(".wav"), 2
        while path.exists():  # never overwrite: undo may still point at the earlier file
            path, n = stem.parent / f"{stem.name}-{n}.wav", n + 1
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, SR, subtype="FLOAT")
        return path

    def add_file(self, src: Path, name: str, kind: str = "audio", start: float = 0.0, prompt: str = "") -> Track:
        """Copy an audio file into the project and put it on a new track."""
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        dest = self.audio_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}{src.suffix}"
        shutil.copy2(src, dest)
        track = Track(name=name, kind=kind, clips=[Clip(dest, start)], prompt=prompt)
        self.tracks.append(track)
        return track

    def export(self, dest: Path) -> list[Path]:
        """Write every track as a full-length stem plus the full mix, all starting at 0:00."""
        dest.mkdir(parents=True, exist_ok=True)
        written = [self.export_track(t, dest / f"{i:02d}-{_safe(t.name)}.wav")
                   for i, t in enumerate(self.tracks, 1)]
        mix = dest / "mix.wav"
        sf.write(str(mix), np.clip(self.render(), -1, 1), SR, subtype="PCM_24")
        return written + [mix]

    def export_track(self, track: Track, path: Path) -> Path:
        sf.write(str(path), np.clip(self.render([track], self.length), -1, 1), SR, subtype="PCM_24")
        return path

    def rename(self, new_name: str) -> None:
        """Rename the song (its folder), keeping every clip pointing at the moved files."""
        new_folder = self.folder.with_name(_safe(new_name))
        if new_folder.exists():
            raise FileExistsError(f"A song called '{new_folder.name}' already exists")
        old = self.folder
        if old.exists():
            old.rename(new_folder)
        self.folder = new_folder
        for clip in [c for t in self.tracks for c in t.clips] + [p.clip for p in self.pads if p.clip]:
            if clip.path.is_relative_to(old):
                clip.path = new_folder / clip.path.relative_to(old)


    # --- persistence -----------------------------------------------------------------

    def save(self) -> None:
        data = {
            "bpm": self.bpm, "key": self.key, "beats_per_bar": self.beats_per_bar,
            "tracks": [{
                "name": t.name, "kind": t.kind, "volume": t.volume, "pan": t.pan, "muted": t.muted,
                "solo": t.solo, "prompt": t.prompt, "gen": t.gen, "fx": t.fx,
                "clips": [{"path": self._rel(c.path), "start": c.start, "offset": c.offset, "length": c.length}
                          for c in t.clips],
            } for t in self.tracks],
            "pads": [None if pad.clip is None and pad.play == "sample" else {
                "name": pad.name, "gain": pad.gain, "pitch": pad.pitch, "kind": pad.kind, "mode": pad.mode,
                "play": pad.play, "degree": pad.degree, "flavour": pad.flavour, "strum": pad.strum, "root": pad.root,
                "clip": None if pad.clip is None else
                {"path": self._rel(pad.clip.path), "offset": pad.clip.offset, "length": pad.clip.length},
            } for pad in self.pads],
            "chords": {"migrated": True},
        }
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self.folder / "project.json.tmp"
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.folder / "project.json")  # atomic: a crash mid-save can't corrupt the song

    def _rel(self, path: Path) -> str:
        """Project-relative when the file lives inside the project, absolute otherwise."""
        return str(path.relative_to(self.folder)) if path.is_relative_to(self.folder) else str(path.resolve())

    @classmethod
    def load(cls, folder: Path) -> Project:
        data = json.loads((folder / "project.json").read_text())
        tracks = [Track(name=t["name"], kind=t["kind"], volume=t["volume"], pan=t.get("pan", 0.0),
                        muted=t["muted"], solo=t["solo"], prompt=t.get("prompt", ""), gen=t.get("gen", {}), fx=t.get("fx", {}),
                        clips=[Clip(folder / c["path"], c["start"], c.get("offset", 0.0), c.get("length"))
                               for c in t["clips"]])
                  for t in data["tracks"]]
        pads = [Pad() for _ in range(PAD_COUNT)]
        for pad, saved in zip(pads, data.get("pads", [])):
            if saved:
                c = saved.get("clip")
                pad.clip = Clip(folder / c["path"], 0.0, c["offset"], c["length"]) if c else None
                pad.name, pad.gain, pad.pitch = saved["name"], saved["gain"], saved["pitch"]
                pad.kind = saved.get("kind", "drums")
                pad.mode = saved.get("mode", "oneshot")
                pad.play, pad.degree = saved.get("play", "sample"), saved.get("degree", 0)
                pad.flavour, pad.strum = saved.get("flavour", "triad"), saved.get("strum", False)
                pad.root = saved.get("root")
        legacy = data.get("chords", {})
        if legacy.get("enabled", "slots" in legacy or "source" in legacy) and not legacy.get("migrated"):
            _migrate_chord_rows(folder, pads, legacy)
        return cls(folder, data["bpm"], data["key"], data["beats_per_bar"], tracks, pads, {"migrated": True})

    @staticmethod
    def recent(root: Path, limit: int = 10) -> list[Path]:
        songs = [p.parent for p in root.glob("*/project.json")]
        return sorted(songs, key=lambda p: (p / "project.json").stat().st_mtime, reverse=True)[:limit]


def _migrate_chord_rows(folder: Path, pads: list[Pad], legacy: dict) -> None:
    """Songs from before per-pad chords had fixed rows: QWER = chord slots sharing one chord sound,
    1234 = hooks. Turn those into ordinary pads that play chords / samples."""
    src = legacy.get("source")
    for i, degree in zip(range(8, 12), legacy.get("slots", [0, 3, 4, 5])):
        pad = pads[i]
        pad.clear()
        pad.play, pad.degree, pad.mode, pad.kind = "chord", degree, "hold", "keys"
        pad.flavour, pad.strum = legacy.get("flavour", "triad"), bool(legacy.get("strum", False))
        if src:
            try:
                pad.clip = Clip(folder / src["path"], 0.0, src["offset"], src["length"])
                pad.name, pad.root = src.get("name", "chords"), src.get("root")
            except Exception:  # noqa: BLE001 - missing chord sound: the synth plays instead
                pad.clip = None
        pad.name = pad.name or "chords"
    for i in range(12, 16):
        if pads[i].clip is not None:
            pads[i].kind = "hook"


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in " -_+()." else "_" for c in name).strip() or "untitled"
