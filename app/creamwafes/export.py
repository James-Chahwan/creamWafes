"""Exporting the song as one audio file: formats (MP3, WAV, FLAC, OGG, Opus, AIFF, M4A), sample rate,
and level (as mixed, peak-normalised, or loudness-matched for streaming/clubs with a limiter).

Loudness is measured as in ITU-R BS.1770-4 / EBU R128 (K-weighting, 400 ms blocks, gated).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import minimum_filter1d

from .model import SR

# key: (label, extension, qualities [(label, value)], default quality index)
FORMATS: dict[str, tuple[str, str, list[tuple[str, object]], int]] = {
    "mp3": ("MP3", ".mp3", [("320 kbps", 320), ("256 kbps", 256), ("192 kbps", 192), ("128 kbps", 128)], 0),
    "wav": ("WAV", ".wav", [("24-bit", "PCM_24"), ("16-bit (CD)", "PCM_16"), ("32-bit float", "FLOAT")], 0),
    "flac": ("FLAC (lossless, smaller)", ".flac", [("24-bit", "PCM_24"), ("16-bit (CD)", "PCM_16")], 0),
    "ogg": ("OGG Vorbis", ".ogg", [("high", 0.1), ("medium", 0.4), ("small", 0.7)], 0),
    "opus": ("Opus", ".opus", [("high", 0.1), ("medium", 0.4), ("small", 0.7)], 0),
    "aiff": ("AIFF", ".aiff", [("24-bit", "PCM_24"), ("16-bit (CD)", "PCM_16")], 0),
    "m4a": ("M4A / AAC (via ffmpeg)", ".m4a", [("256 kbps", 256), ("192 kbps", 192), ("128 kbps", 128)], 0),
}
# libsndfile's MP3 encoder: compression level → constant bitrate (measured with ffprobe)
_MP3_LEVEL = {320: 0.0, 256: 0.125, 192: 0.4, 128: 0.625}
LEVELS = [("mixed", "As mixed (only kept from clipping)"), ("peak", "Normalise peaks to −1 dB"),
          ("stream", "Loud for streaming (−14 LUFS)"), ("club", "Louder, club/DJ (−9 LUFS)")]
TARGET_LUFS = {"stream": -14.0, "club": -9.0}
CEILING = 10 ** (-1 / 20)  # −1 dBFS: room for MP3/AAC encoding overshoot


def available() -> list[str]:
    """Formats this machine can write (M4A needs ffmpeg)."""
    return [k for k in FORMATS if k != "m4a" or shutil.which("ffmpeg")]


def fixed_rate(fmt: str) -> int | None:
    return 48000 if fmt == "opus" else None  # Opus is always 48 kHz


# --- loudness ---------------------------------------------------------------------------------------

def _k_weighting(sr: int) -> np.ndarray:
    """BS.1770 K-weighting (high shelf + high pass) as SOS for any sample rate."""
    # stage 1: high shelf, +4 dB above ~1.5 kHz
    f0, g, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = np.tan(np.pi * f0 / sr)
    vh = 10 ** (g / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    b1 = [(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0]
    a1 = [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    # stage 2: high pass at ~38 Hz
    f0, q = 38.13547087602444, 0.5003270373238773
    k = np.tan(np.pi * f0 / sr)
    a0 = 1 + k / q + k * k
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    return np.array([b1 + a1, b2 + a2])


def lufs(audio: np.ndarray, sr: int = SR) -> float:
    """Integrated loudness (LUFS), gated like EBU R128. -inf for silence."""
    if len(audio) < int(0.4 * sr):
        return float("-inf")
    y = signal.sosfilt(_k_weighting(sr), audio.astype(np.float64), axis=0)
    block, hop = int(0.4 * sr), int(0.1 * sr)
    sq = np.cumsum(np.concatenate([np.zeros((1, y.shape[1])), y ** 2]), axis=0)
    starts = np.arange(0, len(y) - block + 1, hop)
    z = ((sq[starts + block] - sq[starts]) / block).sum(axis=1)  # L + R, weight 1 each
    loud = -0.691 + 10 * np.log10(np.maximum(z, 1e-20))
    z = z[loud > -70]
    if not len(z):
        return float("-inf")
    rel = -0.691 + 10 * np.log10(z.mean()) - 10
    z = z[-0.691 + 10 * np.log10(z) > rel]
    return float(-0.691 + 10 * np.log10(z.mean())) if len(z) else float("-inf")


def limit(audio: np.ndarray, ceiling: float = CEILING, lookahead: float = 0.005, release: float = 0.08,
          sr: int = SR) -> np.ndarray:
    """Brickwall limiter: never above ``ceiling``; turns down just before peaks, recovers smoothly."""
    peak = np.abs(audio).max(axis=1)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    ahead = max(1, int(lookahead * sr))
    need = minimum_filter1d(need, size=2 * ahead + 1)  # start turning down before the peak arrives
    block = 32
    n = -(-len(need) // block)
    padded = np.concatenate([need, np.ones(n * block - len(need))])
    lows = padded.reshape(n, block).min(axis=1)
    up = np.exp(-block / (release * sr))
    gain, g = np.empty(n), 1.0
    for k, v in enumerate(lows.tolist()):
        g = v if v < g else g * up + v * (1 - up)  # instant down, slow back up
        gain[k] = g
    per_sample = np.minimum(np.repeat(gain, block)[:len(need)], need)
    out = audio * per_sample[:, None]
    return np.clip(out, -ceiling, ceiling).astype(np.float32)


def master(audio: np.ndarray, level: str, sr: int = SR) -> tuple[np.ndarray, dict]:
    """Set the level for export. Returns (audio, report {lufs, peak_db, gain_db})."""
    audio = np.asarray(audio, np.float32)
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    gain = 1.0
    if level == "peak" and peak > 0:
        gain = CEILING / peak
    elif level in TARGET_LUFS:
        now = lufs(audio, sr)
        if np.isfinite(now):
            gain = 10 ** ((TARGET_LUFS[level] - now) / 20)
    out = audio * gain
    if level != "peak":  # "as mixed" only ever gets kept from clipping; loudness targets get limited
        if np.abs(out).max() > (CEILING if level in TARGET_LUFS else 1.0):
            out = limit(out, CEILING if level in TARGET_LUFS else 0.999, sr=sr)
    out_peak = float(np.abs(out).max()) if len(out) else 0.0
    return out.astype(np.float32), {"lufs": lufs(out, sr), "peak_db": 20 * np.log10(max(out_peak, 1e-9)),
                                    "gain_db": 20 * np.log10(gain)}


def trim_tail(audio: np.ndarray, keep: int, floor_db: float = -70.0) -> np.ndarray:
    """Cut silence after the music, but never before ``keep`` frames (the song's own length)."""
    level = np.abs(audio).max(axis=1)
    loud = np.flatnonzero(level > 10 ** (floor_db / 20))
    end = max(keep, (loud[-1] + int(0.05 * SR)) if len(loud) else keep)
    return audio[:min(len(audio), end)]


# --- writing ------------------------------------------------------------------------------------------

def write(path: Path, audio: np.ndarray, fmt: str, quality, rate: int, src_rate: int = SR,
          title: str = "", artist: str = "") -> Path:
    """Write ``audio`` (at ``src_rate``) as ``fmt`` at ``rate``. Returns the file written."""
    import soxr
    path = Path(path).with_suffix(FORMATS[fmt][1])
    path.parent.mkdir(parents=True, exist_ok=True)
    rate = fixed_rate(fmt) or rate
    if rate != src_rate:
        audio = soxr.resample(audio, src_rate, rate, quality="VHQ").astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    if fmt == "m4a":
        with tempfile.TemporaryDirectory(prefix="creamwafes-export-") as tmp:
            wav = Path(tmp) / "mix.wav"
            sf.write(wav, audio, rate, subtype="FLOAT")
            cmd = [shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                   "-c:a", "aac", "-b:a", f"{quality}k"]
            for key, value in (("title", title), ("artist", artist)):
                if value:
                    cmd += ["-metadata", f"{key}={value}"]
            subprocess.run([*cmd, str(path)], check=True, timeout=600)
        return path
    kwargs = {"wav": dict(format="WAV", subtype=quality), "aiff": dict(format="AIFF", subtype=quality),
              "flac": dict(format="FLAC", subtype=quality),
              "ogg": dict(format="OGG", subtype="VORBIS", compression_level=quality),
              "opus": dict(format="OGG", subtype="OPUS", compression_level=quality),
              "mp3": dict(format="MP3", subtype="MPEG_LAYER_III", compression_level=_MP3_LEVEL.get(quality, 0.0),
                          bitrate_mode="CONSTANT")}[fmt]
    with sf.SoundFile(path, "w", rate, audio.shape[1], **kwargs) as f:
        for key, value in (("title", title), ("artist", artist), ("software", "creamWafes")):
            if value:
                try:
                    setattr(f, key, value)
                except Exception:  # noqa: BLE001 - not every format takes tags
                    pass
        f.write(audio)
    return path
