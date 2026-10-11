"""Slice audio into pad-sized pieces: at drum hits (transients) or on the beat grid."""

from __future__ import annotations

import numpy as np

from .model import SR

HOP, WIN = 256, 1024


def onsets(mono: np.ndarray, min_gap: float) -> np.ndarray:
    """Sample positions of note/drum onsets, via log-spectral flux with an adaptive threshold."""
    if len(mono) < WIN * 2:
        return np.array([0])
    frames = np.lib.stride_tricks.sliding_window_view(mono, WIN)[::HOP] * np.hanning(WIN)
    mag = np.log1p(np.abs(np.fft.rfft(frames, axis=1)) * 10)
    flux = np.concatenate([[0.0], np.maximum(0.0, np.diff(mag, axis=0)).sum(axis=1)])
    width = max(3, int(0.25 * SR / HOP))
    local = np.convolve(flux, np.ones(width) / width, mode="same")
    threshold = local * 1.4 + flux.mean() * 0.2
    gap = max(1, int(min_gap * SR / HOP))
    picked: list[int] = []
    for i in range(1, len(flux) - 1):
        if flux[i] > threshold[i] and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
            if picked and i - picked[-1] < gap:
                if flux[i] > flux[picked[-1]]:
                    picked[-1] = i
                continue
            picked.append(i)
    starts = [i * HOP for i in picked]
    if np.sqrt(np.mean(mono[:WIN] ** 2)) > 0.25 * np.sqrt(np.mean(mono ** 2)) and (not starts or starts[0] > WIN):
        starts.insert(0, 0)  # a hit right at the start never shows up as a *rise* in flux
    return snap_to_attacks(mono, starts, int(min_gap * SR))


def snap_to_attacks(mono: np.ndarray, starts: list[int], min_gap: int = 0) -> np.ndarray:
    """Flux fires when a hit enters the analysis window, up to ~20 ms early. Move each start to
    2 ms before the sound actually rises, so pads play the moment they're hit."""
    env = np.abs(mono)
    lead = int(0.002 * SR)
    out: list[int] = []
    for s in starts:
        seg = env[s:s + WIN + 2 * HOP]
        if not len(seg) or seg.max() <= 0:
            continue
        attack = s + int(np.argmax(seg > 0.2 * seg.max()))
        start = max(0, attack - lead)
        if not out or start - out[-1] > max(lead * 4, min_gap):  # one sound, one hit
            out.append(start)
    return np.array(out, dtype=int)


def by_hits(audio: np.ndarray, beat: float, count: int = 16) -> list[tuple[int, int]]:
    """(start, end) sample ranges for the first ``count`` hits, each ending at the next hit."""
    # 80 ms floor: one kick's click and body must not become two "hits" (it did, at 176 BPM)
    hits = onsets(audio.mean(axis=1), min_gap=max(0.08, beat / 4 * 0.8))
    longest = int(beat * 2 * SR)
    ranges = []
    for i, start in enumerate(hits[:count]):
        end = hits[i + 1] if i + 1 < len(hits) else len(audio)
        ranges.append((int(start), int(min(end, start + longest))))
    return [r for r in ranges if r[1] - r[0] > int(0.03 * SR)]


def by_grid(audio: np.ndarray, step: float, count: int = 16) -> list[tuple[int, int]]:
    """(start, end) sample ranges of ``count`` consecutive ``step``-second slices."""
    n = int(round(step * SR))
    return [(i * n, min((i + 1) * n, len(audio))) for i in range(count) if i * n < len(audio) - int(0.03 * SR)]


# --- kits: one clean example of each kind of drum sound -------------------------------------

ROLE_ORDER = ["kick", "snare", "hat", "clap", "open hat", "perc", "tom", "fx"]


def _power(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.pad(x[:n], (0, max(0, n - len(x))))
    return np.abs(np.fft.rfft(x * np.hanning(n))) ** 2, np.fft.rfftfreq(n, 1 / SR)


def hit_features(audio: np.ndarray, before: np.ndarray | None = None) -> dict:
    """Describe one drum hit. ``before`` is the audio just ahead of it: whatever was already
    ringing (a kick's tail under a hat) is subtracted, so only what the hit *adds* is measured."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    n = int(0.08 * SR)  # the attack + body is what identifies a drum
    spec, freqs = _power(mono, n)
    if before is not None and len(before):
        prev = before.mean(axis=1) if before.ndim == 2 else before
        spec = np.maximum(spec - _power(prev[-n:], n)[0], 0.0)
    spec = spec + 1e-12
    total = spec.sum()
    top = np.sort(spec)[-max(1, len(spec) // 50):].sum()  # a tone packs its energy into a few bins
    hop = int(0.0025 * SR)  # 2.5 ms RMS envelope: smooth enough to time the decay
    frames = mono[: len(mono) // hop * hop].reshape(-1, hop) if len(mono) >= hop else mono[None, :]
    env = np.sqrt((frames ** 2).mean(axis=1))
    peak_i = int(np.argmax(env))
    floor = float(np.median(env[-max(1, len(env) // 4):]))  # distorted kits sit on a noise bed
    below = np.flatnonzero(env[peak_i:] < max(env[peak_i] * 0.1, floor * 2))
    # claps are a few quick high-frequency bursts: count prominent peaks in the >1 kHz envelope
    head = np.pad(mono[: int(0.06 * SR)], (0, max(0, int(0.06 * SR) - len(mono))))
    hspec = np.fft.rfft(head)
    hspec[np.fft.rfftfreq(len(head), 1 / SR) < 1000] = 0
    high = np.fft.irfft(hspec, len(head))
    ms = int(0.001 * SR)
    henv = np.sqrt((high[: len(high) // ms * ms].reshape(-1, ms) ** 2).mean(axis=1))
    bursts, trough, rising = 0, henv[0] if len(henv) else 0.0, True
    for v in henv:
        if rising and v < trough * 0.6 and trough > 0.25 * henv.max():
            bursts, rising = bursts + 1, False  # a peak, then a real dip: one burst
        if v > trough if rising else v < trough:
            trough = v
        if not rising and v > trough / 0.6 and v > 0.25 * henv.max():
            rising, trough = True, v
    return {
        "low": float(spec[freqs < 150].sum() / total),
        "body": float(spec[(freqs >= 150) & (freqs < 400)].sum() / total),
        "bright": float((spec * freqs).sum() / total),
        "noisy": float(1 - top / total),
        "bursts": bursts,
        "decay": (below[0] * hop / SR) if len(below) else len(env) * hop / SR,
        "peak": float(np.abs(mono).max()) + 1e-9,
    }


def drum_role(f: dict) -> str:
    """Name a hit from its features. Tuned on real generated kits (incl. distorted ones whose snares
    sit low in the mids) and on synthetic ones with known answers (tests/test_pads.py)."""
    if f["low"] > 0.55:
        return "kick" if f["bright"] < 500 else "tom"
    if f["bright"] > 5000:
        return "open hat" if f["decay"] > 0.15 else "hat"
    if f["body"] > 0.3:
        return "snare"  # a strong 150–400 Hz body under the noise (also dark, distorted snares)
    if f["low"] > 0.3 and f["bright"] < 2000:
        return "tom"
    if f["bright"] >= 1500 and (f["bursts"] >= 2 or f["noisy"] >= 0.25):
        return "clap"  # bright noise with little body; several hands a few ms apart
    if f["noisy"] > 0.08 and f["bright"] < 1500:
        return "snare"
    return "perc"


def kit(audio: np.ndarray, beat: float, slots: int = 8) -> list[tuple[int, int, str, float]]:
    """Pick up to ``slots`` distinct drum sounds from a loop: (start, end, role, gain) per pad,
    in drum-machine order (kick, snare, hat, clap, …), each levelled to a similar loudness."""
    hits = by_hits(audio, beat, count=10_000)
    found: dict[str, list] = {}
    for start, end in hits:
        f = hit_features(audio[start:end], audio[max(0, start - int(0.08 * SR)):start])
        if f["peak"] < 0.02:
            continue
        role = drum_role(f)
        lead = audio[max(0, start - 256):start]
        clean = f["peak"] / ((np.abs(lead).max() if len(lead) else 0.0) + 1e-3)  # little bleed before it
        found.setdefault(role, []).append((clean * f["peak"], start, end, f))
    picks: list[tuple[int, int, str, float]] = []
    for round_ in range(3):  # first one of each role, then second variants, …
        for role in ROLE_ORDER:
            options = sorted(found.get(role, []), key=lambda o: -o[0])
            chosen = [p for p in picks if p[2] == role]
            for _score, start, end, f in options:
                if len(picks) >= slots or len(chosen) > round_:
                    break
                # a variant must actually sound different from what's already on a pad
                if any(abs(hit_features(audio[s:e], audio[max(0, s - int(0.08 * SR)):s])["bright"]
                           - f["bright"]) < 0.15 * f["bright"]
                       for s, e, r, _ in chosen):
                    continue
                gain = float(np.clip(0.7 / f["peak"], 0.5, 8.0))  # level everything to ~-3 dB
                picks.append((start, end, role, gain))
                chosen.append(picks[-1])
    return picks
