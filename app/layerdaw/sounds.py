"""Single-sound design: describe one sound, generate it, and pull out the best candidates.

The small model is asked for one kind of sound at a time (a kick, an 808, a Rhodes note…) with a
tag-style caption it understands, and the generated take is mined for clean examples of exactly
that sound, so the user picks from a few good ones instead of getting a whole song.
"""

from __future__ import annotations

import numpy as np

from . import chop
from .chords import detect_pitch
from .model import SR
from .theory import note_name

# type: (caption tags, how to find it in the generated audio, pad kind, play mode)
SOUND_TYPES: dict[str, tuple[str, str, str, str]] = {
    "kick": ("kick drum one-shots, solo kick drum, punchy kick hits, clean studio drum samples",
             "drum:kick", "drums", "oneshot"),
    "snare": ("snare drum one-shots, solo snare hits, crisp snare, clean studio drum samples",
              "drum:snare", "drums", "oneshot"),
    "clap": ("hand clap one-shots, solo claps, layered clap samples, clean studio drum samples",
             "drum:clap", "drums", "oneshot"),
    "hi-hat": ("closed hi-hat one-shots, crisp hats, clean studio drum samples", "drum:hat", "drums", "oneshot"),
    "open hat": ("open hi-hat one-shots, sizzling open hats, clean studio drum samples",
                 "drum:open hat", "drums", "oneshot"),
    "perc": ("percussion one-shots, rim shots, shakers, toms, woodblock", "drum:any", "drums", "oneshot"),
    "808": ("808 bass one-shots, deep sub bass 808 notes, long booming 808, trap 808", "note", "bass", "hold"),
    "bass": ("solo bass, single bass notes, synth bass plucks", "note", "bass", "hold"),
    "keys": ("solo electric piano, single sustained notes, warm keys", "note", "keyboard", "hold"),
    "lead": ("solo synth lead, single held notes, bright lead synth", "note", "synth", "hold"),
    "pad": ("lush ambient synth pad, long sustained chords, evolving warm texture", "sustain", "synth", "hold"),
    "stab": ("chord stab one-shots, short punchy synth stabs", "hit", "synth", "oneshot"),
    "vocal chop": ("vocal chops, pitched vocal one-shots, chopped vocal samples", "hit", "vocals", "oneshot"),
    "fx": ("sound effects, risers, impacts, sweeps, transitions", "long", "fx", "oneshot"),
}
_WORDS = [("808", "808"), ("kick", "kick"), ("snare", "snare"), ("clap", "clap"), ("open hat", "open hat"),
          ("hat", "hi-hat"), ("hihat", "hi-hat"), ("shaker", "perc"), ("perc", "perc"), ("rim", "perc"),
          ("tom", "perc"), ("sub", "808"), ("bass", "bass"), ("piano", "keys"), ("rhodes", "keys"),
          ("keys", "keys"), ("organ", "keys"), ("lead", "lead"), ("pluck", "lead"), ("pad", "pad"),
          ("drone", "pad"), ("stab", "stab"), ("chop", "vocal chop"), ("vocal", "vocal chop"),
          ("vox", "vocal chop"), ("riser", "fx"), ("impact", "fx"), ("sweep", "fx"), ("fx", "fx")]


def guess_type(text: str) -> str:
    t = (text or "").lower()
    return next((kind for word, kind in _WORDS if word in t), "lead")


def caption(sound_type: str, description: str) -> str:
    """The user's words first (the caption matters most), then the tags that keep it to one sound."""
    tags = SOUND_TYPES[sound_type][0]
    return f"{description.strip()}, {tags}" if description.strip() else tags


def _distinct(cands: list[tuple], key, count: int, rel: float = 0.15) -> list[tuple]:
    out: list[tuple] = []
    for c in cands:
        if all(abs(key(c) - key(o)) > rel * max(abs(key(c)), 1e-9) for o in out):
            out.append(c)
        if len(out) == count:
            break
    return out


def candidates(audio: np.ndarray, sound_type: str, beat: float, bar: float,
               count: int = 4) -> list[tuple[int, int, str]]:
    """Up to ``count`` (start, end, label) examples of ``sound_type`` found in ``audio``."""
    how = SOUND_TYPES[sound_type][1]
    if how.startswith("drum:"):
        want = how.split(":", 1)[1]
        found = []
        for start, end in chop.by_hits(audio, beat, count=10_000):
            f = chop.hit_features(audio[start:end], audio[max(0, start - int(0.08 * SR)):start])
            if f["peak"] < 0.02:
                continue
            role = chop.drum_role(f)
            lead = audio[max(0, start - 256):start]
            clean = f["peak"] / ((np.abs(lead).max() if len(lead) else 0.0) + 1e-3)
            match = want == "any" or role == want
            found.append((match, clean * f["peak"], start, end, f["bright"], role))
        matches = sorted((c for c in found if c[0]), key=lambda c: -c[1])
        pool = matches or sorted(found, key=lambda c: -c[1])  # nothing of that kind: show what's there
        picks = _distinct(pool, lambda c: c[4], count)
        for c in pool:  # not enough different ones: top up with the next-best of the same kind
            if len(picks) >= count:
                break
            if all(c[2] != p[2] for p in picks):
                picks.append(c)
        return [(s, e, f"{role} {i}") for i, (_, _, s, e, _, role) in enumerate(picks, 1)]

    if how == "sustain":
        win, hop = int(min(2.0, bar) * SR), int(0.25 * SR)
        scored = []
        for s in range(0, max(1, len(audio) - win), hop):
            seg = audio[s:s + win]
            env = np.sqrt((seg[: win // 2400 * 2400].mean(axis=1).reshape(-1, 2400) ** 2).mean(axis=1))
            if env.mean() > 1e-3:
                scored.append((env.mean() / (1 + 3 * env.std() / env.mean()), s))
        picks: list[int] = []
        for _score, s in sorted(scored, reverse=True):
            if all(abs(s - p) >= win for p in picks):
                picks.append(s)
            if len(picks) == count:
                break
        return [(s, s + win, f"pad {i}") for i, s in enumerate(sorted(picks), 1)]

    longest = {"note": bar, "hit": bar, "long": 2 * bar}[how]
    hits = chop.onsets(audio.mean(axis=1), max(0.08, beat / 4))
    found = []
    for i, s in enumerate(hits):
        e = int(min(hits[i + 1] if i + 1 < len(hits) else len(audio), s + longest * SR))
        if e - s < int(0.08 * SR):
            continue
        seg = audio[s:e]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        pitch = detect_pitch(seg) if how == "note" and e - s >= int(0.15 * SR) else None
        if how == "note" and pitch is None:
            continue
        found.append((rms, int(s), e, pitch if pitch is not None else chop.hit_features(seg)["bright"]))
    found.sort(reverse=True)
    if how == "note":  # different notes (a semitone apart at least), not the same one four times
        picks, notes = [], set()
        for c in found:
            if round(c[3]) not in notes:
                picks.append(c)
                notes.add(round(c[3]))
            if len(picks) == count:
                break
        return [(s, e, f"{sound_type} {note_name(p)}") for _, s, e, p in picks]
    picks = _distinct(found, lambda c: c[3], count)
    return [(s, e, f"{sound_type} {i}") for i, (_, s, e, _) in enumerate(picks, 1)]
