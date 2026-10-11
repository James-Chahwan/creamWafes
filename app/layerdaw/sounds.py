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
    # more kinds
    "rim": ("rim shot one-shots, side stick, rimshot clicks, clean studio drum samples", "drum:any", "drums", "oneshot"),
    "crash": ("crash cymbal one-shots, cymbal crashes, clean studio drum samples", "long", "drums", "oneshot"),
    "ride": ("ride cymbal hits, ride bell one-shots, clean studio drum samples", "hit", "drums", "oneshot"),
    "sub bass": ("pure sub bass notes, clean sine sub, deep low bass one-shots", "note", "bass", "hold"),
    "guitar": ("solo electric guitar, single plucked notes, clean guitar tone", "note", "guitar", "hold"),
    "strings": ("solo string section, sustained strings, legato violins and cellos", "sustain", "strings", "hold"),
    "brass": ("solo brass section, held brass notes, trumpets and horns", "note", "brass", "hold"),
    "bells": ("bells and mallets, glockenspiel, music box, celesta single notes", "note", "keyboard", "hold"),
    "flute": ("solo flute, breathy single notes, wooden flute", "note", "woodwinds", "hold"),
    "choir": ("choir, sustained vocal ahhs, layered voices", "sustain", "vocals", "hold"),
    "riser": ("risers, white noise sweeps, build ups, uplifters", "long", "fx", "oneshot"),
    "impact": ("impacts, booms, downlifters, cinematic hits", "hit", "fx", "oneshot"),
    # loops: whole bars at the song's tempo, played as loop pads
    "drum loop": ("drum loop, solo drums, full drum groove, kick snare hi-hats", "loop", "drums", "loop"),
    "perc loop": ("percussion loop, shakers congas bongos groove, solo percussion", "loop", "drums", "loop"),
    "bass loop": ("bass line loop, solo bass groove, repeating bass riff", "loop", "bass", "loop"),
    "chord loop": ("chord progression loop, solo piano chords, repeating chords", "loop", "keyboard", "loop"),
    "melody loop": ("melody loop, solo lead melody, catchy repeating riff", "loop", "synth", "loop"),
    "vocal loop": ("vocal loop, repeating vocal phrase, solo vocals, chopped vocal hook", "loop", "vocals", "loop"),
}
GROUPS = [("Drums", ["kick", "snare", "clap", "hi-hat", "open hat", "rim", "perc", "crash", "ride"]),
          ("Bass", ["808", "sub bass", "bass"]),
          ("Melodic", ["keys", "bells", "guitar", "strings", "brass", "flute", "lead", "pad", "stab"]),
          ("Vocal", ["vocal chop", "choir"]),
          ("FX", ["riser", "impact", "fx"]),
          ("Loops", ["drum loop", "perc loop", "bass loop", "chord loop", "melody loop", "vocal loop"])]
GENRES = ["trap", "drill", "hip hop", "lofi", "R&B", "afrobeats", "pop", "house", "techno", "dnb", "phonk",
          "cinematic"]
MOODS = ["dark", "bright", "warm", "gritty", "clean", "vintage", "aggressive", "dreamy", "punchy", "airy"]
LENGTHS = {"short": ("Short", "1 bar"), "medium": ("Medium", "2 bars"), "long": ("Long", "4 bars")}
LOOP_BARS = {"short": 1, "medium": 2, "long": 4}
_LENGTH_WORDS = {"short": "short, tight", "long": "long, sustained, long tail"}


def is_loop(sound_type: str) -> bool:
    return SOUND_TYPES[sound_type][1] == "loop"


def take_seconds(sound_type: str, takes: int, length: str, bar: float) -> float:
    """How much audio to ask for: enough to find ``takes`` good examples (the engine needs ≥10 s)."""
    if is_loop(sound_type):
        return float(min(45.0, max(10.5, LOOP_BARS[length] * bar * (takes + 2))))
    return 10.5 if takes <= 4 else 18.0
_WORDS = [("drum loop", "drum loop"), ("beat loop", "drum loop"), ("perc loop", "perc loop"),
          ("bass loop", "bass loop"), ("bassline", "bass loop"), ("chord loop", "chord loop"),
          ("melody loop", "melody loop"), ("vocal loop", "vocal loop"), ("loop", "melody loop"),
          ("808", "808"), ("kick", "kick"), ("snare", "snare"), ("clap", "clap"), ("open hat", "open hat"),
          ("ride", "ride"), ("crash", "crash"), ("cymbal", "crash"),
          ("hat", "hi-hat"), ("hihat", "hi-hat"), ("rim", "rim"), ("shaker", "perc"), ("perc", "perc"),
          ("tom", "perc"), ("sub", "sub bass"), ("bass", "bass"), ("piano", "keys"), ("rhodes", "keys"),
          ("keys", "keys"), ("organ", "keys"), ("bell", "bells"), ("mallet", "bells"), ("glock", "bells"),
          ("music box", "bells"), ("guitar", "guitar"), ("string", "strings"), ("violin", "strings"),
          ("cello", "strings"), ("brass", "brass"), ("trumpet", "brass"), ("horn", "brass"),
          ("flute", "flute"), ("choir", "choir"), ("ahh", "choir"), ("lead", "lead"), ("pluck", "lead"),
          ("pad", "pad"), ("drone", "pad"), ("stab", "stab"), ("chop", "vocal chop"), ("vocal", "vocal chop"),
          ("vox", "vocal chop"), ("riser", "riser"), ("build", "riser"), ("sweep", "riser"),
          ("impact", "impact"), ("boom", "impact"), ("fx", "fx")]


def guess_type(text: str) -> str:
    t = (text or "").lower()
    return next((kind for word, kind in _WORDS if word in t), "lead")


def caption(sound_type: str, description: str, genre: str = "", moods=(), length: str = "medium") -> str:
    """The user's words first (the caption matters most), then genre/mood, then the tags that keep
    it to one kind of sound."""
    words = [description.strip()] if description.strip() else []
    style = " ".join(x for x in (*moods, genre) if x)  # "dark gritty trap"
    if style:
        words.append(style)
    if length in _LENGTH_WORDS and not is_loop(sound_type):
        words.append(_LENGTH_WORDS[length])
    words.append(SOUND_TYPES[sound_type][0])
    return ", ".join(words)


def variation_source(sound: np.ndarray, sound_type: str, beat: float, seconds: float = 10.5) -> np.ndarray:
    """A pad's sound repeated into ~``seconds`` of audio, for the engine to remake ("cover") into
    variations: hits on the beat for one-shots, back to back for notes, sustains and loops."""
    n = int(seconds * SR)
    out = np.zeros((n, 2), np.float32)
    if not len(sound):
        return out
    how = SOUND_TYPES[sound_type][1]
    if how in ("loop", "sustain"):
        hop = len(sound)                           # back to back
    elif how.startswith("drum") or how == "hit":
        hop = int(beat * SR)                       # a hit on every beat
    else:
        hop = max(len(sound), int(beat * SR))      # notes / long sounds: one after another
    fade = min(len(sound) // 4, int(0.01 * SR))
    piece = sound.astype(np.float32).copy()
    if fade:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)[:, None]
        piece[:fade] *= ramp
        piece[-fade:] *= ramp[::-1]
    for a in range(0, n, max(1, hop)):
        m = min(len(piece), n - a)
        out[a:a + m] += piece[:m]
    return out


def _loops(audio: np.ndarray, bars: int, bar: float, count: int, name: str = "loop") -> list[tuple[int, int, str]]:
    """Whole-bar windows that loop well: steady level across the beats, a hit on the downbeat."""
    mono = audio.mean(axis=1)
    win, step = int(round(bars * bar * SR)), int(round(bar * SR))
    first = chop.onsets(mono[:step * 2], bar / 16) if len(mono) > step else []
    origin = int(first[0]) if len(first) else 0  # line the bar grid up with where the music starts
    beats = bars * 4
    scored = []
    for s in range(origin, len(mono) - win + 1, step):
        seg = mono[s:s + win]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms < 1e-3:
            continue
        per_beat = np.sqrt((seg[: win // beats * beats].reshape(beats, -1) ** 2).mean(axis=1))
        even = per_beat.min() / (per_beat.max() + 1e-9)
        downbeat = np.abs(seg[: int(0.04 * SR)]).max() / (np.abs(seg).max() + 1e-9)
        scored.append((rms * (0.4 + even) * (0.5 + downbeat), s))
    picks: list[int] = []
    for _score, s in sorted(scored, reverse=True):
        if all(abs(s - p) >= win for p in picks):
            picks.append(s)
        if len(picks) == count:
            break
    unit = "bar" if bars == 1 else "bars"
    return [(s, s + win, f"{name} {i} ({bars} {unit})") for i, s in enumerate(sorted(picks), 1)]


def _distinct(cands: list[tuple], key, count: int, rel: float = 0.15) -> list[tuple]:
    out: list[tuple] = []
    for c in cands:
        if all(abs(key(c) - key(o)) > rel * max(abs(key(c)), 1e-9) for o in out):
            out.append(c)
        if len(out) == count:
            break
    return out


def candidates(audio: np.ndarray, sound_type: str, beat: float, bar: float,
               count: int = 4, length: str = "medium") -> list[tuple[int, int, str]]:
    """Up to ``count`` (start, end, label) examples of ``sound_type`` found in ``audio``.
    ``length``: short / medium / long sounds (for loops: 1 / 2 / 4 bars)."""
    how = SOUND_TYPES[sound_type][1]
    if how == "loop":
        return _loops(audio, LOOP_BARS[length], bar, count, sound_type)
    if how.startswith("drum:"):
        picks = _drum_candidates(audio, how, beat, count)
        if length == "short":  # tight hits: cut each at 0.2 s
            picks = [(s, min(e, s + int(0.2 * SR)), label) for s, e, label in picks]
        return picks
    return _tonal_candidates(audio, sound_type, how, beat, bar, count, length)


def _drum_candidates(audio: np.ndarray, how: str, beat: float, count: int) -> list[tuple[int, int, str]]:
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


def _tonal_candidates(audio, sound_type, how, beat, bar, count, length) -> list[tuple[int, int, str]]:
    if how == "sustain":
        win, hop = int({"short": 1.0, "medium": min(2.0, bar), "long": 4.0}[length] * SR), int(0.25 * SR)
        win = min(win, max(int(0.5 * SR), len(audio) // 2))
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

    longest = {"note": bar, "hit": bar, "long": 2 * bar}[how] * {"short": 0.35, "medium": 1.0, "long": 2.0}[length]
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


# --- filling a bank of pads from one vibe ------------------------------------------------------------
_BANK_CORE = [("kit", 3), ("808", 1), ("chords", 2), ("melody", 1), ("hook", 1)]       # the first 8 pads
_BANK_EXTRA = [("kit", 2), ("chords", 2), ("hook", 1), ("fx", 1), ("loop", 1), ("perc", 1)]  # up to 16
_BANK_LAYOUT = ["kit", "808", "perc", "loop", "chords", "melody", "hook", "fx"]  # low pads first: drums


def plan_bank(targets: list[int], include: list[str]) -> dict[str, list[int]]:
    """Which of ``targets`` each part goes on: the essentials first, extras if there's room; drums
    on the lowest pads (bottom rows), chords in the middle, hooks/fx on top."""
    counts: dict[str, int] = {}
    left = len(targets)
    for part, n in _BANK_CORE + _BANK_EXTRA:
        if part in include and left:
            k = min(n, left)
            counts[part] = counts.get(part, 0) + k
            left -= k
    plan, pads = {}, sorted(targets)
    for part in _BANK_LAYOUT:
        if counts.get(part):
            plan[part], pads = pads[:counts[part]], pads[counts[part]:]
    return plan
