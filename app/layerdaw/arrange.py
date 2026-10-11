"""Arrange the user's own pad sounds into a song: every pad gets a part (drum pattern, chord
progression, bass line, hooks…) across a song structure. Pure logic — rendering happens in the UI.

Times and lengths are in beats from the start of the song.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

STYLES = ["trap", "drill", "house", "lofi", "pop", "dnb"]

# 16 steps per bar; several variations per part, the last one is the end-of-section fill
PATTERNS: dict[str, dict[str, list[str]]] = {
    "trap": {"kick": ["x......x..x.....", "x.....x...x..x..", "x......x..x...x."],
             "snare": ["........x.......", "........x.......", "........x.....xx"],
             "clap": ["........x.......", "........x.......", "........x...x.x."],
             "hat": ["x.x.x.x.x.x.x.x.", "x.x.x.x.x.xxx.x.", "x.x.x.x.xxxxxxxx"],
             "openhat": ["..............x.", "......x.......x."], "perc": ["....x.......x...", "...x......x....."]},
    "drill": {"kick": ["x.....x.........", "x......x..x.....", "x.....x...x....."],
              "snare": ["........x.....x.", "........x.......", "........x.x..x.x"],
              "clap": ["........x.....x.", "........x......."],
              "hat": ["x..x..x.x..x..x.", "x..x..x.x..xx.x.", "x..x..x.x.xxxxxx"],
              "openhat": ["...........x...."], "perc": ["..x.......x....."]},
    "house": {"kick": ["x...x...x...x...", "x...x...x...x..."], "snare": ["....x.......x..."],
              "clap": ["....x.......x...", "....x.......x..x"], "hat": ["..x...x...x...x.", "..x...x...x..xx."],
              "openhat": ["..x...x...x...x."], "perc": ["...x..x....x..x.", "x..x..x...x..x.."]},
    "lofi": {"kick": ["x.........x.....", "x......x..x....."], "snare": ["....x.......x...", "....x.......x..x"],
             "clap": ["....x.......x..."], "hat": ["x.x.x.x.x.x.x.x.", "x.x.x.x.x.x.x.xx"],
             "openhat": ["..............x."], "perc": ["......x.......x."]},
    "pop": {"kick": ["x.......x.......", "x.......x.x....."], "snare": ["....x.......x...", "....x.......x.xx"],
            "clap": ["....x.......x..."], "hat": ["x.x.x.x.x.x.x.x.", "x.x.x.x.x.x.xxxx"],
            "openhat": ["..............x."], "perc": ["...x.......x...."]},
    "dnb": {"kick": ["x.........x.....", "x.........x..x.."], "snare": ["....x.......x...", "....x.......x.xx"],
            "clap": ["....x.......x..."], "hat": ["x.x.x.x.x.x.x.x.", "xxx.x.xxx.x.x.xx"],
            "openhat": ["..x...x...x...x."], "perc": ["..x.....x.x....."]},
}
SWING = {"lofi": 0.12, "drill": 0.04}
BASS_ON_KICK = {"trap", "drill", "dnb"}

# (section, bars in "short", bars in "full", parts playing, loudness)
SECTIONS = [
    ("intro", 4, 8, {"chords", "stab", "hat", "fx", "texture"}, 0.7),
    ("verse", 8, 16, {"kick", "snare", "clap", "hat", "perc", "bass", "chords", "stab", "texture"}, 0.85),
    ("chorus", 8, 16, {"kick", "snare", "clap", "hat", "openhat", "perc", "bass", "chords", "stab", "lead",
                       "hook", "texture"}, 1.0),
    ("break", 4, 8, {"chords", "hook", "fx", "perc", "texture"}, 0.75),
    ("chorus", 8, 16, {"kick", "snare", "clap", "hat", "openhat", "perc", "bass", "chords", "stab", "lead",
                       "hook", "texture"}, 1.0),
    ("outro", 0, 8, {"chords", "hat", "kick", "texture"}, 0.7),
]
# a section never thins out to (nearly) nothing: when the pads picked don't cover its usual parts,
# borrow these (in order) until at least MIN_PARTS of them play. Breaks stay kick/snare-free.
MIN_PARTS = {"intro": 2, "verse": 3, "chorus": 3, "break": 3, "outro": 2}
FALLBACK = {"intro": ["stab", "hat", "hook", "perc", "lead", "bass"],
            "verse": ["stab", "lead", "hook", "texture", "perc"],
            "chorus": ["texture", "perc", "fx"],
            "break": ["stab", "hat", "lead", "bass", "perc", "openhat", "hook"],
            "outro": ["stab", "hook", "hat", "perc", "lead"]}
DRUM_ROLES = {"kick", "snare", "clap", "hat", "openhat", "perc"}
KIND_ROLE = {"kick": "kick", "snare": "snare", "clap": "clap", "hi-hat": "hat", "open hat": "openhat",
             "perc": "perc", "percussion": "perc", "808": "bass", "bass": "bass", "keys": "stab", "keyboard": "stab",
             "stab": "stab", "lead": "stab", "synth": "stab", "pad": "texture", "hook": "hook", "vocals": "hook",
             "vocal chop": "hook", "fx": "fx"}
NAME_ROLE = [("open", "openhat"), ("kick", "kick"), ("snare", "snare"), ("clap", "clap"), ("hat", "hat"),
             ("tom", "perc"), ("perc", "perc"), ("808", "bass"), ("bass", "bass")]


@dataclass
class Event:
    beat: float                 # when, in beats from the song start
    length: float               # how long it's held, in beats
    velocity: float = 1.0
    degree: int | None = None   # chord / bass / lead: the chord of the key underneath
    tone: int | None = None     # lead: which note of that chord (0 root, 1 third, 2 fifth)


@dataclass
class Plan:
    style: str
    sections: list[tuple[str, int, int]]          # (name, start bar, bars)
    bars: int
    progression: list[int]
    roles: dict[int, str]
    events: dict[int, list[Event]] = field(default_factory=dict)


def auto_style(bpm: int) -> str:
    if bpm < 95:
        return "lofi"
    if bpm < 118:
        return "pop"
    if bpm < 132:
        return "house"
    if bpm < 160:
        return "trap"
    return "dnb"


def role_of(pad, classify=None) -> str:
    """What a pad does in the song: chords/bass/lead from how it plays, otherwise from its category,
    its name ("kick 2"), or (for a generic 'drums' pad) the sound itself via ``classify(pad)``."""
    if pad.play == "chord":
        return "chords"
    if pad.play == "note":
        return "bass" if pad.kind in ("808", "bass") else "lead"
    if pad.kind in KIND_ROLE:
        return KIND_ROLE[pad.kind]
    name = (pad.name or "").lower()
    for word, role in NAME_ROLE:
        if word in name:
            return role
    if classify is not None and pad.kind in ("drums", "other", "audio"):
        return classify(pad) or "perc"
    return "perc" if pad.kind in ("drums",) else "stab"


ROLES = ["kick", "snare", "clap", "hat", "openhat", "perc", "bass", "lead", "chords", "stab", "texture", "hook", "fx"]
ARPS = [(0, 1, 2), (2, 1, 0), (0, 2, 1), (1, 0, 2)]           # lead: which chord tones, per variation
STABS = [((0,), (0, 6)), ((0, 10), (0, 6, 10)), ((0, 3, 8), (0, 3, 8, 11))]  # stab steps: (verse, chorus)


def make_plan(pads: dict[int, object], *, bpm: int, key_degrees_default: list[int] | None = None,
              style: str = "auto", length: str = "full", hook_beats: dict[int, float] | None = None,
              classify=None, seed: int = 0, roles: dict[int, str] | None = None,
              variations: dict[int, int] | None = None) -> Plan:
    """Lay out every pad's part across a song. ``pads``: {pad index: Pad}. ``roles`` overrides the
    part a pad plays; ``variations`` picks another pattern for a pad's part (0 = the first one)."""
    rng = random.Random(seed)
    style = auto_style(bpm) if style == "auto" else style
    patterns = PATTERNS[style]
    bar_s = 240 / bpm
    var = variations or {}
    roles = {i: (roles or {}).get(i) or role_of(p, classify) for i, p in pads.items()}
    chord_pads = [i for i in sorted(pads) if roles[i] == "chords"]
    progression = [pads[i].degree for i in chord_pads] or (key_degrees_default or [0, 5, 3, 4])
    bars_per_chord = 2 if bar_s < 2.0 else 1
    sections, bar = [], 0
    for name, short, full, *_ in SECTIONS:
        n = short if length == "short" else full
        if n:
            sections.append((name, bar, n))
            bar += n
    plan = Plan(style, sections, bar, progression, roles, {i: [] for i in pads})
    hook_pads = [i for i in sorted(pads) if roles[i] == "hook"]
    by_role: dict[str, list[int]] = {}
    for i in sorted(pads):
        by_role.setdefault(roles[i], []).append(i)

    def chord_at(b: int) -> tuple[int, int]:
        k = (b // bars_per_chord) % len(progression)
        return k, progression[k]

    present = set(roles.values())
    for s_index, (name, start, n) in enumerate(sections):
        active, loud = next((set(a), v) for sn, _, _, a, v in SECTIONS if sn == name)
        for role in FALLBACK.get(name, []):
            if len(active & present) >= MIN_PARTS.get(name, 0):
                break
            active.add(role)
        if not active & present:  # nothing it would normally use: everything picked plays (sections set loudness)
            active |= present
        playing = active & present  # holding a part back only works when something else is playing
        for b in range(start, start + n):
            last_bar = b == start + n - 1
            pos = b - start
            beat0 = b * 4
            # drums: each drum pad plays its role's pattern (fills on the section's last bar)
            for role in DRUM_ROLES & active:
                for idx, i in enumerate(by_role.get(role, [])):
                    variants = patterns.get(role) or patterns.get("perc")
                    pat = (variants[-1] if last_bar and len(variants) > 2 else
                           variants[(pos // 2 + idx + var.get(i, 0)) % len(variants)])
                    if name == "intro" and role == "hat" and pos < n // 2 and playing - {"hat"}:
                        continue  # hats come in halfway through the intro
                    if name == "outro" and role == "kick" and pos >= n // 2 and playing - {"kick"}:
                        continue  # kick drops out for the last half of the outro
                    if idx > 0 and role in ("kick", "snare", "clap"):
                        pat = "".join(c if k % 8 == 0 else "." for k, c in enumerate(pat))  # layers on the downbeats
                    if var.get(i, 0):  # a variation adds a couple of ghost notes, different every 2 bars
                        vr = random.Random(hash((seed, i, var[i], pos // 2)))
                        steps = list(pat)
                        empty = [k for k, c in enumerate(steps) if c == "." and k % 4]
                        for k in vr.sample(empty, min(len(empty), 1 + var[i] % 3)):
                            steps[k] = "g"
                        pat = "".join(steps)
                    for step, c in enumerate(pat):
                        if c in "xg":
                            swing = SWING.get(style, 0) if step % 2 else 0
                            vel = loud * (1.0 if step % 4 == 0 else 0.8 if c == "x" else 0.5) * (0.9 + 0.1 * rng.random())
                            plan.events[i].append(Event(beat0 + step / 4 + swing / 2, 0.25, vel))
            k, degree = chord_at(b)
            change = b % bars_per_chord == 0
            if "chords" in active and change and chord_pads:  # each chord pad plays its own chord in turn
                plan.events[chord_pads[k]].append(Event(beat0, 4 * bars_per_chord - 0.1, loud * 0.85, degree=degree))
            if "bass" in active:
                for i in by_role.get("bass", []):
                    if style in BASS_ON_KICK and by_role.get("kick"):
                        hits = [st for st, c in enumerate(patterns["kick"][(pos // 2) % len(patterns["kick"])]) if c == "x"]
                    elif style == "house":
                        hits = [2, 6, 10, 14]
                    else:
                        hits = [0, 8]
                    for j, st in enumerate(hits):
                        nxt = hits[j + 1] if j + 1 < len(hits) else 16
                        plan.events[i].append(Event(beat0 + st / 4, (nxt - st) / 4 - 0.05, loud * 0.9,
                                                    degree=degree))
            if "lead" in active:
                for i in by_role.get("lead", []):
                    arp = ARPS[var.get(i, 0) % len(ARPS)]
                    for st in range(0, 16, 2):  # 8th-note arpeggio through the chord
                        plan.events[i].append(Event(beat0 + st / 4, 0.45, loud * 0.7, degree=degree,
                                                    tone=arp[(st // 2) % 3]))
            if "stab" in active:  # every bar (on chord changes only, a fast song leaves empty bars)
                for i in by_role.get("stab", []):
                    verse_steps, chorus_steps = STABS[var.get(i, 0) % len(STABS)]
                    for st in chorus_steps if name == "chorus" else verse_steps:
                        plan.events[i].append(Event(beat0 + st / 4, 1.0, loud * 0.8))
            if "texture" in active and change:
                for i in by_role.get("texture", []):
                    plan.events[i].append(Event(beat0, 4 * bars_per_chord - 0.1, loud * 0.6))
            if "hook" in active and hook_pads:
                every = max(1, int(math.ceil(max((hook_beats or {}).get(i, 4.0) for i in hook_pads) / 4)))
                tease = name == "break" and playing - {"hook"}  # the break teases it, unless it's all there is
                if pos % every == 0 and not (tease and pos >= max(1, n // 2)):
                    i = hook_pads[(pos // every) % len(hook_pads)]
                    plan.events[i].append(Event(beat0, every * 4 - 0.1, loud))
            nxt_name = sections[s_index + 1][0] if s_index + 1 < len(sections) else None
            if name == "break" and last_bar and nxt_name == "chorus":  # snare roll building into the chorus
                roll = (by_role.get("snare") or by_role.get("clap") or [])[:1]
                for i in roll:
                    for k, step in enumerate(range(8, 16)):
                        plan.events[i].append(Event(beat0 + step / 4, 0.25, loud * (0.45 + 0.07 * k)))
        # a riser/impact into every chorus
        nxt = sections[s_index + 1][0] if s_index + 1 < len(sections) else None
        if nxt == "chorus":
            for i in by_role.get("fx", []):
                plan.events[i].append(Event((start + n - 1) * 4, 4.0, 0.9))
    for i in plan.events:
        plan.events[i].sort(key=lambda e: e.beat)
    return plan
