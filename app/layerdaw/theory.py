"""Keys, scales and diatonic chords for the chord pads (Orchid-style: pick a degree + a flavour)."""

from __future__ import annotations

import re

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_PC = {**{n.lower(): i for i, n in enumerate(NOTE_NAMES)}, **{n.lower(): i for i, n in enumerate(SHARP_NAMES)},
       "db": 1, "gb": 6, "cb": 11, "e#": 5, "fb": 4, "b#": 0}
SCALES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}
ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii"]
FLAVOURS = ["triad", "7th", "sus2", "sus4", "add9"]
# scale steps above the chord's root for each flavour (diatonic, so qualities follow the key)
FLAVOUR_STEPS = {"triad": [0, 2, 4], "7th": [0, 2, 4, 6], "sus2": [0, 1, 4], "sus4": [0, 3, 4],
                 "add9": [0, 2, 4, 8]}


def parse_key(text: str) -> tuple[int, str]:
    """'F# minor' / 'F#m' / 'Eb major' / 'A' → (pitch class, 'major' | 'minor')."""
    m = re.match(r"\s*([A-Ga-g])([#b]?)\s*(.*)$", text or "")
    if not m:
        return 9, "minor"
    pc = _PC.get((m.group(1) + m.group(2)).lower(), 9)
    rest = m.group(3).strip().lower()
    mode = "minor" if rest.startswith(("min", "m")) and not rest.startswith("maj") else "major"
    return pc, mode


def root_midi(pc: int) -> int:
    """Put the key's root in a comfortable chord register: E3 (52) … D#4 (63)."""
    return 52 + (pc - 4) % 12


def degree_midi(pc: int, mode: str, step: int) -> int:
    scale = SCALES[mode]
    return root_midi(pc) + 12 * (step // 7) + scale[step % 7]


def chord_notes(key: str, degree: int, flavour: str = "triad") -> list[int]:
    """MIDI notes of the diatonic chord on ``degree`` (0 = tonic, 7 = tonic an octave up)."""
    pc, mode = parse_key(key)
    return [degree_midi(pc, mode, degree + s) for s in FLAVOUR_STEPS.get(flavour, FLAVOUR_STEPS["triad"])]


def chord_name(key: str, degree: int, flavour: str = "triad") -> str:
    notes = chord_notes(key, degree, "7th")
    root = notes[0] % 12
    third, fifth, seventh = (n - notes[0] for n in notes[1:])
    quality = {(4, 7): "", (3, 7): "m", (3, 6): "°", (4, 8): "+"}.get((third, fifth), "")
    name = SHARP_NAMES[root] if parse_key(key)[0] in (1, 3, 6, 8, 10, 2, 4, 9, 11) else NOTE_NAMES[root]
    if flavour == "7th":
        if quality == "°":
            return name + ("ø7" if seventh == 10 else "°7")
        return name + quality + ("maj7" if seventh == 11 else "7")
    if flavour in ("sus2", "sus4"):
        return name + flavour
    return name + quality + ("add9" if flavour == "add9" else "")


def roman(key: str, degree: int) -> str:
    notes = chord_notes(key, degree)
    third, fifth = notes[1] - notes[0], notes[2] - notes[0]
    r = ROMAN[degree % 7]
    r = r.upper() if third == 4 else r
    return r + ("°" if fifth == 6 else "") + ("'" if degree >= 7 else "")


def note_name(midi: float) -> str:
    m = int(round(midi))
    return f"{SHARP_NAMES[m % 12]}{m // 12 - 1}"


# --- progressions -----------------------------------------------------------------------------
# Scale-degree progressions (0 = tonic) that work in each mode, grouped by mood.
PROGRESSIONS = {
    "minor": {
        "dark": [[0, 5, 2, 6], [0, 6, 5, 6], [0, 3, 6, 5], [0, 5, 3, 4]],
        "sad": [[0, 3, 5, 4], [0, 5, 3, 4], [5, 3, 0, 4], [0, 2, 5, 3]],
        "epic": [[0, 5, 6, 0], [0, 5, 2, 6], [5, 6, 0, 0], [0, 6, 5, 4]],
        "trap": [[0, 5, 3, 4], [0, 6, 5, 4], [0, 1, 5, 4], [0, 0, 5, 6]],
        "chill": [[0, 3, 6, 2], [0, 6, 5, 3], [3, 0, 6, 5]],
        "jazzy": [[3, 6, 2, 5], [1, 4, 0, 5], [0, 3, 1, 4]],
        "uplifting": [[5, 6, 0, 0], [2, 6, 0, 3], [5, 6, 2, 0]],
        "pop": [[0, 5, 2, 6], [5, 6, 0, 3], [0, 6, 5, 6]],
    },
    "major": {
        "pop": [[0, 4, 5, 3], [0, 5, 3, 4], [5, 3, 0, 4], [3, 0, 4, 5]],
        "uplifting": [[3, 4, 5, 0], [0, 4, 5, 3], [3, 0, 4, 5]],
        "sad": [[5, 3, 0, 4], [0, 2, 5, 3], [5, 4, 3, 0]],
        "chill": [[0, 5, 3, 4], [3, 2, 1, 0], [0, 2, 3, 3]],
        "jazzy": [[1, 4, 0, 5], [0, 5, 1, 4], [3, 2, 1, 4]],
        "dark": [[5, 3, 4, 2], [0, 6, 3, 0], [5, 2, 3, 4]],
        "epic": [[5, 3, 0, 4], [0, 4, 5, 3], [3, 4, 5, 5]],
        "trap": [[5, 3, 0, 4], [5, 4, 3, 4], [0, 5, 3, 4]],
    },
}
MOOD_WORDS = {
    "dark": ("dark", "moody", "evil", "sinister", "horror", "cyberpunk", "drill", "menacing", "gloomy"),
    "sad": ("sad", "emotional", "melancholy", "heartbreak", "lonely", "cry", "nostalgic"),
    "epic": ("epic", "cinematic", "anthem", "big", "heroic", "trailer", "orchestral"),
    "trap": ("trap", "hip hop", "hiphop", "rap", "808", "phonk"),
    "chill": ("chill", "lofi", "lo-fi", "relax", "ambient", "dreamy", "calm"),
    "jazzy": ("jazz", "jazzy", "neo soul", "r&b", "rnb", "soul", "smooth"),
    "uplifting": ("uplifting", "happy", "bright", "euphoric", "house", "edm", "dance", "hopeful"),
    "pop": ("pop", "catchy", "radio"),
}
JAZZY_FLAVOUR = {"jazzy": "7th", "chill": "7th"}


def mood_of(text: str) -> str | None:
    t = (text or "").lower()
    for mood, words in MOOD_WORDS.items():
        if any(w in t for w in words):
            return mood
    return None


def write_progression(key: str, mood_text: str = "", rng=None) -> dict:
    """A 4-chord progression in ``key`` for a mood ("dark", "lofi", "cinematic"…).
    Returns {"mood", "degrees", "flavour", "names", "romans"}."""
    import random
    rng = rng or random.Random()
    _pc, mode = parse_key(key)
    mood = mood_of(mood_text) or rng.choice(list(PROGRESSIONS[mode]))
    degrees = list(rng.choice(PROGRESSIONS[mode][mood]))
    flavour = JAZZY_FLAVOUR.get(mood, "triad")
    return {"mood": mood, "degrees": degrees, "flavour": flavour,
            "names": [chord_name(key, d, flavour) for d in degrees],
            "romans": [roman(key, d) for d in degrees]}


def pad_slots(degrees: list[int]) -> list[int]:
    """The progression's distinct chords in order, topped up with useful ones to fill 4 pads."""
    slots: list[int] = []
    for d in degrees + [0, 3, 4, 5, 6, 2, 1]:
        if d not in slots:
            slots.append(d)
    return slots[:4]
