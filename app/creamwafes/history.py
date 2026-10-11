"""Undo/redo by snapshotting project state.

Snapshots keep references to the same Track/Clip objects and audio arrays (never copies of
audio), so they're cheap; restoring puts the objects back with their old field values, which
keeps identity stable for anything else holding a reference (pending jobs, the UI).
"""

from __future__ import annotations

import copy

from .model import Project

_TRACK_FIELDS = ("name", "kind", "volume", "pan", "muted", "solo", "prompt", "fx")  # fx is replaced, never mutated
_CLIP_FIELDS = ("path", "start", "offset", "length", "audio")
_PAD_FIELDS = ("clip", "name", "gain", "pitch", "kind", "mode", "play", "degree", "flavour", "strum", "root")


def snapshot(project: Project) -> dict:
    return {
        "bpm": project.bpm, "key": project.key,
        "tracks": [(t, {f: getattr(t, f) for f in _TRACK_FIELDS}, copy.deepcopy(t.gen),
                    [(c, {f: getattr(c, f) for f in _CLIP_FIELDS}) for c in t.clips])
                   for t in project.tracks],
        "pads": [(pad, {f: getattr(pad, f) for f in _PAD_FIELDS}) for pad in project.pads],
        "chords": copy.deepcopy(project.chords),
    }


def restore(project: Project, snap: dict) -> None:
    project.bpm, project.key = snap["bpm"], snap["key"]
    project.tracks = []
    for track, fields, gen, clips in snap["tracks"]:
        for name, value in fields.items():
            setattr(track, name, value)
        track.gen = copy.deepcopy(gen)
        track.clips = []
        for clip, clip_fields in clips:
            for name, value in clip_fields.items():
                setattr(clip, name, value)
            track.clips.append(clip)
        project.tracks.append(track)
    project.pads = [pad for pad, _ in snap["pads"]]  # the tray may have grown since
    for pad, fields in snap["pads"]:
        for name, value in fields.items():
            setattr(pad, name, value)
    project.chords = copy.deepcopy(snap["chords"])


class History:
    def __init__(self, depth: int = 100):
        self.depth = depth
        self._undo: list[tuple[str, dict]] = []
        self._redo: list[tuple[str, dict]] = []

    def checkpoint(self, project: Project, label: str) -> None:
        """Call *before* changing the project."""
        self._undo.append((label, snapshot(project)))
        del self._undo[:-self.depth]
        self._redo.clear()

    def undo(self, project: Project) -> str | None:
        return self._step(project, self._undo, self._redo)

    def redo(self, project: Project) -> str | None:
        return self._step(project, self._redo, self._undo)

    def next_undo(self) -> str | None:
        return self._undo[-1][0] if self._undo else None

    def next_redo(self) -> str | None:
        return self._redo[-1][0] if self._redo else None

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    @staticmethod
    def _step(project: Project, source: list, target: list) -> str | None:
        if not source:
            return None
        label, snap = source.pop()
        target.append((label, snapshot(project)))
        restore(project, snap)
        return label
