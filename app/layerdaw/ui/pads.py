"""Pad logic. Every pad is the same kind of thing: a sound that plays as a sample, a chord of the
key, or a note of the key. The inspector edits the selected pad (or pads); nothing is tied to rows."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor

from .. import chop, theory
from ..audio import Note, render_note
from ..chords import PadInstrument
from ..diag import log
from ..model import SR, Clip, Track, load_audio
from .pad_inspector import CHOPS as _ALL_CHOPS
from .pad_grid import CHORD_COLOR, COLUMNS, KEY_TO_PAD, KEYED, MIDI_BASE, family, guess_kind, note_to_pad, pad_key  # noqa: F401
from .timeline import KIND_COLORS

CHOPS = [(label, mode) for label, mode in _ALL_CHOPS if mode not in (None, "file")]
DRUM_KINDS = {"kick", "snare", "clap", "hi-hat", "open hat", "perc", "drums"}
LOW_KINDS = {"808", "bass"}


class PadsMixin:
    """Mixed into MainWindow."""

    # --- selection -------------------------------------------------------------------------
    @property
    def selected_pads(self) -> list[int]:
        return self.pad_panel.grid.selected

    @property
    def primary_pad(self) -> int:
        return self.selected_pads[-1] if self.selected_pads else 0

    def select_pad(self, index: int) -> None:
        self.pad_panel.grid.select([index])

    def select_pads(self, indices: list[int]) -> None:
        self.pad_panel.grid.select(indices)

    # --- what a pad plays ------------------------------------------------------------------
    @property
    def instrument(self) -> PadInstrument:
        if not hasattr(self, "_instrument"):
            self._instrument = PadInstrument()
        return self._instrument

    def flavour_for(self, pad) -> str:
        """A held flavour key (G/H/B/N) wins over the pad's own flavour."""
        held = getattr(self, "_held_flavours", [])
        return held[-1] if held else pad.flavour

    def key_root(self) -> int:
        return theory.root_midi(theory.parse_key(self.project.key)[0])

    def chord_midis(self, pad) -> list[int]:
        return theory.chord_notes(self.project.key, pad.degree, self.flavour_for(pad))

    def note_midi(self, pad, degree: int | None = None) -> int:
        pc, mode = theory.parse_key(self.project.key)
        midi = theory.degree_midi(pc, mode, pad.degree if degree is None else degree)
        return midi - (24 if pad.kind in LOW_KINDS else 0)

    def pad_label(self, pad) -> tuple[str, str] | None:
        key = self.project.key
        if pad.play == "chord":
            return theory.chord_name(key, pad.degree, self.flavour_for(pad)), theory.roman(key, pad.degree)
        if pad.play == "note":
            return theory.note_name(self.note_midi(pad)), theory.roman(key, pad.degree)
        return None

    def _pad_voices(self, index: int, velocity: float) -> list[Note]:
        pad = self.project.pads[index]
        if pad.play == "chord":
            notes = self.chord_midis(pad)
            gain = velocity * pad.gain * 0.9 / np.sqrt(len(notes))
            strum = 0.035 if pad.strum else 0.0
            out = []
            for k, midi in enumerate(notes):
                buf, loop = self.instrument.note(pad, midi, self.key_root())
                out.append(Note(buf, gain, loop, k * strum))
            return out
        if pad.play == "note":
            buf, loop = self.instrument.note(pad, self.note_midi(pad), self.key_root())
            return [Note(buf, pad.gain * velocity, loop)]
        if pad.clip is None:
            return []
        gain = pad.gain * velocity
        if pad.mode == "hold":
            buf, loop = pad.freeze_buffer()
            return [Note(buf, gain, loop)]
        if pad.mode == "playon":
            return [Note(pad.continue_buffer(), gain, gate=True)]
        if pad.mode == "loop":
            buf, loop = pad.hold_buffer()
            return [Note(buf, gain, loop)]
        return [Note(pad.buffer(), gain)]

    def hit_pad(self, index: int, velocity: float = 1.0) -> None:
        if not 0 <= index < len(self.project.pads):
            return  # e.g. a MIDI note past the end of the tray
        self.pad_panel.grid.flash(index)
        voices = self._pad_voices(index, velocity)
        pad = self.project.pads[index]
        log.info("pad %d hit vel=%.2f play=%s mode=%s voices=%d name=%r", index, velocity, pad.play, pad.mode,
                 len(voices), pad.name)
        tag = ("pad", index)
        for note in voices:
            self.player.play_note(note, tag, choke=note.sustains and note.delay == 0)
        if voices and self.pad_panel.rec.isChecked() and self.player.playing:
            self._pad_hits.append({"t": self.player.position, "end": None, "tag": tag, "voices": voices})

    def release_pad(self, index: int) -> None:
        log.info("pad %d release", index)
        tag = ("pad", index)
        self.player.release(tag)
        for hit in reversed(self._pad_hits):
            if hit["tag"] == tag and hit["end"] is None:
                hit["end"] = self.player.position
                break

    def _audition(self, index: int) -> None:
        self.hit_pad(index)
        QTimer.singleShot(450, lambda: self.release_pad(index))

    @staticmethod
    def default_mode(kind: str) -> str:
        return "oneshot" if kind in DRUM_KINDS or kind == "percussion" else "hold"

    # --- view ------------------------------------------------------------------------------
    # --- the tray: pads below the 4×4 that never run out ------------------------------------
    def _ensure_pads(self, count: int) -> None:
        from ..model import Pad
        while len(self.project.pads) < count:
            self.project.pads.append(Pad())

    @staticmethod
    def _has_sound(pad) -> bool:
        return pad.clip is not None or pad.play != "sample"

    def _fit_tray(self) -> None:
        """Always one empty tray row after the last pad with a sound; no more than that."""
        used = [i for i, p in enumerate(self.project.pads) if self._has_sound(p) and i >= KEYED]
        want = KEYED + COLUMNS * ((used[-1] - KEYED) // COLUMNS + 2 if used else 1)
        self._ensure_pads(want)
        while len(self.project.pads) > want and not self._has_sound(self.project.pads[-1]):
            self.project.pads.pop()

    def tray_slots(self, count: int) -> list[int]:
        """``count`` empty tray pads, in order (the tray grows to fit)."""
        out, i = [], KEYED
        while len(out) < count:
            self._ensure_pads(i + 1)
            if not self._has_sound(self.project.pads[i]) and i not in self.pad_panel.grid.generating:
                out.append(i)
            i += 1
        return out

    def add_to_tray(self, track, clip, bars: bool = False) -> int | None:
        """Put a clip (or just the selected bars of it) on the next empty tray pad."""
        lo, hi = 0.0, clip.length
        if bars and self.state.selection:
            lo = max(0.0, self.state.selection[0] - clip.start)
            hi = min(clip.length, self.state.selection[1] - clip.start)
            if hi - lo < 0.03:
                self.engine_label.setText("The selected bars don't overlap that clip")
                return None
        slot = self.tray_slots(1)[0]
        kind = guess_kind(track.name) if track.kind in ("song", "audio") else track.kind
        self.pads_dock.show()
        self._put_on_pad(slot, clip.path, clip.offset + lo, hi - lo, f"{track.name}" + (" bars" if bars else ""), kind)
        self.select_pad(slot)
        self.pad_panel.grid.show_pad(slot)
        self.engine_label.setText(f"Added to the tray as pad {pad_key(slot)}")
        return slot

    def select_all_pads(self) -> None:
        used = [i for i, p in enumerate(self.project.pads) if self._has_sound(p)]
        self.select_pads(used or list(range(KEYED)))

    def refresh_pads(self) -> None:
        self._fit_tray()
        grid = self.pad_panel.grid
        grid.set_count(len(self.project.pads))
        kept = [i for i in grid.selected if i < len(self.project.pads)] or [0]
        if kept != grid.selected:  # the tray shrank (undo…): forget pads that are gone
            grid.selected = kept
            for b in grid.buttons:
                b.selected = b.index in kept
        for b, pad in zip(self.pad_panel.grid.buttons, self.project.pads):
            color = QColor(CHORD_COLOR) if pad.play == "chord" else QColor(KIND_COLORS.get(family(pad.kind), "#8e9bb0"))
            b.set_pad(pad, color, self.pad_label(pad))
        self.refresh_inspector()

    def refresh_inspector(self, _indices=None) -> None:
        sel = self.selected_pads
        if len(sel) > 1:
            self.pad_panel.inspector.show_multi(sel)
            return
        i = self.primary_pad
        pad = self.project.pads[i]
        key = self.project.key
        if pad.play == "chord":
            labels = [f"{theory.roman(key, d)}   {theory.chord_name(key, d, pad.flavour)}" for d in range(8)]
        else:
            labels = [f"{theory.roman(key, d)}   {theory.note_name(self.note_midi(pad, d))}" for d in range(8)]
        held = getattr(self, "_held_flavours", [])
        self.pad_panel.inspector.show_single(i, pad, labels, held[-1] if held else None)

    # --- editing the selected pad ------------------------------------------------------------
    def _edit(self, label: str, change, audition: bool = True) -> None:
        i = self.primary_pad
        pad = self.project.pads[i]
        self._checkpoint(label)
        change(pad)
        self.refresh_pads()
        self._changed()
        if audition and (pad.clip is not None or pad.play != "sample"):
            self._audition(i)

    def rename_pad(self, name: str) -> None:
        if name != self.project.pads[self.primary_pad].name:
            self._edit("Rename pad", lambda p: setattr(p, "name", name), audition=False)

    def set_pad_category(self, kind: str) -> None:
        self._edit("Pad category", lambda p: setattr(p, "kind", kind), audition=False)

    def set_pad_play(self, play: str) -> None:
        def change(p):
            if play != "sample" and p.play == "sample":
                p.mode = "hold"  # chords and notes sustain while held (One-shot is still there)
                if p.kind in DRUM_KINDS:
                    p.kind = "keys" if play == "chord" else "lead"
            p.play = play
            if play != "sample" and p.clip is None and not p.name:
                p.name = "synth"
        self._edit(f"Play as {play}", change)

    def set_pad_mode(self, mode: str) -> None:
        self._edit("Pad mode", lambda p: setattr(p, "mode", mode))

    def set_pad_degree(self, degree: int) -> None:
        self._edit("Pad chord/note", lambda p: setattr(p, "degree", degree))

    def set_pad_flavour(self, flavour: str) -> None:
        self._edit("Chord flavour", lambda p: setattr(p, "flavour", flavour))

    def set_pad_strum(self, on: bool) -> None:
        self._edit("Strum", lambda p: setattr(p, "strum", on))

    def nudge_pad_pitch(self, semitones: int) -> None:
        self._edit("Tune pad", lambda p: setattr(p, "pitch", int(max(-24, min(24, p.pitch + semitones)))))

    def nudge_pad_gain(self, factor: float) -> None:
        self._edit("Pad volume", lambda p: setattr(p, "gain", float(max(0.05, min(4.0, p.gain * factor)))))

    def edit_pad(self, attr: str, value) -> None:  # kept for menus/tests: "mode" | "pitch" | "gain"
        {"mode": self.set_pad_mode, "pitch": self.nudge_pad_pitch, "gain": self.nudge_pad_gain}[attr](value)

    # --- several pads at once ---------------------------------------------------------------
    def _targets(self) -> list[int]:
        """The selected pads in grid order (bottom-left first), or every pad if just one is selected."""
        sel = self.selected_pads
        return sorted(sel) if len(sel) > 1 else list(range(KEYED))

    def set_selected_play(self, play: str) -> None:
        self._checkpoint(f"Play as {play}")
        for i in self.selected_pads:
            pad = self.project.pads[i]
            if play != "sample" and pad.play == "sample":
                pad.mode = "hold"
                if pad.kind in DRUM_KINDS and pad.clip is None:
                    pad.kind = "keys" if play == "chord" else "lead"
            pad.play = play
        self.refresh_pads()
        self._changed()

    def set_selected_pitch(self, semitones: int) -> None:
        """Tune every selected pad to ``semitones`` (0 = as recorded)."""
        self._checkpoint("Tune pad")
        for i in self.selected_pads:
            self.project.pads[i].pitch = int(max(-24, min(24, semitones)))
        self.refresh_pads()
        self._changed()
        self._audition(self.primary_pad)

    def set_selected_mode(self, mode: str) -> None:
        self._checkpoint("Pad mode")
        for i in self.selected_pads:
            self.project.pads[i].mode = mode
        self.refresh_pads()
        self._changed()

    def set_all_modes(self, mode: str) -> None:
        self._checkpoint("Pad mode")
        for pad in self.project.pads:
            pad.mode = mode
        self.refresh_pads()
        self._changed()

    def clear_selected(self) -> None:
        self._checkpoint("Clear pads")
        for i in self.selected_pads:
            self.project.pads[i].clear()
        self.refresh_pads()
        self._changed()

    def clear_pads(self) -> None:
        self._checkpoint("Clear pads")
        for pad in self.project.pads:
            pad.clear()
        self.refresh_pads()
        self._changed()

    # --- putting sounds on pads -------------------------------------------------------------
    def _put_on_pad(self, index: int, path: Path, offset: float, length: float, name: str, kind: str) -> None:
        self._checkpoint("Replace pad")
        pad = self.project.pads[index]
        keep_play = pad.play if pad.play != "sample" else None  # a chord pad stays a chord pad: new timbre
        mode = pad.mode if pad.clip is not None or keep_play else self.default_mode(kind)
        degree, flavour, strum = pad.degree, pad.flavour, pad.strum
        pad.clear()
        pad.clip = Clip(path, 0.0, offset, length)
        pad.name, pad.kind, pad.mode = name, kind, mode
        if keep_play:
            pad.play, pad.degree, pad.flavour, pad.strum = keep_play, degree, flavour, strum
        self.refresh_pads()
        self._changed()
        log.info("pad %d ← %s @%.2fs (%.2fs)", index, name, offset, length)
        self._audition(index)

    def _copy_into_song(self, path: Path) -> Path:
        self.project.audio_dir.mkdir(parents=True, exist_ok=True)
        dest = self.project.audio_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{path.name}"
        shutil.copy2(path, dest)
        return dest

    def _file_hits(self, path: Path, t0: float, t1: float) -> list[float]:
        """Hit times (seconds into the file) between t0 and t1."""
        full = load_audio(path)
        # analyse from a bit before t0: starting mid-sound would make the sound already playing
        # there look like a fresh hit at t0
        a, b = max(0, int((t0 - 0.3) * SR)), min(len(full), int(t1 * SR))
        if b - a < 2048:
            return []
        hits = [(a + int(s)) / SR for s in chop.onsets(full[a:b].mean(axis=1), self.project.beat_seconds / 4 * 0.75)]
        return [t for t in hits if t >= t0 - 1e-3]

    def _slice_from(self, path: Path, start: float) -> tuple[float, float]:
        """A hit starting at ``start`` that ends at the next hit (at most 2 beats)."""
        nxt = [t for t in self._file_hits(path, start + 0.02, start + 2 * self.project.beat_seconds + 0.5)
               if t > start + 0.03]
        end = nxt[0] if nxt else start + 2 * self.project.beat_seconds
        return start, min(end, start + 2 * self.project.beat_seconds) - start

    def replace_pad(self, what: str) -> None:
        i = self.primary_pad
        pad = self.project.pads[i]
        if what == "clear":
            self._checkpoint("Clear pad")
            pad.clear()
            self.refresh_pads()
            self._changed()
            return
        if what == "file":
            self.load_pad_from_file(i)
            return
        if what == "generate":
            self.design_pad_sound(i)
            return
        if what in ("next", "prev"):
            if pad.clip is None:
                self.engine_label.setText("Load something on this pad first, then step through its hits")
                return
            here = pad.clip.offset
            gap = max(0.03, self.project.beat_seconds / 8)  # a genuinely different hit, not this one again
            if what == "next":
                hits = self._file_hits(pad.clip.path, here + gap, here + 8.0)
                target = hits[0] if hits else None
            else:
                hits = [t for t in self._file_hits(pad.clip.path, max(0.0, here - 8.0), here) if t < here - gap]
                target = hits[-1] if hits else None
            if target is None:
                self.engine_label.setText(f"No {'later' if what == 'next' else 'earlier'} hit in that audio")
                return
            start, length = self._slice_from(pad.clip.path, target)
            self._put_on_pad(i, pad.clip.path, start, length, pad.name, pad.kind)
            return
        clip = self.state.selected_clip
        track = self._track_of(clip) if clip is not None else None
        if clip is None:
            self.engine_label.setText("Click a clip on the timeline first")
            return
        if what == "clip":
            self._put_on_pad(i, clip.path, clip.offset, clip.length, track.name, guess_kind(track.name)
                             if track.kind in ("song", "audio") else track.kind)
        elif what == "bars":
            if not self.state.selection:
                self.engine_label.setText("Drag on the ruler to select some bars first")
                return
            lo = max(0.0, self.state.selection[0] - clip.start)
            hi = min(clip.length, self.state.selection[1] - clip.start)
            if hi - lo < 0.03:
                self.engine_label.setText("The selected bars don't overlap that clip")
                return
            self._put_on_pad(i, clip.path, clip.offset + lo, hi - lo, f"{track.name} bars", track.kind)
        elif what == "playhead":
            t = self.player.position - clip.start
            if not 0 <= t < clip.length:
                self.engine_label.setText("Put the playhead inside the selected clip (click the ruler)")
                return
            near = [h for h in self._file_hits(clip.path, clip.offset + t - 0.06, clip.offset + t + 0.5)
                    if h >= clip.offset + t - 0.06]
            start = near[0] if near else clip.offset + t  # snap to the hit right at/after the playhead
            start, length = self._slice_from(clip.path, start)
            self._put_on_pad(i, clip.path, start, length, f"{track.name} hit", track.kind)

    def pad_file_dropped(self, index: int, path: str) -> None:
        self.select_pad(index)
        self.load_pad_from_file(index, path)

    def load_pad_from_file(self, index: int, path: str | None = None) -> None:
        from PySide6.QtWidgets import QFileDialog
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Load audio onto pad", str(Path.home()),
                                                  "Audio (*.wav *.flac *.mp3 *.ogg *.aiff *.aif)")
            if not path:
                return
        src = Path(path)
        try:
            dest = self._copy_into_song(src)
            length = Clip(dest).length
        except Exception as exc:  # noqa: BLE001 - unreadable/corrupt file: tell the user, don't crash
            self.engine_label.setText(f"Couldn't load {src.name}: {exc}")
            return
        self.pads_dock.show()
        self._put_on_pad(index, dest, 0.0, length, src.stem, guess_kind(src.stem))

    def load_pad_from_clip(self, index: int) -> None:
        self.select_pad(index)
        self.replace_pad("clip")

    # --- filling several pads -----------------------------------------------------------------
    def chop_to_pads(self, mode: str, track=None, clip=None, use_selection: bool = True,
                     targets: list[int] | None = None) -> None:
        clip = clip or self.state.selected_clip
        track = track or (self._track_of(clip) if clip is not None else None)
        if clip is None or track is None:
            self.engine_label.setText("Click a clip on the timeline first, then chop it onto the pads")
            return
        targets = targets if targets is not None else self._targets()
        lo, hi = 0, len(clip.audio)
        if use_selection and self.state.selection:  # only the selected bars of that clip
            a, b = (int(round((t - clip.start) * SR)) for t in self.state.selection)
            if b > 0 and a < len(clip.audio):
                lo, hi = max(0, a), min(len(clip.audio), b)
        audio = clip.audio[lo:hi]
        beat = self.project.beat_seconds
        roles: list[tuple[str, float, str]] = []
        if mode == "kit":  # one clean example of each drum sound, drum-machine order, levelled
            picks = chop.kit(audio, beat, len(targets))
            ranges = [(s, e) for s, e, _, _ in picks]
            seen: dict[str, int] = {}
            for _, _, role, gain in picks:
                seen[role] = seen.get(role, 0) + 1
                kind = {"hat": "hi-hat", "tom": "perc"}.get(role, role)
                roles.append((role if seen[role] == 1 else f"{role} {seen[role]}", gain, kind))
        else:
            ranges = (chop.by_hits(audio, beat) if mode == "hits" else
                      chop.by_grid(audio, {"beat": beat, "half": beat / 2, "bar": self.project.bar_seconds}[mode]))
        if not ranges:
            self.engine_label.setText("Couldn't find anything to chop there")
            return
        self._checkpoint("Chop to pads")
        ranges = ranges[:len(targets)]
        kind = guess_kind(track.name) if track.kind in ("song", "audio") else track.kind
        for n, index in enumerate(targets):
            pad = self.project.pads[index]
            pad.clear()
            if n < len(ranges):
                s, e = ranges[n]
                pad.clip = Clip(clip.path, 0.0, clip.offset + (lo + s) / SR, (e - s) / SR)
                pad.name = f"{track.name[:10]} {n + 1}"
                pad.kind, pad.mode = kind, self.default_mode(kind)
                if roles:
                    pad.name, pad.gain, pad.kind = roles[n]
                    pad.mode = "oneshot"
        self.pads_dock.show()
        self.refresh_pads()
        self.pad_panel.grid.show_pad(targets[min(len(ranges), len(targets)) - 1])
        self.engine_label.setText(f"Chopped {track.name} into {len(ranges)} pads "
                                  f"({pad_key(targets[0])}–{pad_key(targets[min(len(ranges), len(targets)) - 1])})")
        log.info("chop %s mode=%s → pads %s", track.name, mode, targets[:len(ranges)])
        self._changed()

    def chop_file_to_pads(self, targets: list[int] | None = None) -> None:
        from PySide6.QtWidgets import QFileDialog, QInputDialog
        path, _ = QFileDialog.getOpenFileName(self, "Chop an audio file onto the pads", str(Path.home()),
                                              "Audio (*.wav *.flac *.mp3 *.ogg *.aiff *.aif)")
        if not path:
            return
        modes = [label for label, _ in CHOPS]
        choice, ok = QInputDialog.getItem(self, "Chop", "Slice it:", modes, 0, False)
        if not ok:
            return
        src = Path(path)
        try:
            clip = Clip(self._copy_into_song(src))
        except Exception as exc:  # noqa: BLE001 - unreadable/corrupt file: tell the user, don't crash
            self.engine_label.setText(f"Couldn't load {src.name}: {exc}")
            return
        track = Track(name=src.stem[:12], kind=guess_kind(src.stem), clips=[clip])
        self.chop_to_pads(CHOPS[modes.index(choice)][1], track=track, clip=clip, use_selection=False, targets=targets)

    def generate_kit(self, targets: list[int] | None = None) -> None:
        from PySide6.QtWidgets import QInputDialog
        targets = targets if targets is not None else self._targets()
        text, ok = QInputDialog.getText(self, "Generate kit", f"Describe the drums ({len(targets)} pads):",
                                        text=getattr(self, "_last_kit", "punchy drum kit, kick snare hats claps"))
        if not ok or not text.strip():
            return
        self._last_kit = text.strip()
        self._submit_kit(text.strip(), targets)

    def _submit_kit(self, text: str, targets: list[int]) -> None:
        bars = max(4, int(np.ceil(12 / self.project.bar_seconds)))  # the engine needs ≥10 s
        params = self._base_params(f"{text}, solo drum kit, drum one-shots, clean studio drums", "[Instrumental]", 32, -1)
        params.update(duration=bars * self.project.bar_seconds, instrumental=True)
        gen = {"mode": "kit", "kind": "drums", "prompt": text, "to_pads": True, "targets": targets}
        self._submit("text2music", params, gen, name="kit → pads", mute=[], pads=targets)

    # --- right-click on a pad ------------------------------------------------------------------
    def pad_menu(self, index: int, pos) -> None:
        if index not in self.selected_pads:  # (the grid already did this for a real right-click)
            self.select_pad(index)
        pad = self.project.pads[index]
        sel = self.selected_pads
        many = len(sel) > 1
        key = pad_key(index)
        items = []
        if many:
            items += [(f"✦ Make a song (or a part) from these {len(sel)} pads…", self.open_song_maker),
                      (f"✦ Fill these {len(sel)} pads (a whole bank)…", lambda: self.fill_bank(sorted(sel))),
                      (f"✦ Sounds onto these {len(sel)} pads…", lambda: self.generate_sounds(sorted(sel))), None]
        items += [
            (f"✦ Generate a sound for pad {key}…", lambda: self.design_pad_sound(index)),
            *([(f"✦ Variation of {pad.name or 'this sound'}…", lambda: self.design_pad_sound(index, vary=True))]
              if pad.clip is not None else []),
            (f"Replace pad {key} with", [(label, lambda w=what: self.replace_pad(w)) for label, what in (
                ("The hit at the playhead (selected clip)", "playhead"), ("The selected bars", "bars"),
                ("The whole selected clip", "clip"), ("An audio file…", "file"),
                ("◀ Previous hit", "prev"), ("Next hit ▶", "next"))]),
            None,
        ]
        which = f" ({len(sel)} pads)" if many else ""
        play = self.set_selected_play if many else self.set_pad_play
        mode = self.set_selected_mode if many else self.set_pad_mode
        items += [
            ("Play as" + which, [(("● " if pad.play == p else "   ") + p.capitalize(), lambda p=p: play(p))
                                 for p in ("sample", "chord", "note")]),
            ("Mode" + which, [(("● " if pad.mode == m else "   ") + label, lambda m=m: mode(m))
                              for m, label in (("oneshot", "One-shot"), ("hold", "Hold"), ("loop", "Loop"),
                                               ("playon", "Play-on"))]),
            ("Pitch" + which, [(("● " if pad.pitch == st else "   ") + label, lambda st=st: self.set_selected_pitch(st))
                               for st, label in ((12, "+12 octave up"), (7, "+7 fifth up"), (5, "+5"), (2, "+2"),
                                                 (1, "+1"), (0, "0 as recorded"), (-1, "−1"), (-2, "−2"), (-5, "−5"),
                                                 (-7, "−7 fifth down"), (-12, "−12 octave down"))]),
            None,
            (f"Clear these {len(sel)} pads" if many else "Clear pad", self.clear_selected if many
             else lambda: self.replace_pad("clear")),
        ]
        self._show_menu(items, pos)

    # --- recording ---------------------------------------------------------------------------
    def _record_toggled(self, on: bool) -> None:
        if on:
            self._pad_hits = []
            if not self.player.playing:
                self.player.play()
            self.engine_label.setText("Recording pads… press ● again or stop playback to finish")
        else:
            self._commit_pad_take()

    def _commit_pad_take(self) -> None:
        hits, self._pad_hits = self._pad_hits, []
        if not hits:
            return
        now = self.player.position
        q = self.project.beat_seconds / 4 if self.pad_panel.quantize.isChecked() else None
        notes = []  # (start time, rendered audio)
        for hit in hits:
            t = (round(hit["t"] / q) * q) if q else hit["t"]
            held = max(0.0, (hit["end"] if hit["end"] is not None else now) - hit["t"])
            for note in hit["voices"]:
                audio = render_note(note, int(max(0.0, held - note.delay) * SR))
                notes.append((t + note.delay, audio))
        bar = self.project.bar_seconds
        start = np.floor(min(t for t, _ in notes) / bar) * bar
        end = max(t + len(a) / SR for t, a in notes)
        mix = np.zeros((int((end - start) * SR) + 2, 2), np.float32)
        for t, audio in notes:
            i = int(round((t - start) * SR))
            n = min(len(audio), len(mix) - i)
            mix[i:i + n] += audio[:n]
        path = self.project.write_wav(mix, "pads-take")
        self._checkpoint("Record pads")
        track = Track(name=self._unique_name("pads take"), kind="drums", clips=[Clip(path, start)])
        self.project.tracks.append(track)
        self.refresh()
        self._changed()
        self.engine_label.setText(f"Recorded {len(hits)} hits onto '{track.name}'")
        log.info("recorded %d hits onto %s (%.2fs)", len(hits), track.name, len(mix) / SR)
