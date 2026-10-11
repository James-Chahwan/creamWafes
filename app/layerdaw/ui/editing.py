"""Editing: undo/redo, delete/duplicate/split, rename, right-click menus, file drops."""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMenu

from ..diag import log
from ..model import AUDIO_EXTENSIONS, Clip, Track


class EditMixin:
    """Mixed into MainWindow; needs project, history, state, lanes, headers, refresh(), _changed()."""

    # --- undo --------------------------------------------------------------------------

    def _checkpoint(self, label: str, coalesce=None) -> None:
        """Snapshot before a change. Repeated changes with the same ``coalesce`` key within a
        second (wheel/arrow nudges on one slider) become a single undo step."""
        now = time.monotonic()
        if coalesce is not None and coalesce == self._last_coalesce[0] and now - self._last_coalesce[1] < 1.0:
            self._last_coalesce = (coalesce, now)
            return
        self._last_coalesce = (coalesce, now)
        self.history.checkpoint(self.project, label)
        self._update_actions()

    def undo(self) -> None:
        track = self.selected_track_obj()
        if (label := self.history.undo(self.project)) is not None:
            self._after_history(f"Undid: {label}", track)

    def redo(self) -> None:
        track = self.selected_track_obj()
        if (label := self.history.redo(self.project)) is not None:
            self._after_history(f"Redid: {label}", track)

    def _after_history(self, message: str, track: Track | None = None) -> None:
        log.info(message)
        self._last_coalesce = (None, 0.0)
        self.state.selected_clip = None
        # keep the selected track if it's still there (e.g. undoing a Layer panel tweak keeps the panel)
        self.state.selected_track = self.project.tracks.index(track) if track in self.project.tracks else None
        self.bpm.blockSignals(True)
        self.bpm.setValue(self.project.bpm)
        self.bpm.blockSignals(False)
        self.key.setCurrentText(self.project.key)
        self.engine_label.setText(message)
        self.refresh()
        self._changed()

    # --- track / clip operations -------------------------------------------------------

    def selected_track_obj(self) -> Track | None:
        i = self.state.selected_track
        return self.project.tracks[i] if i is not None and i < len(self.project.tracks) else None

    def _track_of(self, clip: Clip) -> Track | None:
        return next((t for t in self.project.tracks if any(c is clip for c in t.clips)), None)

    def delete_selected(self) -> None:
        clip = self.state.selected_clip
        if clip is not None and (track := self._track_of(clip)) is not None:
            self.delete_clip(track, clip)
        elif (track := self.selected_track_obj()) is not None:
            self.delete_track(track)

    def delete_track(self, track: Track) -> None:
        self._checkpoint(f"Delete track {track.name}")
        self.project.tracks.remove(track)
        self.state.selected_track = None
        self.state.selected_clip = None
        self.engine_label.setText(f"Deleted {track.name} — Ctrl+Z to undo")
        self.refresh()
        self._changed()

    def delete_clip(self, track: Track, clip: Clip) -> None:
        self._checkpoint("Delete clip")
        track.clips = [c for c in track.clips if c is not clip]
        self.state.selected_clip = None
        self.refresh()
        self._changed()

    def duplicate_clip(self, clip: Clip | None = None) -> None:
        clip = clip or self.state.selected_clip
        track = self._track_of(clip) if clip is not None else None
        if track is None:
            return
        self._checkpoint("Duplicate clip")
        copy = clip.copy(start=clip.end)
        track.clips.append(copy)
        self.state.selected_clip = copy
        self.refresh()
        self._changed()

    def duplicate_track(self, track: Track) -> None:
        self._checkpoint(f"Duplicate {track.name}")
        self.project.tracks.insert(self.project.tracks.index(track) + 1, track.copy())
        self.refresh()
        self._changed()

    def split_at(self, t: float, track: Track | None = None) -> None:
        """Cut clips at time ``t``: on ``track``, else the selected clip's/track's, else every track."""
        targets = [track] if track else ([self._track_of(self.state.selected_clip)] if self.state.selected_clip
                                         else [self.selected_track_obj()] if self.selected_track_obj()
                                         else self.project.tracks)
        cuts = [(tr, c, parts) for tr in targets if tr for c in tr.clips if (parts := c.split(t))]
        if not cuts:
            self.engine_label.setText("Nothing under the playhead to split")
            return
        self._checkpoint("Split")
        for tr, clip, (left, right) in cuts:
            i = next(i for i, c in enumerate(tr.clips) if c is clip)
            tr.clips[i:i + 1] = [left, right]
        self.state.selected_clip = None
        self.refresh()
        self._changed()

    def rename_track(self, track: Track, name: str) -> None:
        self._checkpoint(f"Rename {track.name}")
        track.name = name
        self.refresh(remix=False)
        self._changed()

    def export_track(self, track: Track) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export track", str(Path.home() / f"{track.name}.wav"),
                                              "WAV (*.wav)")
        if path:
            self.project.export_track(track, Path(path))
            self.engine_label.setText(f"Exported {Path(path).name}")

    def _clip_edit_started(self) -> None:
        self._checkpoint("Move clip")

    def _clip_moved(self) -> None:
        self.refresh()
        self._changed()

    # --- context menus -----------------------------------------------------------------

    def track_menu(self, track: Track, pos) -> None:
        self._show_menu(self._track_items(track, None), pos)

    def lane_menu(self, track: Track | None, clip: Clip | None, pos) -> None:
        t = self.state.t(self.lanes.mapFromGlobal(pos).x())
        items = []
        if clip is not None:
            items += [("Split here", lambda: self.split_at(t, track)),
                      ("Duplicate clip\tCtrl+D", lambda: self.duplicate_clip(clip)),
                      ("Chop to pads", [(label, lambda m=mode: self.chop_to_pads(m, track, clip))
                                        for label, mode in (("At each hit (drums)", "hits"), ("Every beat", "beat"),
                                                            ("Every half beat", "half"), ("Every bar", "bar"))]),
                      ("Add to the pad tray", self._tray_items(track, clip)),
                      ("Delete clip", lambda: self.delete_clip(track, clip)), None]
        if track is not None:
            items += self._track_items(track, t) + [None]
        if self.project.tracks:
            items.append(("Extend everything playing (merged into one take)", self._extend_items(None, t)))
        if self.state.selection:
            items.append((f"Redo {self._sel_text()} of everything playing", lambda: self.redo_bars(None)))
        items += [("＋ Add layer…", lambda: self.add_layer()), ("Play from here", lambda: self._play_from(t))]
        self._show_menu(items, pos)

    def _tray_items(self, track: Track, clip: Clip) -> list:
        """Collect bits of the timeline on the pads, below the 4×4: the clip, the selected bars, or chops."""
        items = [("The whole clip", lambda: self.add_to_tray(track, clip))]
        if self.state.selection:
            items.append((f"Just {self._sel_text()}", lambda: self.add_to_tray(track, clip, bars=True)))
        items += [None] + [(f"Chop it: {label}", lambda m=mode: self.chop_to_pads(m, track, clip,
                                                                                    targets=self.tray_slots(16)))
                           for label, mode in (("at each hit", "hits"), ("every beat", "beat"), ("every bar", "bar"))]
        return items

    def _extend_items(self, track: Track | None, t: float | None = None) -> list:
        """+4/+8/+16 bars always; "to here" when the click was past the end."""
        bar = self.project.bar_seconds
        end = track.end if track else self.project.length
        items = [(f"+{n} bars (to bar {round(end / bar + n) + 1})", lambda n=n: self.extend_to(track, end + n * bar))
                 for n in (4, 8, 16)]
        if t is not None and t > end + bar / 2:
            here = round(t / bar + 0.4999) + 1
            items.insert(0, (f"To here (bar {here})", lambda: self.extend_to(track, t)))
        return items

    def _track_items(self, track: Track, t: float | None) -> list:
        items = [("Rename…", lambda: self.headers.start_rename(track))]
        if track.gen.get("mode") == "song":
            items.append(("Song part", self.song_part_items(track)))
        elif track.gen:
            items.append(("Regenerate (new take)", lambda: self.regenerate(track)))
        items.append((f"Extend '{track.name}'", self._extend_items(track, t)))
        items.append(("Keep only (separate it)", [(label, lambda s=stem: self.isolate_track(track, s)) for label, stem in (
            ("Vocals", "vocals"), ("Drums", "drums"), ("Bass", "bass"),
            ("Everything else (no vocals, drums or bass)", "other"))]))
        if self.state.selection:
            items.append((f"Redo {self._sel_text()} of this track", lambda: self.redo_bars(track)))
        items += [("Duplicate track", lambda: self.duplicate_track(track)),
                  ("Export track as WAV…", lambda: self.export_track(track)),
                  ("Delete track\tDel", lambda: self.delete_track(track))]
        return items

    def _sel_text(self) -> str:
        a, b = (round(s / self.project.bar_seconds) + 1 for s in self.state.selection)
        return f"bars {a}–{b - 1}"

    def _show_menu(self, items: list, pos) -> None:
        self._build_menu(items).exec(pos)

    def _build_menu(self, items: list, title: str = "", parent=None) -> QMenu:
        """``items``: (label, callable) | (label, [sub-items]) | None for a separator."""
        menu = QMenu(title, parent or self)
        for item in items:
            if item is None:
                if menu.actions() and not menu.actions()[-1].isSeparator():
                    menu.addSeparator()
                continue
            label, target = item
            if isinstance(target, list):
                menu.addMenu(self._build_menu(target, label, menu))
            else:
                menu.addAction(label).triggered.connect(lambda _=False, f=target: f())
        return menu

    def _play_from(self, t: float) -> None:
        self.player.seek(t)
        self.player.play()

    # --- importing ---------------------------------------------------------------------

    def import_audio(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Import audio", str(Path.home()),
                                                "Audio (*.wav *.flac *.mp3 *.ogg *.aiff *.aif)")
        self._import([Path(p) for p in paths], 0.0)

    def _import(self, paths: list[Path], start: float) -> None:
        paths = [p for p in paths if p.suffix.lower() in AUDIO_EXTENSIONS]
        if not paths:
            return
        self._checkpoint("Import audio")
        for p in paths:
            self.project.add_file(p, p.stem, start=start)
        self.refresh()
        self._changed()

    def dragEnterEvent(self, ev):
        if any(Path(u.toLocalFile()).suffix.lower() in AUDIO_EXTENSIONS for u in ev.mimeData().urls()):
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        x = self.lanes.mapFrom(self, ev.position().toPoint()).x()
        bar = self.project.bar_seconds
        start = max(0.0, round(self.state.t(x) / bar) * bar) if x > 0 else 0.0
        self._import([Path(u.toLocalFile()) for u in ev.mimeData().urls()], start)
