"""Main window: menus, transport, timeline layout, project handling, autosave."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from PySide6.QtCore import QEvent, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox, QDockWidget, QFileDialog, QGridLayout,
                               QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
                               QScrollArea, QSizePolicy, QSpinBox, QToolBar, QWidget)

from ..audio import Player
from ..diag import log
from ..engine import Engine
from ..history import History
from ..model import Project
from .editing import EditMixin
from .generate import GenerateMixin
from .layer import LayerMixin
from ..midi import MidiInput
from .meter import Meter
from .creative import CreativeMixin, SongMixin, SoundMixin
from .pad_grid import KEY_TO_PAD, note_to_pad
from .pad_panel import PadPanel
from .pads import PadsMixin

FLAVOUR_KEYS = {Qt.Key_G: "7th", Qt.Key_H: "sus4", Qt.Key_B: "sus2", Qt.Key_N: "add9"}
from .timeline import Lanes, Ruler, TimelineState
from .tracks import TrackHeaders
from ..paths import DATA_DIR

PROJECTS_DIR = DATA_DIR
KEYS = [f"{n} {m}" for m in ("major", "minor") for n in
        ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")]
SHORTCUTS = """\
Space            play / stop (anywhere except text boxes)
Home             back to the start
L                loop the selected bars
Esc              clear the selection
Click ruler      move the playhead · drag ruler or empty lane = select bars
Drag a clip      move it (snaps to beats; hold Alt for free)
Right-click      menus: extend to here, redo bars, split, duplicate, rename, export…
Ctrl+Z / Ctrl+Shift+Z   undo / redo
Del              delete the selected clip (or track)
Ctrl+D           duplicate the selected clip after itself
Ctrl+E           split at the playhead
Ctrl+L           add layer · Ctrl+Shift+X export song
P                show/hide the pads · keys 1234/QWER/ASDF/ZXCV play them (Shift = softer)
Pads             click = play & edit it · Ctrl+click add/remove · Shift+click range
                 any pad plays as a Sample, a Chord or a Note of the song's key
                 hold G 7th · H sus4 · B sus2 · N add9 while hitting chord pads
                 select several → ✦ Make a song from these
Ctrl+wheel       zoom · Shift+wheel scroll sideways · Ctrl+0 fit song
Double-click     track name to rename · volume/pan slider to reset
"""


class MainWindow(EditMixin, GenerateMixin, PadsMixin, CreativeMixin, SoundMixin, SongMixin, LayerMixin, QMainWindow):
    def __init__(self, project: Project | None = None):
        super().__init__()
        self.project = project or self._fresh_project()
        self.state = TimelineState()
        self.history = History()
        self._last_coalesce = (None, 0.0)
        self.player = Player()
        self.engine = Engine(self)
        self.jobs: dict[str, dict] = {}
        self._pad_hits: list = []
        self._held_flavours: list[str] = []

        self._build_pads()
        self._build_layer_panel()
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_timeline()
        self._build_statusbar()
        self._wire_engine()
        self._wire_midi()
        self.setAcceptDrops(True)
        QApplication.instance().installEventFilter(self)

        self.autosave_timer = QTimer(self, singleShot=True, interval=800, timeout=self._autosave)
        self.remix_timer = QTimer(self, singleShot=True, interval=60, timeout=self._remix)
        self.timer = QTimer(self, interval=33, timeout=self._tick)
        self.timer.start()
        self.refresh()
        self.engine.start()
        self.resize(1400, 780)
        QTimer.singleShot(0, self.lanes.setFocus)

    @staticmethod
    def _fresh_project(bpm: int = 90, key: str = "A minor") -> Project:
        return Project(PROJECTS_DIR / time.strftime("song-%Y%m%d-%H%M%S"), bpm=bpm, key=key)

    # --- actions & menus ---------------------------------------------------------------

    def _act(self, text: str, slot, key: str | None = None, tip: str | None = None, checkable=False) -> QAction:
        a = QAction(text, self, checkable=checkable)
        a.triggered.connect(lambda _=False: slot())
        if key:
            a.setShortcut(QKeySequence(key))
        a.setToolTip(f"{tip or text.replace('&', '')} ({key})" if key else (tip or text))
        return a

    def _build_actions(self) -> None:
        A = self._act
        self.a_new = A("&New song", self.new_project, "Ctrl+Shift+N", "Start a new empty song")
        self.a_open = A("&Open…", self.open_project, "Ctrl+O")
        self.a_save = A("&Save", self.save_project, "Ctrl+S", "Save now (songs also save automatically)")
        self.a_rename = A("Rename song…", self.rename_song)
        self.a_import = A("&Import audio…", self.import_audio, "Ctrl+I")
        self.a_chop_file = A("Chop an audio file onto the pads…", lambda: self.chop_file_to_pads(),
                             tip="Slice a drum loop or stem from disk across the pads (the selected ones, or all 16)")
        self.a_export_song = A("⤓ Export song…", self.export_song, "Ctrl+Shift+X",
                               "The whole mix as one file: MP3, WAV, FLAC, OGG, Opus, AIFF or M4A")
        self.a_export = A("&Export stems…", self.export_stems, "Ctrl+Shift+E",
                          "Every track + the mix as WAVs for Ardour")
        self.a_undo = A("↶ Undo", self.undo, "Ctrl+Z")
        self.a_redo = A("↷ Redo", self.redo, "Ctrl+Shift+Z")
        self.a_redo.setShortcuts([QKeySequence("Ctrl+Shift+Z"), QKeySequence("Ctrl+Y")])
        self.a_delete = A("Delete", self.delete_selected, "Del", "Delete the selected clip, or the selected track")
        self.a_delete.setShortcuts([QKeySequence("Del"), QKeySequence("Backspace")])
        self.a_dup = A("Duplicate clip", self.duplicate_clip, "Ctrl+D")
        self.a_split = A("Split at playhead", lambda: self.split_at(self.player.position), "Ctrl+E")
        self.a_clear = A("Clear selection", self.clear_selection, "Esc")
        self.a_start = A("⏮", self.to_start, "Home", "Back to the start")
        self.a_play = A("▶ Play", self.toggle_play, None, "Play / stop (Space)")
        self.a_loop = A("⟲ Loop", self._loop_changed, "L", "Loop the selected bars", checkable=True)
        self.a_zoom_in = A("Zoom in", lambda: self.zoom(1.5), "Ctrl+=")
        self.a_zoom_out = A("Zoom out", lambda: self.zoom(1 / 1.5), "Ctrl+-")
        self.a_fit = A("Zoom to fit", self.zoom_fit, "Ctrl+0")
        self.a_layer = A("＋ Add layer", self.add_layer, "Ctrl+L",
                         "Generate one instrument or vocal (on its own if the song is empty)")
        self.a_cancel = A("Cancel generation", self.cancel_generation, None, "Stop everything that's generating")
        self.a_kit = A("Generate a kit onto the selected pads…", lambda: self.generate_kit(), None,
                       "Make a drum kit and put one sound on each selected pad")
        self.a_song = A("✦ Make a song from the selected pads…", self.open_song_maker, None,
                        "Arrange exactly the selected pads' sounds into a song")
        self.a_hooks = A("Generate hooks onto the selected pads…", lambda: self.generate_hooks(), None,
                         "Melodic phrases or one-shot ear candy onto the selected pads")
        self.a_prog = A("Write a chord progression…", self.open_progression_writer, None,
                        "A progression for a mood, on the chord pads or the timeline")
        self.a_chord_sound = A("Generate a sound for the selected pad…", lambda: self.design_pad_sound(self.primary_pad),
                               None, "Describe one sound (or loop); genre, mood, length; pick from 4 or 8 takes")
        self.a_sounds = A("Generate sounds onto the selected pads…", lambda: self.generate_sounds(self._targets()),
                          None, "One kind of sound or loop, a different take on each selected pad")
        self.a_bank = A("✦ Fill the selected pads (a whole bank)…", lambda: self.fill_bank(self._targets()), None,
                        "One vibe → drum kit, 808, chords, melody, hook, fx across the selected pads")
        self.a_pads = self.pads_dock.toggleViewAction()
        self.a_pads.setText("▦ Pads")
        self.a_pads.setShortcut(QKeySequence("P"))
        self.a_pads.setToolTip("Show/hide the pad sampler (P)")
        for a in (self.a_undo, self.a_redo, self.a_delete, self.a_dup, self.a_split, self.a_clear,
                  self.a_start, self.a_loop, self.a_zoom_in, self.a_zoom_out, self.a_fit, self.a_pads):
            self.addAction(a)  # shortcuts work even when the menus aren't open

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file = bar.addMenu("&File")
        file.addActions([self.a_new, self.a_open])
        self.recent_menu = file.addMenu("Open recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        file.addActions([self.a_save, self.a_rename])
        file.addSeparator()
        file.addActions([self.a_import, self.a_chop_file])
        file.addSeparator()
        file.addActions([self.a_export_song, self.a_export])
        file.addSeparator()
        file.addAction(self._act("Quit", self.close, "Ctrl+Q"))
        edit = bar.addMenu("&Edit")
        edit.addActions([self.a_undo, self.a_redo])
        edit.addSeparator()
        edit.addActions([self.a_delete, self.a_dup, self.a_split, self.a_clear])
        play = bar.addMenu("&Transport")
        play_item = play.addAction("Play / Stop\tSpace")
        play_item.triggered.connect(lambda _=False: self.toggle_play())
        play.addActions([self.a_start, self.a_loop])
        view = bar.addMenu("&View")
        view.addActions([self.a_zoom_in, self.a_zoom_out, self.a_fit])
        view.addSeparator()
        view.addAction(self.a_pads)
        layer = self.layer_dock.toggleViewAction()
        layer.setText("Layer panel (shape the selected track's sound)")
        view.addAction(layer)
        gen = bar.addMenu("&Generate")
        gen.addActions([self.a_layer])
        self.extend_menu = gen.addMenu("Extend everything playing")
        self.extend_menu.aboutToShow.connect(self._fill_extend_menu)
        gen.addAction(self.a_song)
        gen.addSeparator()
        gen.addActions([self.a_bank, self.a_chord_sound, self.a_sounds, self.a_kit, self.a_hooks])
        gen.addSeparator()
        gen.addAction(self.a_prog)
        gen.addSeparator()
        gen.addAction(self.a_cancel)
        help_menu = bar.addMenu("&Help")
        help_menu.addAction(self._act("Keyboard && mouse…", self.show_shortcuts, "F1"))

    def _build_toolbar(self) -> None:
        bar = QToolBar("Transport")
        bar.setMovable(False)
        self.addToolBar(bar)
        bar.addActions([self.a_undo, self.a_redo])
        bar.addSeparator()
        bar.addActions([self.a_start, self.a_play, self.a_loop])
        self.time_label = QLabel()
        self.time_label.setStyleSheet("font-family: monospace; font-size: 15px; padding: 0 10px;")
        self.time_label.setToolTip("bar.beat · minutes:seconds")
        bar.addWidget(self.time_label)
        bar.addSeparator()
        bar.addWidget(QLabel(" BPM "))
        self.bpm = QSpinBox(minimum=40, maximum=240, value=self.project.bpm, keyboardTracking=False)
        self.bpm.setFocusPolicy(Qt.ClickFocus)
        self.bpm.valueChanged.connect(self._tempo_changed)
        self.bpm.editingFinished.connect(self.lanes_focus)
        bar.addWidget(self.bpm)
        bar.addWidget(QLabel("  Key "))
        self.key = QComboBox(editable=True)
        self.key.addItems(KEYS)
        self.key.setCurrentText(self.project.key)
        self.key.setFocusPolicy(Qt.ClickFocus)
        self.key.currentTextChanged.connect(self._key_changed)
        self.key.activated.connect(lambda _i: self.lanes_focus())
        bar.addWidget(self.key)
        bar.addSeparator()
        bar.addActions([self.a_layer])
        bar.addSeparator()
        bar.addAction(self.a_pads)
        bar.addSeparator()
        bar.addAction(self.a_export_song)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        self.meter = Meter()
        bar.addWidget(self.meter)

    def _build_timeline(self) -> None:
        self.ruler = Ruler(self.project, self.state)
        self.lanes = Lanes(self.project, self.state)
        self.headers = TrackHeaders()

        self.lanes_scroll = QScrollArea(widgetResizable=False)
        self.lanes_scroll.setWidget(self.lanes)
        self.headers_scroll = QScrollArea(widgetResizable=True)
        self.headers_scroll.setWidget(self.headers)
        self.headers_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.headers_scroll.setFixedWidth(self.headers.width() + 2)
        self.lanes_scroll.verticalScrollBar().valueChanged.connect(self.headers_scroll.verticalScrollBar().setValue)
        self.lanes_scroll.horizontalScrollBar().valueChanged.connect(self._hscroll)

        for w in (self.ruler, self.lanes):
            w.seek.connect(self._seek)
            w.selection_changed.connect(self._selection_changed)
        self.lanes.track_selected.connect(lambda _row: self.refresh(remix=False))
        self.lanes.zoomed.connect(lambda: self.refresh(remix=False))
        self.lanes.hscroll.connect(lambda d: self.lanes_scroll.horizontalScrollBar().setValue(
            self.lanes_scroll.horizontalScrollBar().value() + d))
        self.lanes.clip_edit_started.connect(self._clip_edit_started)
        self.lanes.clip_moved.connect(self._clip_moved)
        self.lanes.context_menu.connect(self.lane_menu)
        self.headers.about_to_change.connect(self._checkpoint)
        self.headers.changed.connect(self._mix_changed)
        self.headers.renamed.connect(self.rename_track)
        self.headers.regenerate.connect(self.regenerate)
        self.headers.delete.connect(self.delete_track)
        self.headers.menu.connect(self.track_menu)
        self.headers.selected.connect(self._header_selected)
        self.headers.shape.connect(self.show_layer)
        self.headers.cancel.connect(self.cancel_generation)

        grid = QGridLayout()
        grid.setSpacing(0)
        grid.setContentsMargins(0, 0, 0, 0)
        self.hint = QLabel()
        self.hint.setStyleSheet("color: #7a828e; font-size: 11px;")
        grid.addWidget(self.hint, 0, 0)
        grid.addWidget(self.ruler, 0, 1)
        grid.addWidget(self.headers_scroll, 1, 0)
        grid.addWidget(self.lanes_scroll, 1, 1)
        central = QWidget()
        central.setLayout(grid)
        self.setCentralWidget(central)

    def _build_pads(self) -> None:
        self.pad_panel = PadPanel()
        self.pads_dock = QDockWidget("Pads", self)
        self.pads_dock.setObjectName("pads")
        self.pads_dock.setWidget(self.pad_panel)
        self.pads_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.pads_dock)
        grid, ins = self.pad_panel.grid, self.pad_panel.inspector
        grid.hit.connect(self.hit_pad)
        grid.released.connect(self.release_pad)
        grid.selection_changed.connect(self.refresh_inspector)
        grid.menu.connect(self.pad_menu)
        grid.file_dropped.connect(self.pad_file_dropped)
        self.pad_panel.record_toggled.connect(self._record_toggled)
        self.pad_panel.select_all.connect(self.select_all_pads)
        ins.renamed.connect(self.rename_pad)
        ins.category_set.connect(self.set_pad_category)
        ins.generate.connect(lambda: self.design_pad_sound(self.primary_pad))
        ins.variation.connect(lambda: self.design_pad_sound(self.primary_pad, vary=True))
        ins.replace.connect(self.replace_pad)
        ins.audition.connect(lambda: self._audition(self.primary_pad))
        ins.play_set.connect(self.set_pad_play)
        ins.mode_set.connect(self.set_pad_mode)
        ins.degree_set.connect(self.set_pad_degree)
        ins.flavour_set.connect(self.set_pad_flavour)
        ins.strum_set.connect(self.set_pad_strum)
        ins.pitch_step.connect(self.nudge_pad_pitch)
        ins.gain_step.connect(self.nudge_pad_gain)
        ins.make_song.connect(self.open_song_maker)
        ins.bank.connect(lambda: self.fill_bank(sorted(self.selected_pads)))
        ins.sounds.connect(lambda: self.generate_sounds(sorted(self.selected_pads)))
        ins.kit.connect(lambda: self.generate_kit(sorted(self.selected_pads)))
        ins.hooks.connect(lambda: self.generate_hooks())
        ins.progression.connect(self.open_progression_writer)
        ins.chop.connect(lambda mode: self.chop_file_to_pads(targets=sorted(self.selected_pads)) if mode == "file"
                         else self.chop_to_pads(mode, targets=sorted(self.selected_pads)))
        ins.play_all.connect(self.set_selected_play)
        ins.mode_all.connect(self.set_selected_mode)
        ins.clear_all.connect(self.clear_selected)
        if not any(pad.clip or pad.play != "sample" for pad in self.project.pads):
            self.pads_dock.hide()

    def _wire_midi(self) -> None:
        self.midi = MidiInput(self)
        self.midi.note_on.connect(lambda note, vel: self.hit_pad(note_to_pad(note), vel))
        self.midi.note_off.connect(lambda note: self.release_pad(note_to_pad(note)))
        self.midi.status.connect(self.pad_panel.midi_label.setText)
        self.pad_panel.midi_label.setText(self.midi.describe())

    def _build_statusbar(self) -> None:
        self.engine_label = QLabel()
        self.tempo_warning = QLabel()
        self.tempo_warning.setStyleSheet("color: #e0b050;")
        self.statusBar().addWidget(self.engine_label, 1)
        self.statusBar().addPermanentWidget(self.tempo_warning)

    def _wire_engine(self) -> None:
        self.engine.status.connect(self.engine_label.setText)
        self.engine.status.connect(lambda m: log.info("engine: %s", m))
        self.engine.ready.connect(lambda _ev: self._update_actions())
        self.engine.progress.connect(self._job_progress)
        self.engine.finished.connect(self._job_done)
        self.engine.failed.connect(self._job_failed)

    # --- refresh -----------------------------------------------------------------------

    def refresh(self, remix: bool = True) -> None:
        self.setWindowTitle(f"creamWafes — {self.project.name}")
        if remix:
            self.player.set_mix(self.project.render(ready_only=True))
            self._fx_check()
        self.headers.rebuild(self.project, self.state.selected_track, self.state.pending)
        self.lanes.update_size(self.lanes_scroll.viewport().width(), self.lanes_scroll.viewport().height())
        self.lanes.update()
        self.ruler.update()
        self._selection_changed()
        self._update_actions()
        self._update_tempo_warning()
        self.refresh_pads()

    def _remix(self) -> None:
        self.player.set_mix(self.project.render(ready_only=True))
        self._fx_check()
        self.lanes.update()

    def _mix_changed(self) -> None:
        """Mute/solo/volume/pan: re-render audio and repaint, but keep the header widgets alive."""
        self.remix_timer.start()
        self._changed()

    def _update_actions(self) -> None:
        ready = self.engine.is_ready
        self.a_layer.setEnabled(ready)
        self.a_cancel.setEnabled(bool(self.jobs))
        self.a_rename.setEnabled(not self.jobs)
        undo, redo = self.history.next_undo(), self.history.next_redo()
        self.a_undo.setEnabled(undo is not None)
        self.a_undo.setToolTip(f"Undo {undo} (Ctrl+Z)" if undo else "Nothing to undo")
        self.a_redo.setEnabled(redo is not None)
        self.a_redo.setToolTip(f"Redo {redo} (Ctrl+Shift+Z)" if redo else "Nothing to redo")

    def _update_tempo_warning(self) -> None:
        off = sorted({t.gen["bpm"] for t in self.project.tracks
                      if t.gen.get("bpm") and t.gen["bpm"] != self.project.bpm})
        self.tempo_warning.setText(
            f"⚠ some tracks were made at {', '.join(map(str, off))} BPM — the grid won't line up with them"
            if off else "")

    def lanes_focus(self) -> None:
        self.lanes.setFocus()

    # --- transport ---------------------------------------------------------------------

    def _tick(self) -> None:
        old_x = int(self.state.x(self.state.playhead))
        self.state.playhead = self.player.position
        bar = self.project.bar_seconds
        b = self.state.playhead / bar
        secs = self.state.playhead
        self.time_label.setText(f"{int(b) + 1:>3}.{int((b % 1) * self.project.beats_per_bar) + 1}"
                                f"  {int(secs // 60)}:{secs % 60:04.1f}")
        self.a_play.setText("■ Stop" if self.player.playing else "▶ Play")
        self.meter.push(self.player.read_peaks())
        if (msg := self.player.check_alive(time.monotonic())) is not None:
            self.engine_label.setText(msg)
            log.warning(msg)
        if self.pad_panel.rec.isChecked() and not self.player.playing:
            self.pad_panel.rec.setChecked(False)  # playback stopped: finish the pad take
        if self.player.playing:
            x, sb = self.state.x(self.state.playhead), self.lanes_scroll.horizontalScrollBar()
            view = self.lanes_scroll.viewport().width()
            if not sb.value() <= x < sb.value() + view - 40:
                sb.setValue(int(x - 40))
        new_x = int(self.state.x(self.state.playhead))
        if new_x != old_x:  # repaint just the two playhead columns, not every waveform
            for x in (old_x, new_x):
                self.lanes.update(x - 2, 0, 5, self.lanes.height())
            self.ruler.update()

    def toggle_play(self) -> None:
        log.info("%s at %.2fs", "stop" if self.player.playing else "play", self.player.position)
        if self.player.playing:
            self.player.stop()
            return
        if self.player.position >= self.project.length - 0.05 and not self.a_loop.isChecked():
            self.player.seek(0)  # finished last time: start over instead of playing silence
        self.player.play()

    def to_start(self) -> None:
        start = self.state.selection[0] if self.state.selection and self.a_loop.isChecked() else 0.0
        self.player.seek(start)
        self.lanes_scroll.horizontalScrollBar().setValue(int(self.state.x(start)))

    def clear_selection(self) -> None:
        self.state.selection = None
        self.state.selected_clip = None
        self._selection_changed()

    def _seek(self, seconds: float) -> None:
        self.player.seek(seconds)
        if not self.a_loop.isChecked():
            self.state.selection = None
        self._selection_changed()

    def _hscroll(self, value: int) -> None:
        self.ruler.offset = value
        self.ruler.update()

    def _header_selected(self, track) -> None:
        if track in self.project.tracks and self.project.tracks.index(track) != self.state.selected_track:
            self.state.selected_track = self.project.tracks.index(track)
            self.state.selected_clip = None
            self.refresh(remix=False)

    def _selection_changed(self) -> None:
        self.refresh_layer()
        if self.state.selection:
            self.hint.setText(f" {self._sel_text()} selected · Esc clears")
        else:
            self.hint.setText(" drag ruler = select bars")
        self._loop_changed()
        self.lanes.update()
        self.ruler.update()

    def _loop_changed(self) -> None:
        sel = self.state.selection if self.a_loop.isChecked() else None
        self.player.set_loop(*(sel or (None, None)))

    def zoom(self, factor: float) -> None:
        self.state.pps = min(600.0, max(4.0, self.state.pps * factor))
        self.refresh(remix=False)

    def zoom_fit(self) -> None:
        length = max(self.project.length, 8 * self.project.bar_seconds)
        self.state.pps = max(4.0, (self.lanes_scroll.viewport().width() - 30) / length)
        self.lanes_scroll.horizontalScrollBar().setValue(0)
        self.refresh(remix=False)

    def _tempo_changed(self, bpm: int) -> None:
        self._checkpoint("Change BPM", coalesce="bpm")
        self.project.bpm = bpm
        self.state.selection = None
        self.refresh(remix=False)
        self._changed()

    def _key_changed(self, key: str) -> None:
        if key and key != self.project.key:
            self._checkpoint("Change key", coalesce="key")
            self.project.key = key
            self._changed()

    def eventFilter(self, obj, ev):
        """Space plays/stops, and pad keys play pads, from anywhere in the window except while typing."""
        if ev.type() == QEvent.KeyPress and self.isActiveWindow() and not self._typing():
            mods = ev.modifiers() & ~Qt.KeypadModifier
            if ev.key() == Qt.Key_Space and not mods:
                self.toggle_play()
                return True
            if (ev.key() in FLAVOUR_KEYS and not mods and self.pads_dock.isVisible()
                    and any(p.play == "chord" for p in self.project.pads)
                    and not self._in_field()):
                if not ev.isAutoRepeat():
                    self._held_flavours.append(FLAVOUR_KEYS[ev.key()])
                    self.refresh_pads()
                return True
            if ev.key() in KEY_TO_PAD and not ev.isAutoRepeat() and self.pads_dock.isVisible() and self._in_field():
                log.info("pad key %s went to %s (typing), not the pads", ev.text(),
                         type(QApplication.focusWidget()).__name__)
            if (ev.key() in KEY_TO_PAD and mods in (Qt.NoModifier, Qt.ShiftModifier)
                    and self.pads_dock.isVisible() and not self._in_field()):
                if not ev.isAutoRepeat():
                    self.hit_pad(KEY_TO_PAD[ev.key()], 0.6 if mods & Qt.ShiftModifier else 1.0)
                return True
        if ev.type() == QEvent.KeyRelease and ev.key() in FLAVOUR_KEYS and not ev.isAutoRepeat():
            flavour = FLAVOUR_KEYS[ev.key()]
            if flavour in self._held_flavours:
                self._held_flavours.remove(flavour)
                self.refresh_pads()
        if (ev.type() == QEvent.KeyRelease and ev.key() in KEY_TO_PAD and not ev.isAutoRepeat()
                and self.pads_dock.isVisible()):
            self.release_pad(KEY_TO_PAD[ev.key()])  # even if focus moved: never leave a note stuck
        return super().eventFilter(obj, ev)

    @staticmethod
    def _in_field() -> bool:
        """Any text/number box (BPM, key…): letters and digits belong to it, not to the pads."""
        focus = QApplication.focusWidget()
        return isinstance(focus, (QLineEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox))

    def _typing(self) -> bool:
        focus = QApplication.focusWidget()
        return focus is not None and (
            focus.window() is not self or isinstance(focus, QPlainTextEdit)
            or (isinstance(focus, QLineEdit) and not isinstance(focus.parent(), (QAbstractSpinBox, QComboBox))))

    # --- project -----------------------------------------------------------------------

    def _changed(self) -> None:
        self.autosave_timer.start()

    def _autosave(self) -> None:
        if self.project.tracks or (self.project.folder / "project.json").exists():
            self.project.save()

    def save_project(self) -> None:
        self.project.save()
        self.engine_label.setText(f"Saved to {self.project.folder}")

    def new_project(self) -> None:
        self._autosave()
        self._set_project(self._fresh_project(self.project.bpm, self.project.key))

    def open_project(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open song", str(PROJECTS_DIR))
        if folder:
            self._open(Path(folder))

    def _open(self, folder: Path) -> None:
        if not (folder / "project.json").exists():
            QMessageBox.warning(self, "Open song", f"{folder} isn't a creamWafes song (no project.json).")
            return
        self._autosave()
        self._set_project(Project.load(folder))

    def _fill_extend_menu(self) -> None:
        self.extend_menu.clear()
        if not self.project.tracks:
            self.extend_menu.addAction("(nothing to extend yet)").setEnabled(False)
            return
        for label, slot in self._extend_items(None):
            self.extend_menu.addAction(label).triggered.connect(lambda _=False, f=slot: f())

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        for folder in Project.recent(PROJECTS_DIR):
            self.recent_menu.addAction(folder.name).triggered.connect(lambda _=False, f=folder: self._open(f))
        if self.recent_menu.isEmpty():
            self.recent_menu.addAction("(no songs yet)").setEnabled(False)

    def rename_song(self) -> None:
        name, ok = QInputDialog.getText(self, "Rename song", "Song name:", text=self.project.name)
        if not ok or not name.strip() or name.strip() == self.project.name:
            return
        self._autosave()
        try:
            self.project.rename(name.strip())
        except OSError as exc:
            QMessageBox.warning(self, "Rename song", str(exc))
            return
        self.project.save()
        self.refresh(remix=False)

    def _set_project(self, project: Project) -> None:
        self.player.stop()
        self.player.seek(0)
        self.project = project
        self.history.clear()
        for w in (self.ruler, self.lanes):
            w.project = project
        for widget, value in ((self.bpm, project.bpm), (self.key, project.key)):
            widget.blockSignals(True)
            widget.setValue(value) if widget is self.bpm else widget.setCurrentText(value)
            widget.blockSignals(False)
        self.state.selection = self.state.selected_track = self.state.selected_clip = None
        self.__dict__.pop("_instrument", None)  # fresh pitched-note cache for the new song
        self.refresh()

    def export_song(self, options: dict | None = None) -> Path | None:
        """File → Export song…: render the mix (with every track's sound settings) to one file."""
        from .. import export
        from ..model import SR
        from .export_dialog import ExportDialog
        if not self.project.tracks:
            self.engine_label.setText("Nothing to export yet — add some tracks first")
            return None
        if options is None:
            dlg = ExportDialog(self, getattr(self, "_export_prefs", None))
            accepted = dlg.exec()
            options, prefs = dlg.options(), dlg.prefs()
            dlg.deleteLater()
            if not accepted:
                return None
            self._export_prefs = prefs
        kind, a, b = options["range"]
        tail = 6.0 if options["tails"] else 0.0
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.engine_label.setText("Exporting…")
        QApplication.processEvents()
        try:
            mix = self.project.render(None, b + tail)[int(round(a * SR)):]
            keep = int(round((b - a) * SR))
            mix = export.trim_tail(mix, keep) if tail else mix[:keep]
            if kind == "sel" and len(mix) > 480:  # a selection ends where it ends: no click
                mix[-480:] *= np.linspace(1, 0, 480, dtype=np.float32)[:, None]
            mastered, report = export.master(mix, options["level"])
            path = options["path"]
            n = 2
            while path.exists():  # never overwrite an earlier export
                path = path.with_name(f"{options['path'].stem} ({n}){path.suffix}")
                n += 1
            path = export.write(path, mastered, options["format"], options["quality"], options["rate"],
                                title=options.get("title", ""), artist=options.get("artist", ""))
        except Exception as exc:  # noqa: BLE001 - tell the user, keep the app
            QApplication.restoreOverrideCursor()
            log.exception("export failed")
            QMessageBox.warning(self, "Export failed", f"Couldn't export:\n\n{exc}")
            return None
        QApplication.restoreOverrideCursor()
        seconds = len(mastered) / SR
        info = (f"{path.name}  ·  {int(seconds // 60)}:{int(seconds % 60):02d}  ·  {path.stat().st_size / 1e6:.1f} MB"
                f"  ·  {report['lufs']:.1f} LUFS, peak {report['peak_db']:.1f} dB")
        self.engine_label.setText(f"Exported {info}")
        log.info("exported %s (%s, level %s, gain %+.1f dB)", path, options["format"], options["level"],
                 report["gain_db"])
        if getattr(self, "_quiet_export", False):
            return path
        box = QMessageBox(QMessageBox.Information, "Exported", f"{info}\n\n{path.parent}", parent=self)
        play = box.addButton("▶ Play it", QMessageBox.ActionRole)
        folder = box.addButton("Open folder", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Ok)
        box.exec()
        if box.clickedButton() is play:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        elif box.clickedButton() is folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))
        return path

    def export_stems(self) -> None:
        if not self.project.tracks:
            return
        dest = self.project.folder / "export" / time.strftime("%Y%m%d-%H%M%S")
        files = self.project.export(dest)
        box = QMessageBox(QMessageBox.Information, "Exported",
                          f"Wrote {len(files)} files to\n{dest}\n\nAll stems start at 0:00 — drop them into "
                          "Ardour at the session start.", parent=self)
        open_btn = box.addButton("Open folder", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Ok)
        box.exec()
        if box.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(dest)))

    def show_shortcuts(self) -> None:
        box = QMessageBox(QMessageBox.NoIcon, "Keyboard & mouse", SHORTCUTS, parent=self)
        box.setStyleSheet("QLabel { font-family: monospace; }")
        box.exec()

    def closeEvent(self, ev):
        self._autosave()
        QApplication.instance().removeEventFilter(self)
        self.engine.stop()
        self.midi.close()
        self.player.close()
        super().closeEvent(ev)
