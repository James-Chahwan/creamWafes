"""Mixed into MainWindow: the Layer panel (shape the selected track's sound) and the background
work that applies track fx without freezing the window."""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import QDockWidget

from .. import fx
from ..diag import log
from .layer_panel import LABELS, SLIDERS, LayerPanel


class _Notifier(QObject):
    done = Signal(int, str)   # job number, error text ("" if fine)


class LayerMixin:
    def _build_layer_panel(self) -> None:
        self.layer_panel = LayerPanel()
        self.layer_dock = QDockWidget("Layer", self)
        self.layer_dock.setObjectName("layer")
        self.layer_dock.setWidget(self.layer_panel)
        self.layer_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable)
        self.setCorner(Qt.BottomRightCorner, Qt.BottomDockWidgetArea)  # the pads keep the full width
        self.addDockWidget(Qt.RightDockWidgetArea, self.layer_dock)
        self.resizeDocks([self.layer_dock], [330], Qt.Horizontal)
        self.layer_panel.changed.connect(self.set_layer_fx)
        self.layer_panel.preset.connect(self.apply_layer_preset)
        self.layer_panel.bars_changed.connect(self._layer_bars)
        self.layer_panel.generate.connect(self.generate_layer_bars)
        self.layer_panel.find_gap.connect(self.find_layer_gap)
        self._fx_timer = QTimer(self, singleShot=True, interval=150, timeout=self._fx_render)
        self._fx_job = 0
        self._fx_failed: set = set()
        self._fx_notifier = _Notifier()
        self._fx_notifier.done.connect(self._fx_finished)

    def layer_track(self):
        return self.selected_track_obj()

    def refresh_layer(self) -> None:
        track = self.layer_track()
        if track is None:
            self.layer_panel.show_track(None)
            return
        auto = fx.sidechain_source(self.project, track, name="")
        others = [t.name for t in self.project.tracks if t is not track]
        self.layer_panel.show_track(track, others, auto.name if auto else None, fx.pitch_available())
        self.layer_panel.set_engine_ready(self.engine.is_ready)
        if self.state.selection:
            bar = self.project.bar_seconds
            a, b = (round(t / bar) for t in self.state.selection)
            self.layer_panel.set_bars(a + 1, max(a + 1, b))

    # --- generating part of the layer -------------------------------------------------------------
    def _layer_bars(self, first: int, last: int) -> None:
        """The bar boxes and the ruler selection are the same thing."""
        bar = self.project.bar_seconds
        self.state.selection = ((first - 1) * bar, last * bar)
        self._selection_changed()

    def generate_layer_bars(self, first: int, last: int, prompt: str = "") -> None:
        track = self.layer_track()
        if track is None:
            return
        self._layer_bars(first, last)
        self.redo_bars(track, prompt or None)

    def find_layer_gap(self) -> bool:
        """Select the longest run of silent bars in the layer (between its first and last sound)."""
        import numpy as np
        from ..model import SR
        track = self.layer_track()
        if track is None:
            return False
        start, audio = fx.raw(track)
        bar = self.project.bar_seconds
        n = int(round(bar * SR))
        first_bar = int(start // n)
        offset = first_bar * n - start  # line the analysis up with the song's bars
        padded = np.concatenate([np.zeros((max(0, -offset), 2), np.float32), audio[max(0, offset):]])
        bars = len(padded) // n
        if bars == 0:
            self.engine_label.setText(f"{track.name} is too short to have a gap")
            return False
        peaks = np.abs(padded[: bars * n]).reshape(bars, n * 2).max(axis=1)
        loud = peaks > max(1e-3, peaks.max() * 10 ** (-40 / 20))
        sounding = np.flatnonzero(loud)
        best = None
        k = sounding[0] if len(sounding) else bars
        while k < (sounding[-1] if len(sounding) else 0):
            if not loud[k]:
                j = k
                while j < bars and not loud[j]:
                    j += 1
                if best is None or j - k > best[1] - best[0]:
                    best = (k, j)
                k = j
            else:
                k += 1
        if best is None:
            self.engine_label.setText(f"No empty bars in {track.name} — drag on the ruler to pick some to redo")
            return False
        a, b = first_bar + best[0] + 1, first_bar + best[1]
        self._layer_bars(a, b)
        self.engine_label.setText(f"Bars {a}–{b} of {track.name} are empty — ✦ Generate to fill them")
        return True

    def show_layer(self, track) -> None:
        """The fx button on a track: select it and bring up its Layer panel."""
        self.state.selected_track = self.project.tracks.index(track)
        self.layer_dock.show()
        self.layer_dock.raise_()
        self.refresh(remix=False)

    def set_layer_fx(self, key: str, value) -> None:
        track = self.layer_track()
        if track is None:
            return
        new = {k: v for k, v in {**track.fx, key: value}.items() if k in fx.DEFAULTS and v != fx.DEFAULTS[k]}
        if new == track.fx:
            return
        drag = key in SLIDERS  # a slider drag is one undo step; toggles and picks are one each
        self._checkpoint(f"{LABELS.get(key, key)} on {track.name}", coalesce=("fx", id(track), key) if drag else None)
        track.fx = new  # always a new dict: undo snapshots hold the old one
        self._fx_after_change(track)

    def apply_layer_preset(self, name: str) -> None:
        track = self.layer_track()
        if track is None:
            return
        keep = {k: v for k, v in track.fx.items() if k == "duck_to"}
        self._checkpoint(f"{name} on {track.name}")
        track.fx = {**fx.PRESETS[name], **keep}
        self._fx_after_change(track)
        self.refresh_layer()

    def _fx_after_change(self, track) -> None:
        self.layer_panel.set_summary(track)
        header = self.headers.headers.get(id(track))
        if header is not None:
            header.update_fx()
        self._changed()
        self._fx_timer.start()

    # --- applying fx in the background ------------------------------------------------------------
    def _fx_pending(self) -> list:
        return [t for t in self.project.tracks if self.project.audible(t) and fx.is_active(t.fx)
                and not fx.ready(self.project, t) and fx._key(self.project, t) not in self._fx_failed]

    def _fx_check(self) -> None:
        """After a mix: if some track's fx aren't worked out yet, do them (it's playing dry meanwhile)."""
        if self._fx_pending():
            self._fx_timer.start()

    def _fx_render(self) -> None:
        pending = self._fx_pending()
        if not pending:
            self._remix()
            return
        self._fx_job += 1
        job = self._fx_job
        names = ", ".join(t.name for t in pending[:3]) + ("…" if len(pending) > 3 else "")
        self.engine_label.setText(f"Shaping {names}…")
        threading.Thread(target=self._fx_work, args=(job, pending), daemon=True).start()

    def _fx_work(self, job: int, tracks: list) -> None:
        error = ""
        for track in tracks:
            try:
                fx.output(self.project, track)
            except Exception as exc:  # noqa: BLE001 - one bad setting mustn't kill the mix
                log.exception("fx on %s failed", track.name)
                self._fx_failed.add(fx._key(self.project, track))
                error = f"Couldn't apply the sound settings on {track.name}: {exc}"
        self._fx_notifier.done.emit(job, error)

    def _fx_finished(self, job: int, error: str) -> None:
        if job != self._fx_job:
            return  # settings moved on while this was working; the newer job will remix
        self._remix()
        if error:
            self.engine_label.setText(error)
        elif self.engine_label.text().startswith("Shaping "):
            self.engine_label.setText("Ready")
