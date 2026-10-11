"""End-to-end: vocals (and other parts) come back on their own, not as a whole new song
(offscreen, fake engine, silent)."""

import os

os.environ["LAYERDAW_NO_AUDIO"] = "1"

from pathlib import Path  # noqa: E402

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from harness import SENT, activate, check, cleanup, deliver, demo_song, fake_engine, finish, open_window  # noqa: E402

fake_engine()
song = demo_song()  # drums + bass
app, w = open_window(song)
drums, bass = w.project.tracks
seen = {}


def answer(kind, prompt="", keep=None):
    def go():
        d = next(x for x in QApplication.topLevelWidgets() if isinstance(x, QDialog) and x.isVisible())
        d.kind.setCurrentText(kind)
        d.prompt.setText(prompt)
        seen.update(visible=d.isolate.isVisibleTo(d), checked=d.isolate.isChecked(), text=d.isolate.text())
        for other in ("keyboard", "drums", "bass", kind):  # what the option does for other parts
            d.kind.setCurrentText(other)
            seen[other] = (d.isolate.isVisibleTo(d), d.isolate.isChecked())
        if keep is not None:
            d.isolate.setChecked(keep)
        d.accept()
    QTimer.singleShot(150, go)


# --- +vocals: just the vocals, the song stays as it was
answer("vocals", "soulful female vocals")
w.add_layer()
activate(w)
task, params, _ = SENT[-1]
check(seen.get("visible") and seen.get("checked") and "Keep only the new vocals" in seen.get("text", ""),
      f"Add layer → vocals offers (and ticks) keeping only the new vocals ({seen.get('text')})")
check(seen.get("keyboard") == (False, False) and seen.get("drums") == (True, False) and seen.get("bass") == (True, False),
      f"…drums/bass offer it unticked, other parts don't ({seen.get('keyboard')}, {seen.get('drums')})")
check(task == "lego" and params.get("isolate") == "vocals", "the engine is asked to separate the vocals out")
deliver(w)
vox = w.project.tracks[-1]
check(vox.name.startswith("+vocals") and vox.kind == "vocals" and vox.gen.get("isolate") == "vocals",
      f"the new track is the vocals ({vox.name})")
check(not drums.muted and not bass.muted, "your other tracks stay unmuted (the take is only the new part)")

w.regenerate(vox)
check(SENT[-1][1].get("isolate") == "vocals", "⟳ on it asks for just the vocals again")
deliver(w)
check(vox.muted and not drums.muted and not bass.muted and w.project.tracks[-1].name.startswith("+vocals"),
      "…and only that take gets muted")

# --- a full-mix layer (keys) still works the old way
answer("keyboard", "rhodes")
w.add_layer()
activate(w)
check("isolate" not in SENT[-1][1], "keys come back as a full take (no stem for keys)")
deliver(w)
check(drums.muted and bass.muted, "…which replaces what it heard (those get muted)")
for t in w.project.tracks:
    t.muted = False
w.refresh()

# --- an existing take with the whole song in it: right-click → Keep only ▸ Vocals
old = w.project.tracks[-1]
old.fx, old.volume, old.pan = {"reverb": 0.3}, 0.8, 0.2
menu = w._build_menu(w._track_items(old, None))
keep = next(a.menu() for a in menu.actions() if a.menu() is not None and a.text().startswith("Keep only"))
labels = [a.text() for a in keep.actions()]
check(labels[:3] == ["Vocals", "Drums", "Bass"] and labels[3].startswith("Everything else"),
      f"right-click a track → Keep only ▸ {labels}")
n = len(w.project.tracks)
next(a for a in keep.actions() if a.text() == "Vocals").trigger()
task, params, _ = SENT[-1]
check(task == "separate" and params["stem"] == "vocals" and Path(params["src"]).exists(),
      "it sends that take's audio to be separated (no new generation)")
deliver(w)
stem = w.project.tracks[-1]
check(len(w.project.tracks) == n + 1 and stem.name == f"{old.name} · vocals" and stem.kind == "vocals"
      and old.muted, f"the vocals land on a new take; the original is muted, not lost ({stem.name})")
check(stem.fx == {"reverb": 0.3} and (stem.volume, stem.pan) == (0.8, 0.2) and stem.clips[0].start == old.clips[0].start,
      "…in the same place, with the same sound settings, volume and pan")
w.regenerate(stem)
check("separated out of another take" in w.engine_label.text(), "⟳ on a separated stem explains where to regenerate")

w.close()
cleanup(song)
finish()
