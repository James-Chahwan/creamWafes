"""End-to-end: single-sound design (describe → generate → pick a candidate), offscreen, fake engine."""

import os
import sys
from pathlib import Path

os.environ["LAYERDAW_NO_AUDIO"] = "1"

import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import harness  # noqa: E402
from harness import SENT, check, cleanup, deliver, demo_song, drum_loop, finish, open_window  # noqa: E402
from layerdaw import sounds  # noqa: E402
from layerdaw.model import SR  # noqa: E402
from layerdaw.ui import creative  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "layerdaw"))
import engine_worker  # noqa: E402

check(engine_worker.pick_model({"base": 1, "turbo": 2}, "lego") == "base" and
      engine_worker.pick_model({"base": 1, "turbo": 2}, "text2music") == "turbo" and
      engine_worker.pick_model({"base": 1}, "text2music") == "base",
      "add-layer runs on base; single sounds on turbo when it's installed")
check(sounds.guess_type("distorted cyberpunk 808") == "808" and sounds.guess_type("glassy rhodes") == "keys"
      and sounds.guess_type("vinyl snare") == "snare", "the kind of sound is guessed from your words")
check(sounds.caption("kick", "punchy techno").startswith("punchy techno, kick drum"), "your words lead the caption")

loop = drum_loop()
kicks = sounds.candidates(loop, "kick", 0.5, 2.0)
check(kicks and all(label.startswith("kick") for _, _, label in kicks), f"asking for a kick gives kicks {[k[2] for k in kicks]}")
hats = sounds.candidates(loop, "hi-hat", 0.5, 2.0)
check(hats and all(h[2].startswith("hat") for h in hats), f"asking for a hat gives only hats {[h[2] for h in hats]}")

t = np.arange(int(0.8 * SR)) / SR
eights = np.zeros(int(10.5 * SR), np.float32)
for k, midi in enumerate((36, 38, 41, 43, 36, 31)):
    f = 440 * 2 ** ((midi - 69) / 12)
    a = int(k * 1.6 * SR)
    eights[a:a + len(t)] = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.5) * 0.8
eights = np.stack([eights, eights], 1)
notes = sounds.candidates(eights, "808", 0.5, 2.0)
check(len(notes) == 4 and len({n[2] for n in notes}) == 4, f"808 candidates are 4 different notes {[n[2] for n in notes]}")

pad_t = np.arange(int(10.5 * SR)) / SR
pad = (0.3 * np.sin(2 * np.pi * 220 * pad_t) * np.minimum(1, pad_t / 2)).astype(np.float32)
pads = sounds.candidates(np.stack([pad, pad], 1), "pad", 0.5, 2.0)
check(len(pads) >= 2 and all(e - s == int(2.0 * SR) for s, e, _ in pads), "pad sounds are steady 2 s stretches")

# the whole flow in the app
harness.fake_engine(eights)
song = demo_song()
app, w = open_window(song)
w.pads_dock.show()


def drive(dialog_cls, action):
    def go():
        d = next(x for x in QApplication.topLevelWidgets() if isinstance(x, dialog_cls) and x.isVisible())
        action(d)
    QTimer.singleShot(150, go)


drive(creative.SoundDialog, lambda d: (d.text.setText("distorted cyberpunk 808"), d.accept()))
w.design_pad_sound(4)
task, params, _ = SENT[-1]
check(task == "text2music" and params["caption"].startswith("distorted cyberpunk 808, 808 bass")
      and params["duration"] >= 10 and w.pad_panel.grid.buttons[4].generating == 0.0,
      "describing a sound asks for ~10 s of just that sound, progress on the pad")
seen = {}
drive(creative.SoundPicker, lambda d: (seen.update(n=len(d.buttons)), d.buttons[1].click(), d._use()))
deliver(w)
pad4 = w.project.pads[4]
check(seen.get("n") == 4, "the picker offers 4 candidates to audition")
check(pad4.clip is not None and pad4.mode == "hold" and pad4.kind == "bass" and "808" in pad4.name,
      f"the chosen candidate lands on the pad, ready to hold ({pad4.name}, {pad4.mode})")
picked_offset = pad4.clip.offset

n = len(SENT)
drive(creative.SoundPicker, lambda d: d.done(creative.SoundPicker.MORE))
w.design_pad_sound(5, "distorted cyberpunk 808", "808")
deliver(w)
check(len(SENT) == n + 2, "✦ More like this generates another take")
drive(creative.SoundPicker, lambda d: d.reject())
deliver(w)
check(w.project.pads[4].clip.offset == picked_offset, "cancelling the picker leaves pads alone")

w.close()
cleanup(song)
finish()
