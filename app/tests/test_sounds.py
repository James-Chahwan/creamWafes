"""End-to-end: single-sound design (describe → generate → pick a candidate), offscreen, fake engine."""

import os
import sys
from pathlib import Path

os.environ["CREAMWAFES_NO_AUDIO"] = "1"

import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialogButtonBox  # noqa: E402
import soundfile as sf  # noqa: E402

import harness  # noqa: E402
from harness import SENT, check, cleanup, deliver, demo_song, drum_loop, finish, open_window  # noqa: E402
from creamwafes import sounds  # noqa: E402
from creamwafes.model import SR  # noqa: E402
from creamwafes.ui import creative  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "creamwafes"))
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

# --- more kinds, loops, genre/mood, lengths, banks (pure logic)
check(all(t in sounds.SOUND_TYPES for _, kinds in sounds.GROUPS for t in kinds)
      and {t for _, kinds in sounds.GROUPS for t in kinds} == set(sounds.SOUND_TYPES),
      f"{len(sounds.SOUND_TYPES)} kinds of sound, every one in the menu's groups")
check([sounds.guess_type(x) for x in ("lofi drum loop", "violin swell", "ride bell", "glockenspiel", "trumpet",
                                       "choir ahhs", "riser", "boom")] ==
      ["drum loop", "strings", "ride", "bells", "brass", "choir", "riser", "impact"], "the new kinds are recognised")
cap = sounds.caption("808", "distorted", "trap", ["dark", "gritty"], "long")
check(cap.startswith("distorted, dark gritty trap, long, sustained") and "808 bass" in cap,
      f"genre and mood go into the caption, after your words ({cap[:60]}…)")
groove = drum_loop(120, 12)
loops = sounds.candidates(groove, "drum loop", 0.5, 2.0, 4, "medium")
check(len(loops) == 4 and all(e - s == 4 * SR for s, e, _ in loops) and all(s % (2 * SR) == 0 for s, _, _ in loops),
      f"loops are whole bars on the beat grid ({[l[2] for l in loops]})")
check(loops[0][2] == "drum loop 1 (2 bars)", "a loop is named for what it is")
check(all(e - s == 8 * SR for s, e, _ in sounds.candidates(groove, "drum loop", 0.5, 2.0, 2, "long")),
      "Long = 4-bar loops")
short = sounds.candidates(groove, "kick", 0.5, 2.0, 8, "short")
check(len(short) >= 4 and max(e - s for s, e, _ in short) <= 0.2 * SR, "Short drum hits are tight (≤ 0.2 s)")
plan16 = sounds.plan_bank(list(range(16)), [p for p, _ in creative.BANK_PARTS])
check(plan16 == {"kit": [0, 1, 2, 3, 4], "808": [5], "perc": [6], "loop": [7], "chords": [8, 9, 10, 11],
                 "melody": [12], "hook": [13, 14], "fx": [15]}, f"a 16-pad bank: drums low, chords on QWER {plan16}")
plan8 = sounds.plan_bank(list(range(8)), [p for p, _ in creative.BANK_PARTS])
check(plan8 == {"kit": [0, 1, 2], "808": [3], "chords": [4, 5], "melody": [6], "hook": [7]},
      f"8 pads get the essentials first {plan8}")
check(sounds.plan_bank([3, 9], ["chords"]) == {"chords": [3, 9]}, "only what you tick")

# the whole flow in the app
mixed = np.concatenate([groove[: int(10 * SR)], eights[: int(10 * SR)]])  # drums, then pitched notes
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

# --- the new dialog: a 4-bar chord loop, lofi + warm, 8 takes
def loop_dialog(d):
    seen["groups"] = [d.kind.itemText(k) for k in range(d.kind.count()) if d.kind.itemData(k) is None]
    d.text.setText("dusty rhodes")
    d.set_kind("chord loop")
    d._kind_touched = True
    d.genre.buttons["lofi"].click()
    d.moods.buttons["warm"].click()
    d.length_buttons["long"].click()
    d.takes_buttons["8"].click()
    seen["length_label"] = d.length_buttons["long"].text()
    d.accept()


harness.fake_engine(groove)
drive(creative.SoundDialog, loop_dialog)
w.design_pad_sound(9)
task, params, _ = SENT[-1]
check(seen.get("groups") == ["— Drums —", "— Bass —", "— Melodic —", "— Vocal —", "— FX —", "— Loops —"]
      and seen.get("length_label") == "4 bars", "kinds come in groups; for loops, length is in bars")
check(params["caption"].startswith("dusty rhodes, warm lofi, chord progression loop")
      and params["duration"] == sounds.take_seconds("chord loop", 8, "long", w.project.bar_seconds) >= 4 * 4 * w.project.bar_seconds,
      f"lofi + warm + 4-bar loop; room for several takes, capped at 45 s ({params['duration']:.0f}s)")
drive(creative.SoundPicker, lambda d: (seen.update(n=len(d.buttons)), d.buttons[0].click(), d._use()))
deliver(w)
pad9 = w.project.pads[9]
check(seen.get("n", 0) >= 2 and pad9.mode == "loop" and abs(pad9.clip.length - 4 * w.project.bar_seconds) < 0.01,
      f"a loop lands as a loop pad, exactly 4 bars long ({seen.get('n')} takes to pick from)")
check(w._sound_prefs()["genre"] == "lofi" and w._sound_prefs()["takes"] == 8, "the dialog remembers genre/takes")

# --- several pads at once: a different kick on each
w.select_pads([0, 1, 2, 3])
drive(creative.SoundDialog, lambda d: (seen.update(many=d.many, ok=d.findChild(QDialogButtonBox).button(
    QDialogButtonBox.Ok).text()), d.text.setText("punchy kick"), d.accept()))
before = w.history.next_undo()
w.generate_sounds([0, 1, 2, 3])
check(seen.get("many") and seen.get("ok") == "✦ Generate 4 sounds" and SENT[-1][2] is not None
      and all(w.pad_panel.grid.buttons[i].generating == 0.0 for i in range(4)),
      "Sounds onto these: one dialog, progress on all 4 pads")
deliver(w)
names = [w.project.pads[i].name for i in range(4)]
check(all(n.startswith("kick") for n in names) and len({w.project.pads[i].clip.offset for i in range(4)}) == 4,
      f"…a different kick on each pad, no picker ({names})")
w.undo()
check(w.history.next_undo() == before, "…as one undo step")

# --- a variation of a pad's own sound
w.select_pad(9)
def vary_dialog(d):
    seen.update(vary=d.vary.isChecked(), kind=d.sound_type(), text=d.vary.text())
    d.how_different.setValue(100)
    d.accept()
drive(creative.SoundDialog, vary_dialog)
w.design_pad_sound(9, vary=True)
task, params, _ = SENT[-1]
src, _sr = sf.read(params["src_audio"], dtype="float32", always_2d=True)
check(seen.get("vary") and seen.get("kind") == "chord loop" and "Start from this pad's sound" in seen.get("text", ""),
      f"✦ Variation starts from the pad's sound and keeps its kind ({seen.get('kind')})")
check(task == "cover" and abs(params["audio_cover_strength"] - 0.3) < 1e-6 and abs(len(src) / SR - 10.5) < 0.01,
      "…the engine remakes ~10 s of it ('cover'), 'a lot' different = strength 0.3")
drive(creative.SoundPicker, lambda d: d.reject())
deliver(w)

# --- a whole bank from one vibe
harness.fake_engine(mixed)
def bank_dialog(d):
    seen["preview"] = d.preview.text()
    d.text.setText("cyberpunk night drive")
    d.genre.buttons["trap"].click()
    d.accept()
drive(creative.BankDialog, bank_dialog)
first = len(SENT)
plan = w.fill_bank(list(range(16)))
jobs = SENT[first:]
check("Z X C V A" in seen.get("preview", "") and len(jobs) == 8 and all(
      "cyberpunk night drive" in p["caption"] and "trap" in p["caption"] for _, p, _ in jobs),
      f"Fill these pads: one generation per part ({len(jobs)}), all in that vibe")
check(all(w.pad_panel.grid.buttons[i].generating is not None for i in range(16)), "…every pad shows it's coming")
for k, (_task, _params, path) in enumerate(jobs, first + 1):
    w._job_done(f"j{k}", {"path": str(path), "seconds": 9.0})
QApplication.processEvents()
chords = [w.project.pads[i] for i in plan["chords"]]
check(all(p.play == "chord" and p.clip is not None for p in chords) and len({p.degree for p in chords}) == 4,
      f"the chord pads play a progression with the generated keys ({[w.pad_label(p)[0] for p in chords]})")
filled = [i for i, p in enumerate(w.project.pads) if p.clip is not None]
check(len(filled) >= 12 and all(w.pad_panel.grid.buttons[i].generating is None for i in range(16)),
      f"the bank fills the pads ({len(filled)}/16 got a sound from this fake take)")
if os.environ.get("CREAMWAFES_SHOTS"):
    QApplication.processEvents()
    w.pads_dock.grab().save(f"{os.environ['CREAMWAFES_SHOTS']}/bank.png")

w.close()
cleanup(song)
finish()
