"""Everything that asks the engine for audio: ideas, layers, extensions, section redos, regenerate.

Every result lands on a *new* track (a take); the tracks it was made from get muted, so a good
take is never overwritten and Ctrl+Z still works.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from PySide6.QtWidgets import QMessageBox

import numpy as np

from ..diag import log
from ..model import SR, STEM_KIND, VOCAL_TYPES, Clip, Track
from .dialogs import IdeaDialog, LayerDialog

SOLO_HINTS = {
    "vocals": "a cappella lead vocal only, dry, no instruments, no music",
    "backing_vocals": "a cappella backing vocal harmonies only, no instruments, no music",
    "drums": "solo drum kit only, drums and percussion, no bass, no melody, no other instruments",
    "percussion": "solo percussion only, no drum kit, no melody, no other instruments",
    "fx": "sound effects and risers only, no drums, no melody",
}


class GenerateMixin:
    """Mixed into MainWindow; needs project, engine, state, jobs, refresh(), _checkpoint()."""

    # --- entry points ------------------------------------------------------------------

    def new_idea(self) -> None:
        dlg = IdeaDialog(self.project.bpm, self.project.key, self)
        if not dlg.exec():
            return
        v = dlg.values()
        params = self._base_params(v["prompt"], v["lyrics"], v["steps"], v["seed"])
        params.update(duration=v["bars"] * self.project.bar_seconds, instrumental=v["instrumental"])
        gen = {"mode": "idea", "kind": "song", "prompt": v["prompt"], "lyrics": v["lyrics"],
               "steps": v["steps"], "bars": v["bars"], "instrumental": v["instrumental"]}
        self._submit("text2music", params, gen, name="idea", mute=[])

    def add_layer(self, kind: str | None = None, prompt: str | None = None) -> None:
        context = [t for t in self.project.tracks if self.project.audible(t)]
        dlg = LayerDialog(self.state.selection, self.project.bar_seconds, kind, prompt,
                          empty=not context, parent=self)
        if not dlg.exec():
            return
        v = dlg.values()
        if not context:
            self._submit_solo(v["kind"], v["prompt"], v["lyrics"], v["steps"], v["seed"], v["bars"])
        else:
            span = self.state.selection if v["only_selection"] else None
            self._submit_layer(v["kind"], v["prompt"], v["lyrics"], v["steps"], v["seed"], span, context,
                               isolate=v.get("isolate"))

    def extend_to(self, track: Track | None, t: float) -> None:
        """Continue ``track`` (or everything audible) from its end up to time ``t``."""
        sources = [track] if track else [x for x in self.project.tracks if self.project.audible(x)]
        if not sources:
            return
        bar = self.project.bar_seconds
        end = max(x.end for x in sources)
        target = math.ceil(t / bar - 1e-6) * bar
        if target <= end + bar / 2:
            QMessageBox.information(self, "Extend", "Right-click further to the right than where the audio ends.")
            return
        single = len(sources) == 1  # one layer: send it dry; the new take gets its fx/volume/pan back
        audio = self.project.render(sources, end, dry=single)
        resume = self._last_loud_bar_end(audio, bar)  # generated songs fade out: replace the ending
        start = max(0.0, min(resume, end) - bar)        # and redo one bar before it so the join is seamless
        # The model winds down at the end of whatever it's asked for, so ask for 2 bars of headroom
        # and trim them off. Pad with silence ourselves: ACE-Step tells the model the song lasts as long
        # as the *unpadded* source, so letting it pad internally makes it write an ending too early.
        gen_end = target + 2 * bar
        wav = self.project.write_wav(self.project.render(sources, gen_end, dry=single), "extend-src")
        base = sources[0] if len(sources) == 1 else None
        # Positive wording only: "no fade out" in a caption makes these models fade out.
        prompt = (base.prompt if base else "") or "continue the song in the same style, full arrangement"
        lyrics = (base.gen.get("lyrics") if base else None) or "[Instrumental]"
        gen = {"mode": "repaint", "kind": base.kind if base else "song", "prompt": prompt, "lyrics": lyrics,
               "steps": 32, "src": str(wav), "span": [start, gen_end], "trim_to": target, "inherit": single}
        name = f"{base.name if base else 'song'} → bar {round(target / bar) + 1}"
        self._submit_repaint(gen, name, sources)

    @staticmethod
    def _last_loud_bar_end(audio, bar: float) -> float:
        """End time of the last bar that's within 12 dB of the song's typical loudness."""
        mono = audio.mean(axis=1)
        n = max(1, int(bar * SR))
        rms = np.array([np.sqrt(np.mean(mono[i:i + n] ** 2)) + 1e-9 for i in range(0, len(mono), n)])
        db = 20 * np.log10(rms)
        loud = np.nonzero(db > np.median(db) - 12)[0]
        return (loud[-1] + 1) * bar if len(loud) else len(mono) / SR

    def redo_bars(self, track: Track | None, prompt: str | None = None) -> None:
        """Regenerate only the selected bars of ``track`` (or of everything audible). Works past the
        track's end and over silent stretches too (filling a gap). ``prompt``: what should go there."""
        if not self.state.selection:
            return
        sources = [track] if track else [x for x in self.project.tracks if self.project.audible(x)]
        if not sources:
            return
        single = len(sources) == 1  # one layer: send it dry; the new take gets its fx/volume/pan back
        length = max(self.project.length, self.state.selection[1])
        wav = self.project.write_wav(self.project.render(sources, length, dry=single), "redo-src")
        base = sources[0] if single else None
        a, b = (round(s / self.project.bar_seconds) + 1 for s in self.state.selection)
        gen = {"mode": "repaint", "kind": base.kind if base else "song",
               "prompt": prompt or self._layer_prompt(base),
               "lyrics": (base.gen.get("lyrics") if base else None) or "[Instrumental]",
               "steps": 32, "src": str(wav), "span": list(self.state.selection), "inherit": single}
        self._submit_repaint(gen, f"{base.name if base else 'song'} (bars {a}–{b - 1} redone)", sources)

    @staticmethod
    def _layer_prompt(track: Track | None) -> str:
        """What to ask for when redoing part of a layer and the user didn't say."""
        if track is None:
            return "same style as the rest of the song"
        if track.prompt and not track.prompt.endswith(" arrangement"):  # song parts say "trap arrangement"
            return track.prompt
        return f"{track.kind}, same sound and style as the rest of this part"

    def regenerate(self, track: Track) -> None:
        """Same request again with a new seed, as a new take; the old take is muted, not lost."""
        g = dict(track.gen)
        mode = g.get("mode", "layer")
        name = track.name.split(" #")[0]
        if mode == "song":  # a part of a song made from pads: a new pattern variation, re-rendered in place
            self.render_song_part(track, vary=True)
        elif mode == "separate":
            self.engine_label.setText("This was separated out of another take — regenerate that one instead")
        elif mode == "solo":
            self._submit_solo(g["kind"], g["prompt"], g["lyrics"], g["steps"], -1, g["bars"], mute=[track])
        elif mode == "idea":
            params = self._base_params(g["prompt"], g["lyrics"], g["steps"], -1)
            params.update(duration=g["bars"] * self.project.bar_seconds, instrumental=g.get("instrumental", True))
            self._submit("text2music", params, g, name=name, mute=[track])
        elif mode == "repaint":
            self._submit_repaint(g, name, [track])
        else:
            names = g.get("context")
            context = [t for t in self.project.tracks if t is not track
                       and (t.name in names if names else self.project.audible(t))]
            if not context:
                QMessageBox.information(self, "Can't regenerate", "The tracks this take was built from are gone.")
                return
            isolate = g.get("isolate")  # a kept-only part replaces just itself; a full take replaces the mix
            self._submit_layer(g["kind"], g["prompt"], g["lyrics"], g["steps"], -1, g.get("span"), context,
                               mute=[track] if isolate else [track, *context], isolate=isolate)

    # --- request builders --------------------------------------------------------------

    def _base_params(self, prompt: str, lyrics: str, steps: int, seed: int) -> dict:
        return {"caption": prompt, "lyrics": lyrics, "bpm": self.project.bpm, "keyscale": self.project.key,
                "timesignature": str(self.project.beats_per_bar), "inference_steps": steps, "seed": seed}

    def _submit_solo(self, kind, prompt, lyrics, steps, seed, bars, mute: list[Track] | None = None) -> None:
        """No context to listen to: make the part alone with text2music, steering it away from a full band."""
        solo = SOLO_HINTS.get(kind, f"solo {kind.replace('_', ' ')} only, isolated, no drums, no other instruments")
        params = self._base_params(f"{prompt}, {solo}" if prompt else solo, lyrics, steps, seed)
        params.update(duration=bars * self.project.bar_seconds, instrumental=kind not in VOCAL_TYPES)
        gen = {"mode": "solo", "kind": kind, "prompt": prompt, "lyrics": lyrics, "steps": steps, "bars": bars}
        self._submit("text2music", params, gen, name=kind, mute=mute or [])

    def _submit_layer(self, kind, prompt, lyrics, steps, seed, span, context: list[Track],
                      mute: list[Track] | None = None, isolate: str | None = None) -> None:
        """Lego: the engine hears ``context`` and returns the whole song with the new part in it.
        ``isolate`` ("vocals"/"drums"/"bass"): separate just that stem out, so the result is the new
        part alone and the context stays unmuted."""
        wav = self.project.write_wav(self.project.render(context, self.project.length), "context")
        params = self._base_params(prompt or kind, lyrics, steps, seed)
        params.update(src_audio=str(wav), duration=self.project.length,
                      instruction=f"Generate the {kind.upper()} track based on the audio context:")
        if span:
            params.update(repainting_start=span[0], repainting_end=span[1])
        if isolate:
            params["isolate"] = isolate
        gen = {"mode": "layer", "kind": kind, "prompt": prompt, "lyrics": lyrics, "steps": steps,
               "span": list(span) if span else None, "context": [t.name for t in context], "isolate": isolate}
        self._submit("lego", params, gen, name=f"+{kind}" + (f" · {prompt[:24]}" if prompt else ""),
                     mute=mute if mute is not None else ([] if isolate else context))

    def isolate_track(self, track: Track, stem: str) -> None:
        """Right-click → Keep only: split a take into stems and keep one, as a new take (this one muted)."""
        from .. import fx
        start, audio = fx.raw(track)  # dry: its sound settings come back on the new take
        if not len(audio):
            return
        wav = self.project.write_wav(audio, "separate-src")
        gen = {"mode": "separate", "kind": STEM_KIND.get(stem, track.kind) if stem != "other" else track.kind,
               "prompt": track.prompt, "stem": stem, "start": start / SR, "inherit": True}
        label = {"other": "everything else"}.get(stem, stem)
        self._submit("separate", {"src": str(wav), "stem": stem, "duration": len(audio) / SR}, gen,
                     name=f"{track.name} · {label}", mute=[track])

    def _submit_repaint(self, gen: dict, name: str, mute: list[Track]) -> None:
        params = self._base_params(gen["prompt"], gen["lyrics"], gen["steps"], -1)
        params.update(src_audio=gen["src"], repainting_start=gen["span"][0], repainting_end=gen["span"][1],
                      duration=gen["span"][1], chunk_mask_mode="explicit")  # "auto" ignores the region → quiet
        self._submit("repaint", params, gen, name=name, mute=mute)

    def _submit(self, task: str, params: dict, gen: dict, name: str, mute: list[Track],
                pads: list[int] | None = None) -> None:
        if not self.engine.is_ready:
            QMessageBox.information(self, "Engine not ready", "The engine is still loading — try again in a moment.")
            return
        gen = {**gen, "bpm": self.project.bpm, "key": self.project.key}
        job_id = self.engine.submit(task, params, self.project.folder / "generated")
        log.info("generate %s %s: %s", job_id, task, {k: v for k, v in params.items() if k != "lyrics"})
        job = {"task": task, "gen": gen, "name": name, "kind": gen["kind"], "prompt": gen.get("prompt", ""),
               "mute": mute, "started": time.time(), "label": name, "progress": 0.0, "row": None,
               "pads": pads}  # pad-bound jobs show progress on those pads, not as a timeline row
        self.jobs[job_id] = job
        self.state.pending.append(job)
        if pads:
            self.pads_dock.show()
            self.pad_panel.grid.set_generating(pads, 0.0)
        self.refresh(remix=False)

    # --- engine callbacks --------------------------------------------------------------

    def _job_progress(self, job_id: str, value: float, desc: str) -> None:
        job = self.jobs.get(job_id)
        if job:
            job["progress"] = max(0.0, min(1.0, value))
            job["label"] = f"{job['name']} — {desc} ({time.time() - job['started']:.0f}s)"
            if job.get("pads"):
                self.pad_panel.grid.set_generating(job["pads"], job["progress"])
            self.lanes.update()

    def _job_done(self, job_id: str, ev: dict) -> None:
        log.info("generated %s in %ss → %s", job_id, ev.get("seconds"), ev.get("path"))
        job = self.jobs.pop(job_id, None)
        if not job:
            return
        self._drop_pending(job)
        mode = job["gen"].get("mode")
        if mode == "sound":
            self.refresh(remix=False)
            self._sound_ready(Path(ev["path"]), job["gen"])
            return
        if mode == "hooks":
            self._hooks_ready(Path(ev["path"]), job["prompt"], job["gen"]["hook_type"], job["gen"]["hook_pads"])
            self.refresh(remix=False)
            return
        if job["gen"].get("to_pad") is not None:
            self._pad_sound_ready(job["gen"]["to_pad"], Path(ev["path"]), job["prompt"], job["kind"])
            self.refresh(remix=False)
            return
        if job["gen"].get("to_pads"):
            kit = Track(name=job["prompt"][:12] or "kit", kind="drums", clips=[Clip(Path(ev["path"]))])
            self.chop_to_pads("kit", track=kit, clip=kit.clips[0], use_selection=False,
                              targets=job["gen"].get("targets"))
            self.refresh(remix=False)
            return
        self._checkpoint(f"Generate {job['name']}")
        trim = job["gen"].get("trim_to")
        clip = Clip(Path(ev["path"]), start=job["gen"].get("start", 0.0),  # a separated take sits where it was
                    length=trim)  # extensions are generated long and trimmed back
        track = Track(name=self._unique_name(job["name"]), kind=job["kind"], clips=[clip],
                      prompt=job["prompt"], gen=job["gen"])
        if job["gen"].get("inherit") and len(job["mute"]) == 1:  # a redo of one layer sounds like that layer
            src = job["mute"][0]
            track.fx, track.volume, track.pan = dict(src.fx), src.volume, src.pan
        self.project.tracks.append(track)
        for heard in job["mute"]:  # the new take already contains these
            if heard in self.project.tracks:
                heard.muted = True
        self.state.selected_track = len(self.project.tracks) - 1
        self.engine_label.setText(f"{job['name']} generated in {ev['seconds']}s")
        self.refresh()
        self._changed()

    def _job_failed(self, job_id: str, message: str) -> None:
        log.error("generation %s failed: %s", job_id, message[-1500:])
        job = self.jobs.pop(job_id, None)
        if job:
            self._drop_pending(job)
            self.refresh(remix=False)
        what = job["name"] if job else "Engine"
        QMessageBox.warning(self, "Generation failed", f"{what} failed:\n\n{message[-2000:]}")

    def cancel_generation(self) -> None:
        if not self.jobs:
            return
        for job in self.jobs.values():
            self._drop_pending(job)
        self.jobs.clear()
        self.state.pending.clear()
        self.engine.restart()
        self.engine_label.setText("Generation cancelled — engine restarting…")
        self.refresh(remix=False)

    def _unique_name(self, name: str) -> str:
        """Takes of the same request get numbered: "+synth", "+synth #2", "+synth #3"…"""
        taken = {t.name for t in self.project.tracks}
        if name not in taken:
            return name
        n = 2
        while f"{name} #{n}" in taken:
            n += 1
        return f"{name} #{n}"

    def _drop_pending(self, job: dict) -> None:
        self.state.pending = [j for j in self.state.pending if j is not job]
        if job.get("pads"):
            self.pad_panel.grid.set_generating(job["pads"], None)
