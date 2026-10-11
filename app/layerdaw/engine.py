"""Talks to engine_worker.py running in the ACE-Step venv via a QProcess."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

ENGINE_DIR = Path(os.environ.get("LAYERDAW_ENGINE",
                                 Path(__file__).resolve().parents[2] / "engine" / "ACE-Step-1.5"))
WORKER = Path(__file__).with_name("engine_worker.py")
LOG_FILE = Path.home() / "Music" / "LayerDAW" / "engine.log"


class Engine(QObject):
    status = Signal(str)                    # human-readable engine state
    ready = Signal(dict)
    progress = Signal(str, float, str)      # job id, 0..1, description
    finished = Signal(str, dict)            # job id, done event
    failed = Signal(str, str)               # job id ("" for engine-level), message

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_ready = False
        self.turbo = False
        self._restarting = False
        self.busy: set[str] = set()
        self._buffer = b""
        self._log: list[str] = []
        self._log_file = None
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(str(ENGINE_DIR))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("LAYERDAW_ENGINE", str(ENGINE_DIR))
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation on 6 GB
        self._proc.setProcessEnvironment(env)
        self._proc.readyReadStandardOutput.connect(self._read_stdout)
        self._proc.readyReadStandardError.connect(self._read_stderr)
        self._proc.finished.connect(self._on_exit)

    def start(self) -> None:
        if os.environ.get("LAYERDAW_NO_ENGINE"):  # tests / inspecting: never load a model onto the GPU
            self.status.emit("Engine disabled (LAYERDAW_NO_ENGINE)")
            return
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        if self._log_file:
            self._log_file.close()
        self._log_file = LOG_FILE.open("w", buffering=1)
        uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
        self.status.emit("Engine: loading model (first start compiles, takes a few minutes)…")
        self._proc.start(uv, ["run", "--no-sync", "--project", str(ENGINE_DIR), "python", str(WORKER)])

    def stop(self) -> None:
        if self._proc.state() != QProcess.NotRunning:
            self._proc.closeWriteChannel()
            if not self._proc.waitForFinished(3000):
                self._proc.kill()

    def restart(self) -> None:
        """Kill the worker (dropping anything queued) and load it again."""
        self._restarting = True
        self.busy.clear()
        self.is_ready = False
        if self._proc.state() != QProcess.NotRunning:
            self._proc.kill()
            self._proc.waitForFinished(5000)
        self._buffer = b""
        self.start()

    def submit(self, task: str, params: dict, out_dir: Path) -> str:
        job_id = uuid.uuid4().hex[:8]
        out_dir.mkdir(parents=True, exist_ok=True)
        job = {"id": job_id, "task": task, "params": params, "out_dir": str(out_dir)}
        self._proc.write((json.dumps(job) + "\n").encode())
        self.busy.add(job_id)
        return job_id

    def recent_log(self, lines: int = 25) -> str:
        return "\n".join(self._log[-lines:])

    def _read_stdout(self) -> None:
        self._buffer += bytes(self._proc.readAllStandardOutput())
        *lines, self._buffer = self._buffer.split(b"\n")
        for raw in lines:
            line = raw.decode(errors="replace")
            if line.startswith("@@ "):
                self._handle(json.loads(line[3:]))
            elif line.strip():
                self._log.append(line)

    def _read_stderr(self) -> None:
        text = bytes(self._proc.readAllStandardError()).decode(errors="replace")
        if self._log_file:
            self._log_file.write(text)
        self._log.extend(l for l in text.splitlines() if l.strip())
        del self._log[:-500]

    def _handle(self, ev: dict) -> None:
        kind, job_id = ev["event"], ev.get("id") or ""
        if kind == "ready":
            self.is_ready = True
            self.status.emit(f"Engine ready: {ev['model']} on {ev['vram_gb']} GB GPU (loaded in {ev['seconds']}s)")
            self.ready.emit(ev)
        elif kind == "models":
            self.turbo = bool(ev.get("turbo"))
            self.status.emit("Engine ready: turbo for single sounds (fast, high quality), base for add-layer"
                             if self.turbo else f"Engine ready (base only: {ev.get('message', 'turbo not installed')})")
        elif kind == "progress":
            self.progress.emit(job_id, ev["value"], ev["desc"])
        elif kind == "done":
            self.busy.discard(job_id)
            self.finished.emit(job_id, ev)
        elif kind in ("error", "fatal"):
            self.busy.discard(job_id)
            self.failed.emit(job_id, ev["message"])

    def _on_exit(self, code: int, _status) -> None:
        if self._restarting:  # deliberate kill from restart(): not a crash
            self._restarting = False
            return
        was_ready, self.is_ready = self.is_ready, False
        for job_id in list(self.busy):
            self.failed.emit(job_id, "engine stopped")
        self.busy.clear()
        if code != 0:
            self.status.emit(f"Engine crashed (exit {code})")
            self.failed.emit("", f"Engine exited with code {code}.\n\n{self.recent_log()}")
        elif was_ready:
            self.status.emit("Engine stopped")
