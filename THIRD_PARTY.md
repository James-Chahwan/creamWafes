# Third-party software and models

creamWafes itself is MIT licensed (see `LICENSE`). It doesn't bundle any of the following; they're
installed separately (by `uv`, `tools/setup_engine.sh`, or your system's package manager).

| What | Used for | Licence |
|---|---|---|
| [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) — code and model weights (base, turbo, VAE) | generating sounds, layers, redoing bars | MIT |
| [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (downloaded with ACE-Step) | ACE-Step's text encoder | Apache-2.0 |
| [Demucs](https://github.com/facebookresearch/demucs) (htdemucs model) | keeping only the vocals/drums/bass of a take | MIT |
| [PySide6 / Qt](https://www.qt.io/qt-for-python) | the user interface | LGPL-3.0 |
| NumPy, SciPy | audio processing | BSD |
| python-soundfile / libsndfile | reading and writing audio (incl. MP3, FLAC, OGG, Opus) | BSD / LGPL-2.1 |
| python-sounddevice / PortAudio | audio output | MIT |
| python-soxr | resampling | LGPL-2.1 |
| mido, python-rtmidi | MIDI pad controllers | MIT |
| PyTorch (in the engine's environment) | running the models | BSD-3-Clause |
| [Rubber Band](https://breakfastquay.com/rubberband/) — optional, called as a separate program if installed | pitch without changing speed (Layer panel) | GPL-2.0+ |
| [FFmpeg](https://ffmpeg.org) — optional, called as a separate program if installed | M4A/AAC export | LGPL/GPL |

Generated audio: read ACE-Step's model card and licence for what the models were trained on and for
any terms about using what they generate.
