"""Shim: the app's last-loud-bar logic without importing Qt."""
import numpy as np

SR = 48000


def last_loud_bar_end(audio, bar):
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    n = max(1, int(bar * SR))
    rms = np.array([np.sqrt(np.mean(mono[i:i + n] ** 2)) + 1e-9 for i in range(0, len(mono), n)])
    db = 20 * np.log10(rms)
    loud = np.nonzero(db > np.median(db) - 12)[0]
    return (loud[-1] + 1) * bar if len(loud) else len(mono) / SR
