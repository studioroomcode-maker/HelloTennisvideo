"""Racket / bounce impact times from the soundtrack.

An impact is a broadband click. Its onset is found as the peak of positive
spectral flux (log magnitude, 1-6 kHz) on a 2 ms hop, refined with a parabola.
Sound travels at ~343 m/s, so an onset heard at the camera is shifted by
distance/343 s (about 80 ms for a serve from the far baseline here).
"""
from __future__ import annotations

import subprocess

import numpy as np

from core.media import exe

SR = 16000
N_FFT = 256
HOP = 32           # 2 ms
BAND = (1000, 6000)
SPEED_OF_SOUND = 343.0


def read_audio(src: str, t0: float, t1: float) -> np.ndarray:
    cmd = [exe("ffmpeg"), "-nostdin", "-loglevel", "error", "-ss", f"{max(0.0, t0):.3f}", "-i", src,
           "-t", f"{t1 - t0:.3f}", "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, np.float32)


def onset_envelope(x: np.ndarray) -> np.ndarray:
    """Positive log-spectral flux per hop (frame k is centred at k*HOP/SR s)."""
    if len(x) < N_FFT:
        return np.zeros(0)
    pad = np.pad(x, (N_FFT // 2, N_FFT // 2))
    n = 1 + (len(pad) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(pad[idx] * np.hanning(N_FFT), axis=1))
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    logm = np.log1p(1000 * spec[:, band])
    flux = np.maximum(0.0, np.diff(logm, axis=0)).sum(axis=1)
    return np.r_[0.0, flux]


def strongest_onset(env: np.ndarray, t0: float, lo: float, hi: float):
    """Time and prominence (peak / median) of the largest onset in [lo, hi] (absolute s)."""
    if len(env) == 0:
        return None
    times = t0 + np.arange(len(env)) * HOP / SR
    sel = np.where((times >= lo) & (times <= hi))[0]
    if len(sel) < 3:
        return None
    k = sel[np.argmax(env[sel])]
    peak = env[k]
    base = float(np.median(env)) + 1e-6
    # parabolic refinement of the peak position
    if 0 < k < len(env) - 1:
        a, b, c = env[k - 1], env[k], env[k + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
    else:
        off = 0.0
    return float(times[k] + off * HOP / SR), float(peak / base)


def hit_candidates(src: str, windows: list[tuple[float, float]], min_ratio: float = 3.0,
                   gap: float = 0.2) -> list[dict]:
    """Impact-like sounds inside the given (start, end) windows: [{"t", "strength"}], time order.

    strength = onset peak / the window's median flux. Neighbouring courts, footsteps and bounces
    make these too: on the test video every hand-labelled hit had a candidate at >= 3, with about
    three candidates per real hit, so they are a list to check, not hits.
    """
    out = []
    for t0, t1 in windows:
        env = onset_envelope(read_audio(src, t0, t1))
        if len(env) == 0:
            continue
        med = float(np.median(env)) + 1e-9
        picked = []
        for i in np.argsort(env)[::-1]:
            r = env[i] / med
            if r < min_ratio:
                break
            t = t0 + i * HOP / SR
            if all(abs(t - u) > gap for u, _ in picked):
                picked.append((t, r))
        out += [{"t": round(float(t), 3), "strength": round(float(r), 1)} for t, r in picked]
    out.sort(key=lambda c: c["t"])
    return out
