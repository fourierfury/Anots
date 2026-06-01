"""Generate the fixed reference test signals.

Six deterministic signals with analytically predictable transforms (a pure tone
lands in one STFT bin, an impulse is broadband, silence must stay finite).
Regenerate with ``python tests/reference_signals/generate.py``; the ``.wav`` files
are git-ignored, this script is the source of truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44_100
OUT_DIR = Path(__file__).resolve().parent
_NOISE_SEED = 1729


def _t(duration_s: float) -> np.ndarray:
    return np.linspace(0.0, duration_s, int(SR * duration_s), endpoint=False)


def sine_1khz() -> np.ndarray:
    return 0.5 * np.sin(2 * np.pi * 1000.0 * _t(2.0))


def chirp_20hz_20khz() -> np.ndarray:
    t = _t(5.0)
    f0, f1, dur = 20.0, 20_000.0, 5.0
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / dur * t**2)
    return 0.5 * np.sin(phase)


def white_noise() -> np.ndarray:
    rng = np.random.default_rng(_NOISE_SEED)
    return (0.2 * rng.standard_normal(int(SR * 5.0))).astype(np.float64)


def impulse() -> np.ndarray:
    x = np.zeros(SR, dtype=np.float64)
    x[SR // 2] = 1.0
    return x


def silence() -> np.ndarray:
    return np.zeros(int(SR * 5.0), dtype=np.float64)


def clip_test() -> np.ndarray:
    x = 1.5 * np.sin(2 * np.pi * 440.0 * _t(2.0))
    return np.clip(x, -1.0, 1.0)


SIGNALS = {
    "sine_1khz_44100hz.wav": sine_1khz,
    "chirp_20hz_20khz_5s.wav": chirp_20hz_20khz,
    "white_noise_44100hz_5s.wav": white_noise,
    "impulse_44100hz.wav": impulse,
    "silence_44100hz_5s.wav": silence,
    "clip_test_44100hz.wav": clip_test,
}


def generate(out_dir: Path = OUT_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fn in SIGNALS.items():
        path = out_dir / name
        sf.write(path, fn().astype(np.float32), SR, subtype="FLOAT")
        written.append(path)
    return written


if __name__ == "__main__":
    for p in generate():
        print(f"wrote {p.name}")
