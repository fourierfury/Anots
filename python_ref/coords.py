"""Mapping between display units (seconds, Hz) and STFT bins.

These are the canonical STFT coordinate formulas. They live here — free of any
GUI or transform import — so both the ``StftTransform`` and the GUI shim
(``python_ref.gui.coords``) can share one source of truth without an import cycle.
"""

from __future__ import annotations

from .params import StftParams


def time_to_frame(t: float, sr: int, params: StftParams) -> int:
    return round(t * sr / params.hop_length)


def frame_to_time(frame: int, sr: int, params: StftParams) -> float:
    return frame * params.hop_length / sr


def hz_to_bin(hz: float, sr: int, params: StftParams) -> int:
    return round(hz * params.n_fft / sr)


def bin_to_hz(bin_index: int, sr: int, params: StftParams) -> float:
    return bin_index * sr / params.n_fft


def n_freq_bins(params: StftParams) -> int:
    return params.n_fft // 2 + 1
