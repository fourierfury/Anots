"""STFT transform — the exact-reconstruction reference domain (design §11).

Mask is applied to the complex STFT (magnitude + phase preserved) and inverted via
overlap-add ISTFT, so the round-trip is exact within ``params.TOLERANCE``.
"""

from __future__ import annotations

import numpy as np

from ..annotation.model import ExtractionMode
from ..annotation.reconstruct import reconstruct as _reconstruct
from ..coords import bin_to_hz, frame_to_time, hz_to_bin, time_to_frame
from ..dsp import amplitude_to_db, stft as _stft
from ..params import STFT, StftParams
from .base import Transform


class StftTransform(Transform):
    name = "stft"
    accuracy_class = "exact"
    y_label = "Frequency (Hz)"

    def __init__(self, params: StftParams = STFT):
        self.params = params

    def forward(self, y: np.ndarray, sr: int) -> np.ndarray:
        return _stft(y, self.params)

    def display_db(self, coeffs: np.ndarray) -> np.ndarray:
        return amplitude_to_db(np.abs(coeffs))

    def reconstruct(
        self,
        y: np.ndarray,
        coeffs: np.ndarray,
        mask: np.ndarray,
        mode: ExtractionMode | str = ExtractionMode.POSITIVE,
        sr: int | None = None,
        length: int | None = None,
    ) -> np.ndarray:
        # STFT inverts its own complex coefficients; the original signal is unused.
        return _reconstruct(coeffs, mask, mode, self.params, length=length)

    def time_to_col(self, t: float, sr: int) -> int:
        return time_to_frame(t, sr, self.params)

    def col_to_time(self, col: int, sr: int) -> float:
        return frame_to_time(col, sr, self.params)

    def value_to_row(self, hz: float, sr: int) -> int:
        return hz_to_bin(hz, sr, self.params)

    def row_to_value(self, row: int, sr: int) -> float:
        return bin_to_hz(row, sr, self.params)

    def reconstruction(self) -> dict:
        return {
            "method": "istft",
            "fft_size": self.params.n_fft,
            "hop_length": self.params.hop_length,
            "window_type": self.params.window,
        }

    def provenance(self) -> dict:
        return {}  # the STFT view's only provenance is its reconstruction method
