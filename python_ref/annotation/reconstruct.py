"""Reconstruct audio from a selection mask (design §8.3, §10.5).

Each layer reconstructs independently from the original complex STFT, so overlapping
layers share the overlap content without cross-contamination.
"""

from __future__ import annotations

import numpy as np

from ..dsp import istft, stft
from ..params import STFT, StftParams
from .model import ExtractionMode


def effective_mask(mask: np.ndarray, mode: ExtractionMode | str = ExtractionMode.POSITIVE) -> np.ndarray:
    """Negative extraction inverts the mask: keep everything except the selection."""
    if mode in (ExtractionMode.NEGATIVE, ExtractionMode.NEGATIVE.value):
        return 1.0 - mask
    return mask


def reconstruct(
    stft_complex: np.ndarray,
    mask: np.ndarray,
    mode: ExtractionMode | str = ExtractionMode.POSITIVE,
    params: StftParams = STFT,
    length: int | None = None,
) -> np.ndarray:
    """Apply ``mask`` to the complex STFT and invert to a time-domain clip."""
    return istft(stft_complex * effective_mask(mask, mode), params=params, length=length)


def reconstruct_from_signal(
    y: np.ndarray,
    mask: np.ndarray,
    mode: ExtractionMode | str = ExtractionMode.POSITIVE,
    params: StftParams = STFT,
) -> np.ndarray:
    """Convenience: STFT the signal, reconstruct, and return a clip of the same length."""
    return reconstruct(stft(y, params), mask, mode=mode, params=params, length=len(y))
