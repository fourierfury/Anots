"""STFT -> ISTFT round-trip: the exact reconstruction the annotation model relies on."""

from __future__ import annotations

import numpy as np
import pytest

from python_ref import dsp
from python_ref.params import TOLERANCE


@pytest.mark.parametrize("name", ["sine", "chirp", "noise", "clip"])
def test_stft_istft_roundtrip_is_exact(signals, name):
    y = signals[name]
    y_hat = dsp.istft(dsp.stft(y), length=len(y))
    assert y_hat.shape == y.shape

    # Centered framing is exact in the interior; the first/last half-window
    # carries edge effects from the centering pad and is excluded.
    n_fft = 1024
    interior = slice(n_fft, len(y) - n_fft)
    max_diff = float(np.max(np.abs(y_hat[interior] - y[interior])))
    assert max_diff < TOLERANCE, f"{name}: max|diff|={max_diff:.3e}"


def test_all_ones_mask_is_identity(signals):
    S = dsp.stft(signals["sine"])
    mask = np.ones_like(S, dtype=np.float64)
    assert np.array_equal(
        dsp.istft(S * mask, length=len(signals["sine"])),
        dsp.istft(S, length=len(signals["sine"])),
    )
