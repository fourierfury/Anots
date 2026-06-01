"""STFT -> ISTFT round-trip reconstruction.

The STFT path is an *exact* reconstruction path (magnitude + phase preserved).
A full-signal round-trip should return the original waveform to within
floating-point precision. This is the property the whole mask-and-reconstruct
annotation model depends on: an all-ones mask must be a no-op.
"""

from __future__ import annotations

import numpy as np
import pytest

from python_ref import dsp
from python_ref.params import TOLERANCE


@pytest.mark.parametrize("name", ["sine", "chirp", "noise", "clip"])
def test_stft_istft_roundtrip_is_exact(signals, name):
    """istft(stft(y)) == y within TOLERANCE over the valid (non-edge) region."""
    y = signals[name]
    S = dsp.stft(y)
    y_hat = dsp.istft(S, length=len(y))

    assert y_hat.shape == y.shape

    # Centered STFT reconstruction is exact in the interior; the first/last
    # half-window can carry edge effects from the centering pad. Compare the
    # interior, which is what any extracted clip uses.
    n_fft = 1024
    interior = slice(n_fft, len(y) - n_fft)
    max_diff = float(np.max(np.abs(y_hat[interior] - y[interior])))
    assert max_diff < TOLERANCE, f"{name}: max|diff|={max_diff:.3e}"


def test_all_ones_mask_is_identity(signals):
    """Applying an all-ones mask to the complex STFT changes nothing."""
    y = signals["sine"]
    S = dsp.stft(y)
    mask = np.ones_like(S, dtype=np.float64)
    masked = dsp.istft(S * mask, length=len(y))
    unmasked = dsp.istft(S, length=len(y))
    assert np.array_equal(masked, unmasked)
