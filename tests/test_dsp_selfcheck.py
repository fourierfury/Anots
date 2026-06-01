"""Analytic self-checks on the DSP reference.

These assert known mathematical properties of the transforms — independent of any
other implementation. They catch gross parameter mistakes (wrong window, wrong FFT
size, NaN on silence) before the native core even exists.
"""

from __future__ import annotations

import numpy as np

from python_ref import dsp
from python_ref.params import STFT
from tests.conftest import SR


def _stft_freqs() -> np.ndarray:
    return np.fft.rfftfreq(STFT.n_fft, d=1.0 / SR)


def test_sine_energy_in_single_bin(signals):
    """A 1 kHz tone should put almost all STFT energy in the nearest bin."""
    S = np.abs(dsp.stft(signals["sine"]))
    # Average magnitude per frequency bin across time.
    per_bin = S.mean(axis=1)
    peak_bin = int(np.argmax(per_bin))
    freqs = _stft_freqs()

    # Peak bin corresponds to ~1 kHz (within half the bin spacing).
    bin_hz = SR / STFT.n_fft
    assert abs(freqs[peak_bin] - 1000.0) <= bin_hz

    # The peak is sharp: the 1 kHz bin towers over the typical (median) bin.
    # Hann leakage spreads some energy to neighbors, so we check dominance over
    # the noise floor rather than a fixed fraction of total magnitude.
    assert per_bin[peak_bin] > 100 * np.median(per_bin)


def test_impulse_magnitude_is_broadband(signals):
    """A single-sample impulse has an (ideally) flat magnitude spectrum.

    With Hann windowing and overlap-add framing it won't be perfectly flat, but no
    frequency region should be empty — energy is spread across the whole band.
    """
    S = np.abs(dsp.stft(signals["impulse"]))
    # Look at the frame with the most energy (the one containing the impulse).
    frame = S[:, int(np.argmax(S.sum(axis=0)))]
    assert np.all(frame > 0.0)
    # Spread: even the quietest bin is within a few orders of magnitude of the peak.
    assert frame.min() > frame.max() * 1e-4


def test_silence_produces_no_nan_or_inf(signals):
    """Every transform must stay finite on a zero signal (no log-of-zero blowups)."""
    y = signals["silence"]
    S = dsp.stft(y)
    assert np.all(np.isfinite(np.abs(S)))

    mel = dsp.melspectrogram(y, SR)
    assert np.all(np.isfinite(mel))

    m = dsp.mfcc(y, SR)
    assert np.all(np.isfinite(m))

    db = dsp.amplitude_to_db(np.abs(S))
    assert np.all(np.isfinite(db))


def test_shapes_are_consistent(signals):
    """STFT/mel/mfcc share a frame count; bin counts match the locked params."""
    y = signals["clip"]
    S = dsp.stft(y)
    mel = dsp.melspectrogram(y, SR)
    m = dsp.mfcc(y, SR)

    n_frames = S.shape[1]
    assert S.shape[0] == STFT.n_fft // 2 + 1
    assert mel.shape == (128, n_frames)
    assert m.shape == (40, n_frames)
