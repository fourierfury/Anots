"""Analytic self-checks: known properties of the transforms, no other impl needed."""

from __future__ import annotations

import numpy as np

from python_ref import dsp
from python_ref.params import STFT
from tests.conftest import SR


def _stft_freqs() -> np.ndarray:
    return np.fft.rfftfreq(STFT.n_fft, d=1.0 / SR)


def test_sine_energy_in_single_bin(signals):
    per_bin = np.abs(dsp.stft(signals["sine"])).mean(axis=1)
    peak_bin = int(np.argmax(per_bin))

    bin_hz = SR / STFT.n_fft
    assert abs(_stft_freqs()[peak_bin] - 1000.0) <= bin_hz
    # Sharp peak: dominates the noise floor despite Hann leakage to neighbors.
    assert per_bin[peak_bin] > 100 * np.median(per_bin)


def test_impulse_magnitude_is_broadband(signals):
    S = np.abs(dsp.stft(signals["impulse"]))
    frame = S[:, int(np.argmax(S.sum(axis=0)))]
    assert np.all(frame > 0.0)
    assert frame.min() > frame.max() * 1e-4


def test_silence_produces_no_nan_or_inf(signals):
    y = signals["silence"]
    assert np.all(np.isfinite(np.abs(dsp.stft(y))))
    assert np.all(np.isfinite(dsp.melspectrogram(y, SR)))
    assert np.all(np.isfinite(dsp.mfcc(y, SR)))
    assert np.all(np.isfinite(dsp.amplitude_to_db(np.abs(dsp.stft(y)))))


def test_shapes_are_consistent(signals):
    S = dsp.stft(signals["clip"])
    n_frames = S.shape[1]
    assert S.shape[0] == STFT.n_fft // 2 + 1
    assert dsp.melspectrogram(signals["clip"], SR).shape == (128, n_frames)
    assert dsp.mfcc(signals["clip"], SR).shape == (40, n_frames)
