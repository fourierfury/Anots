"""The Transform seam: interface conformance, STFT reconstruction parity, and
coordinate round-trips through the transform (mirrors the direct ``coords`` tests).

These guard that the Transform/renderer split is behavior-preserving: the STFT
view must reconstruct identically to the bare ``dsp.stft``/``dsp.istft`` path it
replaced, within ``TOLERANCE``.
"""

from __future__ import annotations

import numpy as np
import pytest

from python_ref import dsp
from python_ref.annotation import ExtractionMode, rectangle_mask
from python_ref.params import STFT, TOLERANCE
from python_ref.transforms import (
    TRANSFORMS,
    ChromaTransform,
    CqtTransform,
    StftTransform,
    Transform,
)

SR = 44_100


@pytest.fixture
def tone() -> np.ndarray:
    t = np.linspace(0.0, 2.0, SR * 2, endpoint=False)
    return 0.5 * np.sin(2 * np.pi * 1000.0 * t)


REQUIRED = (
    "forward", "display_db", "reconstruct",
    "time_to_col", "col_to_time", "value_to_row", "row_to_value", "provenance",
)


@pytest.mark.parametrize("cls", TRANSFORMS.values())
def test_registry_entries_conform(cls):
    assert issubclass(cls, Transform)
    inst = cls()
    assert isinstance(inst.name, str) and inst.name
    assert inst.accuracy_class in ("exact", "approximate")
    for method in REQUIRED:
        assert callable(getattr(inst, method))


def test_stft_forward_matches_dsp(tone):
    tf = StftTransform(STFT)
    assert np.allclose(tf.forward(tone, SR), dsp.stft(tone, STFT), atol=TOLERANCE)


def test_stft_reconstruct_is_exact(tone):
    # A full (all-ones) mask must reproduce the bare istft(stft(y)) round-trip.
    tf = StftTransform(STFT)
    coeffs = tf.forward(tone, SR)
    full = np.ones(coeffs.shape, dtype=np.float64)
    via_transform = tf.reconstruct(tone, coeffs, full, sr=SR, length=len(tone))
    via_dsp = dsp.istft(dsp.stft(tone, STFT), STFT, length=len(tone))
    assert np.max(np.abs(via_transform - via_dsp)) < TOLERANCE


def test_stft_coord_roundtrips():
    tf = StftTransform(STFT)
    assert tf.time_to_col(tf.col_to_time(100, SR), SR) == 100
    assert tf.value_to_row(tf.row_to_value(64, SR), SR) == 64


def test_stft_provenance_records_locked_params():
    prov = StftTransform(STFT).provenance()
    assert prov["fft_size"] == STFT.n_fft
    assert prov["hop_length"] == STFT.hop_length
    assert prov["window_type"] == STFT.window


# --- CQT (Option A: select on CQT, reconstruct through the exact STFT engine) ---


@pytest.fixture
def harmonic() -> np.ndarray:
    sr = 22_050
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440.0 * t)
            + 0.3 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float64)


def test_cqt_forward_shape(harmonic):
    tf = CqtTransform()
    coeffs = tf.forward(harmonic, 22_050)
    assert coeffs.shape[0] == tf.params.n_bins
    assert np.iscomplexobj(coeffs)


def test_cqt_coord_roundtrip_is_log():
    tf = CqtTransform()
    for row in (0, 12, 36, tf.params.n_bins - 1):
        assert tf.value_to_row(tf.row_to_value(row, 22_050), 22_050) == row


def test_cqt_reconstruction_rides_exact_stft(harmonic):
    # The CQT path is built on the linear, exact STFT engine: positive + negative
    # extraction of the SAME selection must sum back to the full istft(stft(y)).
    sr = 22_050
    tf = CqtTransform()
    coeffs = tf.forward(harmonic, sr)
    mask = rectangle_mask(*coeffs.shape, freq_bins=(20, 40), frame_bins=(10, 60))

    pos = tf.reconstruct(harmonic, coeffs, mask, ExtractionMode.POSITIVE, sr, length=len(harmonic))
    neg = tf.reconstruct(harmonic, coeffs, mask, ExtractionMode.NEGATIVE, sr, length=len(harmonic))
    whole = dsp.istft(dsp.stft(harmonic, STFT), STFT, length=len(harmonic))

    interior = slice(STFT.n_fft, len(harmonic) - STFT.n_fft)
    assert np.max(np.abs((pos + neg - whole)[interior])) < TOLERANCE


def test_cqt_selection_isolates_band(harmonic):
    # A band over the 440 Hz fundamental keeps far more energy than a disjoint band.
    sr = 22_050
    tf = CqtTransform()
    coeffs = tf.forward(harmonic, sr)
    r440 = tf.value_to_row(440.0, sr)
    r110 = tf.value_to_row(110.0, sr)
    on = rectangle_mask(*coeffs.shape, freq_bins=(r440 - 2, r440 + 3), frame_bins=(10, 80))
    off = rectangle_mask(*coeffs.shape, freq_bins=(r110 - 2, r110 + 3), frame_bins=(10, 80))

    e_on = np.sum(tf.reconstruct(harmonic, coeffs, on, sr=sr, length=len(harmonic)) ** 2)
    e_off = np.sum(tf.reconstruct(harmonic, coeffs, off, sr=sr, length=len(harmonic)) ** 2)
    assert e_on > 50 * e_off


def test_cqt_provenance_carries_both_view_and_recon_params():
    prov = CqtTransform().provenance()
    assert prov["cqt_bins_per_octave"] == 12          # the view
    assert prov["fft_size"] == STFT.n_fft             # the (true) reconstruction engine


# --- Chroma (pitch-class comb, Option 1 metadata) ---


@pytest.fixture
def octaves_of_A() -> np.ndarray:
    # Pitch class A at three octaves + a C distractor (matches the deciding simulation).
    sr = 22_050
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
    return (0.4 * np.sin(2 * np.pi * 220.0 * t)
            + 0.4 * np.sin(2 * np.pi * 440.0 * t)
            + 0.4 * np.sin(2 * np.pi * 880.0 * t)
            + 0.4 * np.sin(2 * np.pi * 261.63 * t)).astype(np.float64)


def test_chroma_forward_shape(octaves_of_A):
    coeffs = ChromaTransform().forward(octaves_of_A, 22_050)
    assert coeffs.shape[0] == 12


def test_chroma_reconstruction_rides_exact_stft(octaves_of_A):
    sr = 22_050
    tf = ChromaTransform()
    coeffs = tf.forward(octaves_of_A, sr)
    mask = np.zeros((12, coeffs.shape[1]))
    mask[9, :] = 1.0  # pitch class A
    pos = tf.reconstruct(octaves_of_A, coeffs, mask, ExtractionMode.POSITIVE, sr, length=len(octaves_of_A))
    neg = tf.reconstruct(octaves_of_A, coeffs, mask, ExtractionMode.NEGATIVE, sr, length=len(octaves_of_A))
    whole = dsp.istft(dsp.stft(octaves_of_A, STFT), STFT, length=len(octaves_of_A))
    interior = slice(STFT.n_fft, len(octaves_of_A) - STFT.n_fft)
    assert np.max(np.abs((pos + neg - whole)[interior])) < TOLERANCE


def test_chroma_selection_spans_octaves_and_drops_other_classes(octaves_of_A):
    # Selecting "A" keeps 220/440/880 Hz; the C distractor (261.6) is removed.
    sr = 22_050
    tf = ChromaTransform()
    coeffs = tf.forward(octaves_of_A, sr)
    mask = np.zeros((12, coeffs.shape[1]))
    mask[9, :] = 1.0
    pos = tf.reconstruct(octaves_of_A, coeffs, mask, sr=sr, length=len(octaves_of_A))

    spec = np.abs(np.fft.rfft(pos))
    f = np.fft.rfftfreq(len(pos), 1.0 / sr)

    def amp(f0):
        return spec[np.argmin(np.abs(f - f0))]

    assert amp(440) > 1000 and amp(880) > 1000 and amp(220) > 500  # all A octaves kept
    assert amp(261.63) < 0.05 * amp(440)                            # C dropped


def test_chroma_value_to_row_is_pitch_class():
    tf = ChromaTransform()
    assert tf.value_to_row(440.0, 22_050) == 9   # A
    assert tf.value_to_row(261.63, 22_050) == 0  # C


def test_chroma_layer_params_records_pitch_classes():
    tf = ChromaTransform()
    mask = np.zeros((12, 10))
    mask[9, :] = 1.0  # A
    mask[0, :] = 1.0  # C
    assert tf.layer_params(mask, 22_050) == {"pitch_classes": ["C", "A"]}


def test_chroma_freq_fields_are_na_sentinel():
    # row_to_value returns the N/A sentinel (0): a pitch class has no single Hz.
    assert ChromaTransform().row_to_value(9, 22_050) == 0.0
