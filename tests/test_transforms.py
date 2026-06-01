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
    ScalogramTransform,
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


# --- Scalogram (undecimated wavelet — genuinely exact inverse, NOT STFT-bridged) ---


@pytest.fixture
def two_tones() -> np.ndarray:
    # 800 Hz and 6 kHz: land in well-separated dyadic bands (verified in the spike).
    n = 16_384
    t = np.arange(n) / SR
    return (0.5 * np.sin(2 * np.pi * 800.0 * t)
            + 0.5 * np.sin(2 * np.pi * 6000.0 * t)).astype(np.float64)


def test_scalogram_forward_shape(two_tones):
    tf = ScalogramTransform()
    coeffs = tf.forward(two_tones, SR)
    assert coeffs.shape == (tf.params.level + 1, len(two_tones))  # bands x samples, full time res


def test_scalogram_bands_are_additive(two_tones):
    # The defining property: bands sum to the signal -> reconstruction is exact.
    tf = ScalogramTransform()
    coeffs = tf.forward(two_tones, SR)
    assert np.max(np.abs(coeffs.sum(axis=0) - two_tones)) < 1e-9


def test_scalogram_reconstruction_is_genuinely_exact(two_tones):
    # Unlike CQT/chroma (STFT-bridged, ~1e-5), the wavelet inverse is ~1e-13 AND needs
    # no interior trim — positive + negative == the WHOLE original signal, edges included.
    tf = ScalogramTransform()
    coeffs = tf.forward(two_tones, SR)
    mask = rectangle_mask(*coeffs.shape, freq_bins=(3, 6), frame_bins=(2000, 9000))
    pos = tf.reconstruct(two_tones, coeffs, mask, ExtractionMode.POSITIVE, SR, length=len(two_tones))
    neg = tf.reconstruct(two_tones, coeffs, mask, ExtractionMode.NEGATIVE, SR, length=len(two_tones))
    assert np.max(np.abs(pos + neg - two_tones)) < 1e-9   # no interior slice needed


def test_scalogram_band_isolates_frequency(two_tones):
    # Selecting the band over 6 kHz keeps the 6 kHz tone and suppresses 800 Hz.
    tf = ScalogramTransform()
    coeffs = tf.forward(two_tones, SR)
    r_hi = tf.value_to_row(6000.0, SR)
    r_lo = tf.value_to_row(800.0, SR)
    assert r_hi != r_lo
    mask = np.zeros(coeffs.shape)
    mask[r_hi, :] = 1.0
    clip = tf.reconstruct(two_tones, coeffs, mask, sr=SR, length=len(two_tones))

    spec = np.abs(np.fft.rfft(clip))
    f = np.fft.rfftfreq(len(clip), 1.0 / SR)
    amp = lambda f0: spec[np.argmin(np.abs(f - f0))]
    assert amp(6000) > 10 * amp(800)


def test_scalogram_preserves_transient_timing():
    # The view's reason to exist (design §13): an impulse stays a sharp impulse through
    # the undecimated wavelet round-trip — full time resolution, unlike an STFT hop.
    n = 8192
    y = np.zeros(n)
    y[4000] = 1.0
    tf = ScalogramTransform()
    coeffs = tf.forward(y, SR)
    full = np.ones(coeffs.shape)
    clip = tf.reconstruct(y, coeffs, full, sr=SR, length=n)
    assert int(np.argmax(np.abs(clip))) == 4000          # peak stays at the exact sample
    assert np.max(np.abs(clip - y)) < 1e-9


def test_scalogram_coord_mapping_is_monotonic():
    tf = ScalogramTransform()
    freqs = [tf.row_to_value(r, SR) for r in range(tf.params.level + 1)]
    assert freqs == sorted(freqs)                        # row 0 lowest -> rises with index
    assert tf.value_to_row(tf.row_to_value(4, SR), SR) == 4


def test_scalogram_handles_arbitrary_length():
    # Odd, non-power-of-two length must still work (reflect-pad then trim).
    y = np.random.default_rng(0).standard_normal(6789)
    tf = ScalogramTransform()
    coeffs = tf.forward(y, SR)
    assert coeffs.shape[1] == 6789
    assert np.max(np.abs(coeffs.sum(axis=0) - y)) < 1e-9


def test_scalogram_provenance_records_wavelet():
    prov = ScalogramTransform().provenance()
    assert prov["scalogram_wavelet"] == "db4"
    assert prov["scalogram_level"] == 8


def test_annotation_recon_fields_are_honest_per_view():
    # Flag-2 Option A: STFT-bridged views record real fft params; the wavelet-native
    # scalogram leaves them None (never asserts an STFT it didn't use) and carries its
    # true engine in transform_params.
    from python_ref.gui.session import SpectrogramSession

    t = np.arange(16_384) / SR
    y = (0.5 * np.sin(2 * np.pi * 800.0 * t) + 0.5 * np.sin(2 * np.pi * 6000.0 * t))

    stft_sess = SpectrogramSession(y, SR)
    stft_sess.add_rectangle(0.05, 0.30, 1000, 5000, label="e")
    a_stft = stft_sess.to_annotations()[0]
    assert a_stft.fft_size == STFT.n_fft and a_stft.window_type == STFT.window

    scal_sess = SpectrogramSession(y, SR, transform=ScalogramTransform())
    scal_sess.add_rectangle(0.05, 0.30, 1000, 5000, label="e")
    a_scal = scal_sess.to_annotations()[0]
    assert a_scal.fft_size is None and a_scal.hop_length is None and a_scal.window_type is None
    assert a_scal.transform_params["scalogram_wavelet"] == "db4"
