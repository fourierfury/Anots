"""Phase 3: masks, reconstruction, padding, and persistence."""

from __future__ import annotations

import numpy as np
import pytest

from python_ref import dsp
from python_ref.annotation import (
    Annotation,
    DatasetProfile,
    ExtractionMode,
    PaddingMode,
    feather,
    pad,
    preset_samples,
    read_sidecar,
    reconstruct,
    reconstruct_from_signal,
    rectangle_mask,
    write_csv,
    write_sidecar,
)
from python_ref.params import TOLERANCE
from tests.conftest import SR


# --- masks -------------------------------------------------------------------

def test_rectangle_mask_region():
    m = rectangle_mask(64, 100, freq_bins=(10, 20), frame_bins=(5, 50))
    assert m.shape == (64, 100)
    assert m.sum() == 10 * 45
    assert m[15, 25] == 1.0 and m[0, 0] == 0.0


def test_feather_softens_edges_only():
    m = rectangle_mask(64, 64, freq_bins=(16, 48), frame_bins=(16, 48))
    f = feather(m, taper_bins=8)
    assert f.max() <= 1.0 and f.min() == 0.0
    assert f[32, 32] == pytest.approx(1.0)        # deep interior untouched
    assert 0.0 < f[16, 32] < 1.0                  # boundary ramped
    assert f[0, 0] == 0.0                         # outside stays zero


# --- reconstruction ----------------------------------------------------------

def test_positive_and_negative_extraction_are_complementary(signals):
    # istft is linear, so positive + negative reconstruction recovers the whole signal.
    y = signals["chirp"]
    S = dsp.stft(y)
    mask = rectangle_mask(*S.shape, freq_bins=(50, 150), frame_bins=(20, 200))

    pos = reconstruct(S, mask, ExtractionMode.POSITIVE, length=len(y))
    neg = reconstruct(S, mask, ExtractionMode.NEGATIVE, length=len(y))
    whole = dsp.istft(S, length=len(y))

    n_fft = 1024
    interior = slice(n_fft, len(y) - n_fft)
    assert np.max(np.abs((pos + neg - whole)[interior])) < TOLERANCE


def test_all_ones_mask_returns_signal(signals):
    y = signals["sine"]
    out = reconstruct_from_signal(y, np.ones(dsp.stft(y).shape))
    n_fft = 1024
    interior = slice(n_fft, len(y) - n_fft)
    assert np.max(np.abs(out[interior] - y[interior])) < TOLERANCE


# --- padding -----------------------------------------------------------------

def test_pad_placement_modes():
    y = np.ones(100)
    target = 250

    end, rec = pad(y, target, PaddingMode.END, sr=SR)
    assert len(end) == target and rec.signal_offset_samples == 0 and end[-1] == 0.0

    start, rec = pad(y, target, PaddingMode.START, sr=SR)
    assert start[0] == 0.0 and rec.signal_offset_samples == 150

    center, rec = pad(y, target, PaddingMode.CENTER, sr=SR)
    assert rec.signal_offset_samples == 75
    assert np.array_equal(center[75:175], y)


def test_pad_rejects_target_shorter_than_signal():
    with pytest.raises(ValueError):
        pad(np.ones(100), 50)


def test_padding_preserves_signal_and_adds_zeros():
    # Time-domain padding: original samples untouched, remainder exactly silent.
    y = np.sin(2 * np.pi * 440.0 * np.arange(4096) / SR)
    padded, rec = pad(y, 8192, PaddingMode.END, sr=SR)
    assert np.array_equal(padded[:4096], y)
    assert np.all(padded[4096:] == 0.0)
    assert rec.padded_duration_ms == pytest.approx(8192 / SR * 1000.0)


def test_preset_samples():
    assert preset_samples("yamnet", SR) == round(0.96 * SR)


# --- persistence -------------------------------------------------------------

def _annotation(**over) -> Annotation:
    base = dict(
        source_file="rec_001.wav",
        sample_rate=SR,
        start_sample=22_050,
        end_sample=66_150,
        label="event",
        label_class="generic",
        confidence=0.7,
        created_at="2026-06-01T00:00:00.000+00:00",
    )
    base.update(over)
    return Annotation(**base)


def test_annotation_derived_times():
    a = _annotation()
    assert a.start_s == pytest.approx(0.5)
    assert a.end_ms == pytest.approx(1500.0)
    assert a.to_dict()["start_us"] == pytest.approx(500_000.0)


def test_sidecar_roundtrip(tmp_path):
    audio = tmp_path / "rec_001.wav"
    profile = DatasetProfile(sample_rate=SR, n_mels=128)
    write_sidecar(audio, [_annotation(), _annotation(label="other")], profile)

    loaded_profile, anns = read_sidecar(audio)
    assert loaded_profile == profile
    assert [a.label for a in anns] == ["event", "other"]
    assert anns[0].start_sample == 22_050
    assert anns[0].created_at == "2026-06-01T00:00:00.000+00:00"


def test_csv_export(tmp_path):
    path = write_csv(tmp_path / "out.csv", [_annotation()])
    text = path.read_text()
    assert "source_file" in text.splitlines()[0]
    assert "rec_001.wav" in text
