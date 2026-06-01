"""Phase 4: the Qt-free session core and coordinate mapping."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from python_ref.annotation import ExtractionMode
from python_ref.gui import coords
from python_ref.gui.session import SpectrogramSession
from python_ref.params import STFT

SR = 44_100


@pytest.fixture
def session() -> SpectrogramSession:
    t = np.linspace(0.0, 2.0, SR * 2, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
    return SpectrogramSession(y, SR, source_file="tone.wav")


def test_coord_roundtrips():
    assert coords.time_to_frame(coords.frame_to_time(100, SR, STFT), SR, STFT) == 100
    assert coords.hz_to_bin(coords.bin_to_hz(64, SR, STFT), SR, STFT) == 64


def test_magnitude_db_shape(session):
    db = session.magnitude_db()
    assert db.shape == session.stft.shape
    assert np.all(np.isfinite(db))


def test_add_rectangle_creates_feathered_layer(session):
    layer = session.add_rectangle(0.5, 1.5, 800, 1200, label="tone", confidence=0.8)
    assert session.layers == [layer]
    assert layer.confidence == 0.8
    assert 0.0 < layer.mask.max() <= 1.0  # feathered, not pure binary edges


def test_reconstruct_layer_isolates_band(session):
    # A band around 1 kHz keeps most energy; a disjoint band keeps almost none.
    on = session.add_rectangle(0.2, 1.8, 900, 1100, label="on")
    off = session.add_rectangle(0.2, 1.8, 5000, 6000, label="off")
    assert np.sum(session.reconstruct_layer(on) ** 2) > 50 * np.sum(
        session.reconstruct_layer(off) ** 2
    )


def test_to_annotations_are_sample_accurate(session):
    session.add_rectangle(0.5, 1.5, 800, 1200, label="tone", confidence=0.6)
    ann = session.to_annotations()[0]
    assert ann.label == "tone" and ann.confidence == 0.6
    assert ann.start_sample < ann.end_sample <= len(session.y)
    assert ann.freq_min_hz < ann.freq_max_hz


def test_export_writes_clips_sidecar_and_csv(tmp_path, session):
    session.add_rectangle(0.5, 1.5, 800, 1200, label="tone",
                          extraction_mode=ExtractionMode.POSITIVE)
    anns = session.export(tmp_path, pad_to=SR * 2)

    assert (tmp_path / "tone_000_raw.wav").exists()
    assert (tmp_path / "tone_000_padded.wav").exists()
    assert (tmp_path / "tone.dspws.json").exists()
    assert (tmp_path / "tone_annotations.csv").exists()
    assert anns[0].exported_clip_raw is not None


def test_qt_shell_imports():
    # The Qt module must import without a display (no QApplication at import time).
    assert importlib.import_module("python_ref.gui.app") is not None
