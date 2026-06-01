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


def test_export_crops_raw_clip_to_event_bounds(tmp_path):
    # A short event in a longer file must export a clip the length of the EVENT, not the
    # whole recording (design §10.5 clean clip boundaries). Regression for the dogfood bug.
    import soundfile as sf

    sr = SR
    t = np.arange(5 * sr) / sr
    y = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
    sess = SpectrogramSession(y, sr, source_file="long.wav")
    sess.add_rectangle(1.0, 2.0, 800, 1200, label="evt")  # 1 s event in a 5 s file

    anns = sess.export(tmp_path)
    clip, _ = sf.read(tmp_path / "long_000_raw.wav")
    event_len = anns[0].end_sample - anns[0].start_sample
    assert len(clip) < len(y)                                  # NOT the whole 5 s file
    assert abs(len(clip) - event_len) <= 1                     # ~the event's own length


def test_export_context_margin_widens_clip(tmp_path):
    import soundfile as sf

    sr = SR
    t = np.arange(5 * sr) / sr
    y = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
    sess = SpectrogramSession(y, sr, source_file="long.wav")
    sess.add_rectangle(2.0, 3.0, 800, 1200, label="evt")

    tight = sess.export(tmp_path / "tight")
    margin = sess.export(tmp_path / "margin", context_ms=100.0)  # ±100 ms
    t_clip, _ = sf.read(tmp_path / "tight" / "long_000_raw.wav")
    m_clip, _ = sf.read(tmp_path / "margin" / "long_000_raw.wav")
    assert len(m_clip) > len(t_clip)
    assert abs(len(m_clip) - len(t_clip) - 2 * int(0.1 * sr)) <= 2  # ~200 ms added


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


def test_add_rectangle_display_resolves_view_units():
    # On a log view, a GUI selection's y is a BAND INDEX, not Hz. add_rectangle_display
    # must select those bands — where add_rectangle (Hz) would mis-map them to band 0.
    from python_ref.transforms import ScalogramTransform

    t = np.arange(16_384) / SR
    y = 0.5 * np.sin(2 * np.pi * 3000.0 * t)
    sess = SpectrogramSession(y, SR, transform=ScalogramTransform())

    disp = sess.add_rectangle_display(0.05, 0.30, 2, 6, label="bands", taper_bins=0)
    rows = np.where(disp.mask.any(axis=1))[0]
    assert rows.min() == 2 and rows.max() == 5            # bands [2, 6) selected as drawn

    # The SAME numbers read as Hz are below the lowest band, so they collapse to nothing —
    # which is exactly why the GUI must use the display path, not the Hz path.
    hz = sess.add_rectangle(0.05, 0.30, 2, 6, label="hz", taper_bins=0)
    assert hz.mask.sum() == 0


def test_build_session_opens_named_view(tmp_path):
    import soundfile as sf

    from python_ref.gui.app import build_session

    wav = tmp_path / "tone.wav"
    t = np.arange(8192) / SR
    sf.write(wav, (0.4 * np.sin(2 * np.pi * 2000.0 * t)).astype("float32"), SR)

    sess = build_session(str(wav), "scalogram")
    assert sess.transform.name == "scalogram"
    assert sess.coeffs.shape[0] == sess.transform.params.level + 1
