"""Smart Energy Brush: circular stamp + threshold 'select by energy' (design §6.6)."""

from __future__ import annotations

import numpy as np

from python_ref.annotation import energy_brush
from python_ref.annotation.brush import DEFAULT_RADIUS


def _flat_db(shape, value=-20.0):
    return np.full(shape, value, dtype=np.float64)


# --- circular stamp ----------------------------------------------------------

def test_stamp_is_a_disc_of_the_given_radius():
    db = _flat_db((64, 64))
    m = energy_brush(db, center=(32, 32), radius=5)
    assert m[32, 32] == 1.0
    assert m[32, 37] == 1.0 and m[32, 38] == 0.0      # radius 5 along a row
    assert m[27, 32] == 1.0 and m[26, 32] == 0.0      # radius 5 along a column
    assert m[32 + 4, 32 + 4] == 0.0                   # corner (dist ~5.66) excluded


def test_default_radius():
    db = _flat_db((64, 64))
    m = energy_brush(db, center=(20, 20))
    assert m[20, 20 + DEFAULT_RADIUS] == 1.0
    assert m[20, 20 + DEFAULT_RADIUS + 1] == 0.0


def test_stamp_clips_at_edges():
    db = _flat_db((32, 32))
    m = energy_brush(db, center=(0, 0), radius=4)      # corner brush
    assert m[0, 0] == 1.0 and m.shape == (32, 32)
    assert m[0, 4] == 1.0 and m[4, 0] == 1.0


# --- additive accumulation (a drag is many stamps) ---------------------------

def test_strokes_accumulate_without_mutating_input():
    db = _flat_db((64, 64))
    a = energy_brush(db, (10, 10), radius=3)
    b = energy_brush(db, (10, 40), radius=3, mask=a)
    assert b[10, 10] == 1.0 and b[10, 40] == 1.0       # both stamps present
    assert a[10, 40] == 0.0                            # earlier mask not mutated


# --- threshold mode: select by energy ----------------------------------------

def test_threshold_keeps_only_loud_bins():
    # A quiet field with one bright stripe; threshold mode must grab only the stripe.
    db = _flat_db((64, 64), value=-60.0)
    db[30, :] = 0.0                                     # loud row at 0 dB
    m = energy_brush(db, center=(30, 32), radius=6, threshold_db=-20.0)
    assert m[30, 32] == 1.0                             # loud bin under the brush: kept
    assert m[27, 32] == 0.0 and m[33, 32] == 0.0        # quiet bins under the brush: skipped
    assert m.sum() == np.count_nonzero(db[24:37, 26:39] >= -20.0)  # exactly the loud bins in range


def test_standard_mode_ignores_energy():
    db = _flat_db((64, 64), value=-60.0)
    db[30, :] = 0.0
    m = energy_brush(db, center=(30, 32), radius=6)     # no threshold
    assert m[27, 32] == 1.0 and m[33, 32] == 1.0        # quiet bins added too


# --- session integration -----------------------------------------------------

def test_session_add_brush(signals):
    from python_ref.gui.session import SpectrogramSession

    sess = SpectrogramSession(signals["chirp"], 48000, source_file="chirp.wav")
    stroke = [(0.1, 4000), (0.15, 5000), (0.2, 6000)]   # a short drag in (s, Hz)
    layer = sess.add_brush(stroke, label="sweep", radius=4)
    assert layer.tool_used.value == "brush"
    assert layer.mask.shape == sess.coeffs.shape
    assert 0.0 < layer.mask.max() <= 1.0

    clip = sess.reconstruct_layer(layer)
    assert clip.shape == sess.y.shape and np.isfinite(clip).all()


def test_session_brush_threshold_selects_less(signals):
    from python_ref.gui.session import SpectrogramSession

    sess = SpectrogramSession(signals["chirp"], 48000, source_file="chirp.wav")
    stroke = [(0.25, 8000)]
    broad = sess.add_brush(stroke, label="all", radius=10)
    tight = sess.add_brush(stroke, label="loud", radius=10, threshold_db=-30.0, taper_bins=0)
    # Thresholding to the loud bins selects no more than the unthresholded stamp.
    assert (tight.mask > 0).sum() <= (broad.mask > 0).sum()
