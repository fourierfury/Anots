"""Overlap detection between annotation layers (design §5.4)."""

from __future__ import annotations

import numpy as np

from python_ref.annotation import (
    Layer,
    OverlapLevel,
    classify,
    detect_overlaps,
    iou,
    rectangle_mask,
)
from python_ref.annotation.overlap import DUPLICATE_THRESHOLD, SHARE_THRESHOLD


# --- iou metric --------------------------------------------------------------

def test_iou_identical_is_one():
    m = rectangle_mask(64, 100, freq_bins=(10, 20), frame_bins=(5, 50))
    assert iou(m, m) == 1.0


def test_iou_disjoint_is_zero():
    a = rectangle_mask(64, 100, freq_bins=(0, 10), frame_bins=(0, 50))
    b = rectangle_mask(64, 100, freq_bins=(20, 30), frame_bins=(0, 50))
    assert iou(a, b) == 0.0


def test_iou_empty_union_is_zero():
    z = np.zeros((64, 100))
    assert iou(z, z) == 0.0


def test_iou_half_overlap_matches_set_formula():
    # Two equal-area boxes sharing exactly half their bins: |A&B| / |A|+|B|-|A&B|.
    a = rectangle_mask(64, 100, freq_bins=(0, 10), frame_bins=(0, 20))   # 10x20 = 200
    b = rectangle_mask(64, 100, freq_bins=(0, 10), frame_bins=(10, 30))  # 10x20 = 200
    inter = 10 * 10                                                       # frames 10..20
    union = 200 + 200 - inter
    assert iou(a, b) == inter / union


def test_iou_ignores_feather_ramp_with_threshold():
    # A bin counts as selected only above `threshold`; raising it shrinks both sets.
    a = rectangle_mask(32, 32, freq_bins=(8, 24), frame_bins=(8, 24))
    soft = a * 0.2  # everything at 0.2 — selected at threshold 0.0, not at 0.5
    assert iou(soft, soft, threshold=0.0) == 1.0
    assert iou(soft, soft, threshold=0.5) == 0.0


# --- classification ----------------------------------------------------------

def test_classify_thresholds():
    assert classify(0.0) is OverlapLevel.NONE
    assert classify(SHARE_THRESHOLD) is OverlapLevel.NONE          # strict >
    assert classify(SHARE_THRESHOLD + 0.01) is OverlapLevel.SHARE
    assert classify(0.9) is OverlapLevel.SHARE
    assert classify(DUPLICATE_THRESHOLD) is OverlapLevel.SHARE     # strict >
    assert classify(0.99) is OverlapLevel.DUPLICATE


# --- pairwise detection ------------------------------------------------------

def _layer(mask, label="evt"):
    return Layer(mask=mask, label=label)


def test_detect_overlaps_skips_disjoint_pairs():
    a = _layer(rectangle_mask(64, 100, freq_bins=(0, 10), frame_bins=(0, 50)))
    b = _layer(rectangle_mask(64, 100, freq_bins=(20, 30), frame_bins=(0, 50)))
    assert detect_overlaps([a, b]) == []


def test_detect_overlaps_all_pairs_and_indices():
    full = rectangle_mask(16, 16, freq_bins=(0, 16), frame_bins=(0, 16))
    layers = [_layer(full.copy()), _layer(full.copy()), _layer(full.copy())]
    warnings = detect_overlaps(layers)
    assert [(w.i, w.j) for w in warnings] == [(0, 1), (0, 2), (1, 2)]
    assert all(w.level is OverlapLevel.DUPLICATE for w in warnings)


def test_duplicate_risk_requires_same_label():
    full = rectangle_mask(16, 16, freq_bins=(0, 16), frame_bins=(0, 16))
    same = detect_overlaps([_layer(full.copy(), "scream"), _layer(full.copy(), "scream")])
    diff = detect_overlaps([_layer(full.copy(), "scream"), _layer(full.copy(), "siren")])
    assert same[0].is_duplicate_risk is True
    assert diff[0].is_duplicate_risk is False  # near-identical but different labels


def test_share_level_is_not_duplicate_risk():
    a = _layer(rectangle_mask(64, 100, freq_bins=(0, 10), frame_bins=(0, 20)), "x")
    b = _layer(rectangle_mask(64, 100, freq_bins=(0, 10), frame_bins=(8, 28)), "x")
    (w,) = detect_overlaps([a, b])
    assert w.level is OverlapLevel.SHARE
    assert w.is_duplicate_risk is False


# --- session integration -----------------------------------------------------

def test_session_overlaps(signals):
    from python_ref.gui.session import SpectrogramSession

    sess = SpectrogramSession(signals["chirp"], 48000, source_file="chirp.wav")
    sess.add_rectangle(0.1, 0.4, 1000, 4000, label="a")
    sess.add_rectangle(0.1, 0.4, 1000, 4000, label="a")  # same footprint + label
    sess.add_rectangle(0.6, 0.9, 8000, 12000, label="b")  # disjoint in time + freq

    warnings = sess.overlaps()
    assert len(warnings) == 1
    (w,) = warnings
    assert (w.i, w.j) == (0, 1)
    assert w.is_duplicate_risk is True
