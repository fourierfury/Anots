"""Magnetic ridge follower and soft tube mask (design §6.5)."""

from __future__ import annotations

import numpy as np

from python_ref.annotation import ridge_path, tube_mask
from python_ref.annotation.ridge import TUBE_WIDTH


def _curved_ridge(n_freq=128, n_frames=200, amp=30, base=64, sigma=1.5):
    """A bright sinusoidal ridge over a dark background; returns (energy, true_freqs)."""
    t = np.arange(n_frames)
    true = (base + amp * np.sin(2 * np.pi * t / n_frames)).round().astype(int)
    energy = np.full((n_freq, n_frames), 0.01)
    f = np.arange(n_freq)[:, None]
    energy += np.exp(-((f - true[None, :]) ** 2) / (2 * sigma**2))  # gaussian ridge
    return energy, true


# --- path following ----------------------------------------------------------

def test_path_tracks_curved_ridge():
    energy, true = _curved_ridge()
    start = (0, int(true[0]))
    end = (energy.shape[1] - 1, int(true[-1]))
    path = ridge_path(energy, start, end)

    assert path[0] == start
    assert path[-1] == end
    assert len(path) == energy.shape[1]                      # one bin per frame
    assert [t for t, _ in path] == list(range(energy.shape[1]))  # contiguous frames

    recovered = np.array([f for _, f in path])
    assert np.max(np.abs(recovered - true)) <= 1             # tracks within a bin


def test_path_respects_max_jump():
    energy, true = _curved_ridge()
    path = ridge_path(energy, (0, int(true[0])), (199, int(true[-1])), k=20)
    freqs = np.array([f for _, f in path])
    assert np.all(np.abs(np.diff(freqs)) <= 20)


def test_smoothness_penalty_straightens_path():
    # Two bright bins with a dark gap: high lambda should refuse to chase the detour.
    energy = np.full((64, 50), 0.01)
    energy[10, :] = 1.0          # a straight bright line at bin 10
    energy[40, 25] = 5.0         # one very bright off-ridge bin
    start, end = (0, 10), (49, 10)
    straight = ridge_path(energy, start, end, lam=5.0)
    freqs = np.array([f for _, f in straight])
    assert freqs.max() - freqs.min() <= 1   # stays on the line, ignores the lure


def test_endpoints_order_independent():
    energy, true = _curved_ridge()
    fwd = ridge_path(energy, (0, int(true[0])), (199, int(true[-1])))
    rev = ridge_path(energy, (199, int(true[-1])), (0, int(true[0])))
    assert rev[0] == (199, int(true[-1]))   # path runs from the given start
    assert rev[-1] == (0, int(true[0]))
    assert set(fwd) == set(rev)             # same bins, opposite traversal


def test_same_point_returns_single_bin():
    energy, _ = _curved_ridge()
    assert ridge_path(energy, (5, 64), (5, 64)) == [(5, 64)]


def test_k_larger_than_band_count():
    # The scalogram has only ~9 bands; default K (20) exceeds n_freq. Must not crash
    # and the path must stay in range (regression for the GUI-on-scalogram bug).
    energy = np.random.default_rng(0).random((9, 200)) + 0.01
    path = ridge_path(energy, (0, 4), (199, 6), k=20)
    assert len(path) == 200
    assert all(0 <= f < 9 for _, f in path)
    assert path[0] == (0, 4) and path[-1] == (199, 6)


# --- tube mask ---------------------------------------------------------------

def test_tube_mask_peaks_on_path_and_falls_off():
    path = [(t, 30) for t in range(10)]
    mask = tube_mask((64, 10), path, width=8)
    assert np.allclose(mask[30, :], 1.0)                 # cos^2(0) = 1 on the path
    assert np.allclose(mask[30 + 8, :], 0.0, atol=1e-12)  # cos^2(pi/2) = 0 at the edge
    assert np.all(mask[30 + 9, :] == 0.0)                 # nothing beyond the width
    assert mask.max() <= 1.0 and mask.min() >= 0.0


def test_tube_mask_default_width():
    path = [(t, 30) for t in range(5)]
    mask = tube_mask((64, 5), path)
    assert np.isclose(mask[30 + TUBE_WIDTH, 0], 0.0, atol=1e-12)
    assert mask[30, 0] == 1.0


def test_tube_mask_zero_width_is_one_bin():
    mask = tube_mask((64, 5), [(t, 30) for t in range(5)], width=0)
    assert np.count_nonzero(mask) == 5
    assert np.allclose(mask[30, :], 1.0)


# --- session integration -----------------------------------------------------

def test_session_add_ridge(signals):
    from python_ref.gui.session import SpectrogramSession

    sess = SpectrogramSession(signals["chirp"], 48000, source_file="chirp.wav")
    layer = sess.add_ridge(0.1, 2000, 0.5, 8000, label="sweep", width=6)
    assert layer in sess.layers
    assert layer.tool_used.value == "ridge"
    assert layer.mask.shape == sess.coeffs.shape
    assert 0.0 < layer.mask.max() <= 1.0

    clip = sess.reconstruct_layer(layer)   # the soft tube must invert cleanly
    assert clip.shape == sess.y.shape
    assert np.isfinite(clip).all()
