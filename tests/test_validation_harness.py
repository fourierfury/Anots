"""Tests for the validation harness itself.

The harness is what will police the native core, so it must be trustworthy: it
passes on identical output, fails on out-of-tolerance deviation, flags shape
mismatches, and dumps arrays on failure for debugging.
"""

from __future__ import annotations

import numpy as np
import pytest

from python_ref import dsp
from python_ref.params import TOLERANCE
from python_ref.validation import assert_matches, max_abs_diff
from python_ref.validation.compare import compare
from tests.conftest import SR


def test_identical_arrays_pass(signals):
    ref = dsp.melspectrogram(signals["sine"], SR)
    result = compare(ref, ref.copy(), label="mel")
    assert result.passed
    assert result.max_abs_diff == 0.0


def test_within_tolerance_passes(signals):
    ref = dsp.stft(signals["chirp"])
    nudged = ref + (TOLERANCE * 0.1)  # below threshold
    assert assert_matches(ref, nudged, label="stft")


def test_beyond_tolerance_fails(signals):
    ref = dsp.stft(signals["chirp"])
    broken = ref.copy()
    broken[10, 10] += TOLERANCE * 100  # well beyond threshold
    with pytest.raises(AssertionError):
        assert_matches(ref, broken, label="stft")


def test_shape_mismatch_is_reported():
    a = np.zeros((4, 4))
    b = np.zeros((4, 5))
    result = compare(a, b, label="shape")
    assert not result.passed
    assert "shape mismatch" in result.message
    with pytest.raises(ValueError):
        max_abs_diff(a, b)


def test_worst_index_points_at_divergence():
    a = np.zeros((8, 8))
    b = np.zeros((8, 8))
    b[3, 5] = 1.0
    result = compare(a, b)
    assert result.worst_index == (3, 5)


def test_failure_dumps_arrays(tmp_path):
    a = np.zeros((4, 4))
    b = np.ones((4, 4))
    with pytest.raises(AssertionError):
        assert_matches(a, b, label="dumped", dump_dir=tmp_path)
    assert (tmp_path / "dumped_reference.npy").exists()
    assert (tmp_path / "dumped_other.npy").exists()
