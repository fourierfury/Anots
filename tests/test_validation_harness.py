"""The harness must be trustworthy: it polices the native core."""

from __future__ import annotations

import numpy as np
import pytest

from python_ref import dsp
from python_ref.params import TOLERANCE
from python_ref.validation import assert_matches, compare, max_abs_diff
from tests.conftest import SR


def test_identical_arrays_pass(signals):
    ref = dsp.melspectrogram(signals["sine"], SR)
    result = compare(ref, ref.copy(), label="mel")
    assert result.passed
    assert result.max_abs_diff == 0.0


def test_within_tolerance_passes(signals):
    ref = dsp.stft(signals["chirp"])
    assert assert_matches(ref, ref + TOLERANCE * 0.1, label="stft")


def test_beyond_tolerance_fails(signals):
    ref = dsp.stft(signals["chirp"])
    broken = ref.copy()
    broken[10, 10] += TOLERANCE * 100
    with pytest.raises(AssertionError):
        assert_matches(ref, broken, label="stft")


def test_shape_mismatch_is_reported():
    a, b = np.zeros((4, 4)), np.zeros((4, 5))
    assert not compare(a, b, label="shape").passed
    with pytest.raises(ValueError):
        max_abs_diff(a, b)


def test_worst_index_points_at_divergence():
    a, b = np.zeros((8, 8)), np.zeros((8, 8))
    b[3, 5] = 1.0
    assert compare(a, b).worst_index == (3, 5)


def test_failure_dumps_arrays(tmp_path):
    with pytest.raises(AssertionError):
        assert_matches(np.zeros((4, 4)), np.ones((4, 4)), label="dumped", dump_dir=tmp_path)
    assert (tmp_path / "dumped_reference.npy").exists()
    assert (tmp_path / "dumped_other.npy").exists()
