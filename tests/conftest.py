"""Shared pytest fixtures: the reference signals.

Signals are generated on demand (and cached for the session) so the test suite is
self-contained — no need to run the generator script first.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.reference_signals import generate as refgen

SR = refgen.SR


@pytest.fixture(scope="session")
def signals() -> dict[str, np.ndarray]:
    """All six reference signals as float arrays, keyed by short name."""
    return {
        "sine": refgen.sine_1khz(),
        "chirp": refgen.chirp_20hz_20khz(),
        "noise": refgen.white_noise(),
        "impulse": refgen.impulse(),
        "silence": refgen.silence(),
        "clip": refgen.clip_test(),
    }
