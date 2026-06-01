"""Shared fixtures: the six reference signals, generated on demand."""

from __future__ import annotations

import numpy as np
import pytest

from tests.reference_signals import generate as refgen

SR = refgen.SR


@pytest.fixture(scope="session")
def signals() -> dict[str, np.ndarray]:
    return {
        "sine": refgen.sine_1khz(),
        "chirp": refgen.chirp_20hz_20khz(),
        "noise": refgen.white_noise(),
        "impulse": refgen.impulse(),
        "silence": refgen.silence(),
        "clip": refgen.clip_test(),
    }
