"""DSP reference functions.

Thin, parameter-locked wrappers around librosa. Every function here defines the
canonical output the native core must reproduce within ``params.TOLERANCE``. Keep
them thin: no clever optimizations, no deviation from librosa semantics. Where a
default differs from librosa's, it is set explicitly from :mod:`python_ref.params`
so the behavior is visible and locked.
"""

from .spectral import (
    amplitude_to_db,
    istft,
    melspectrogram,
    mfcc,
    stft,
)

__all__ = [
    "stft",
    "istft",
    "amplitude_to_db",
    "melspectrogram",
    "mfcc",
]
