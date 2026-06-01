"""Backwards-compatible shim.

The canonical STFT coordinate math now lives in :mod:`python_ref.coords` so it can
be shared with the transform layer without a GUI import cycle. This module re-exports
it unchanged for callers that still import ``python_ref.gui.coords``.
"""

from __future__ import annotations

from ..coords import (
    bin_to_hz,
    frame_to_time,
    hz_to_bin,
    n_freq_bins,
    time_to_frame,
)

__all__ = ["time_to_frame", "frame_to_time", "hz_to_bin", "bin_to_hz", "n_freq_bins"]
