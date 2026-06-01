"""Anots Python reference prototype.

A librosa-based reference implementation of the Anots DSP and annotation pipeline,
and the numerical ground truth for the future native (Rust + C++) core.
"""

from . import annotation, dsp, params, validation

__all__ = ["annotation", "dsp", "params", "validation"]
__version__ = "0.0.1"
